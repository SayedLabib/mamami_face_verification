import logging
import time
from typing import Dict, List, Optional
import requests
import json

from fastapi import HTTPException

from app.config.config import Config
from app.services.verification_system.face_verification.face_verification_schema import ErrorResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FacePlusPlusManager:
    def __init__(self):
        self.config = Config()
        self.api_key = self.config.fpp_api_key
        self.api_secret = self.config.fpp_api_secret
        self.detect_url = self.config.fpp_detect
        self.compare_url = "https://api-us.faceplusplus.com/facepp/v3/compare"  # Face comparison API
        
        # Rate limiting settings
        self.max_retries = 3
        self.retry_delay = 2  # seconds
        self.last_request_time = 0

        if not all([self.api_key, self.api_secret]):
            error = ErrorResponse(status_code=500, detail="Face++ API credentials not configured")
            raise HTTPException(status_code=error.status_code, detail=error.dict())

    def _get_base_params(self) -> Dict[str, str]:
        """Get base parameters required for all API calls"""
        return {
            'api_key': self.api_key,
            'api_secret': self.api_secret
        }

    def _handle_rate_limit(self):
        """Implement rate limiting with delay between requests"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.retry_delay:
            sleep_time = self.retry_delay - time_since_last_request
            logger.info(f"Rate limiting: Sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()

    def _make_request(self, url: str, data: Dict, files: Dict = None, operation: str = "") -> Dict:
        """Make request to Face++ API with retry logic and proper error handling"""
        for attempt in range(self.max_retries):
            try:
                # Apply rate limiting
                self._handle_rate_limit()
                
                safe_data = {k: v for k, v in data.items() if k not in ['api_key', 'api_secret']}
                logger.info(f"Making {operation} request to Face++ API with data: {json.dumps(safe_data)}")
                
                response = requests.post(url, data=data, files=files)
                
                try:
                    result = response.json()
                    
                    # Check for rate limit error
                    if 'error_message' in result and 'CONCURRENCY_LIMIT_EXCEEDED' in result['error_message']:
                        if attempt < self.max_retries - 1:
                            wait_time = self.retry_delay * (attempt + 1)  # Exponential backoff
                            logger.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{self.max_retries})")
                            time.sleep(wait_time)
                            continue
                    
                    if response.status_code != 200:
                        error = ErrorResponse(
                            status_code=response.status_code,
                            detail=f"{operation} failed: {json.dumps(result)}"
                        )
                        raise HTTPException(status_code=error.status_code, detail=error.dict())
                    
                    if 'error_message' in result:
                        error = ErrorResponse(
                            status_code=400,
                            detail=f"{operation} failed: {result['error_message']}"
                        )
                        raise HTTPException(status_code=error.status_code, detail=error.dict())
                    
                    return result
                    
                except json.JSONDecodeError:
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    error = ErrorResponse(
                        status_code=500,
                        detail=f"Invalid response from Face++ API in {operation}"
                    )
                    raise HTTPException(status_code=error.status_code, detail=error.dict())
                    
            except requests.RequestException as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                error = ErrorResponse(
                    status_code=503,
                    detail=f"API request failed in {operation}: {str(e)}"
                )
                raise HTTPException(status_code=error.status_code, detail=error.dict())

    async def detect_face(self, image_data: bytes) -> Optional[str]:
        """Detect face and get face token from Face++ API with retry logic"""
        try:
            files = {
                'image_file': ('image.jpg', image_data, 'image/jpeg')
            }
            
            result = self._make_request(
                self.detect_url,
                data=self._get_base_params(),
                files=files,
                operation="detect_face"
            )
            
            if not result.get('faces'):
                logger.info("No face detected in the provided image")
                return None
                
            face_token = result['faces'][0]['face_token']
            logger.info(f"Successfully detected face: {face_token}")
            return face_token
            
        except Exception as e:
            logger.error(f"Face detection failed: {str(e)}")
            return None

    async def compare_faces(self, face_token1: str, face_token2: str) -> float:
        """Compare two face tokens using Face++ API and return confidence score"""
        try:
            data = {
                **self._get_base_params(),
                'face_token1': face_token1,
                'face_token2': face_token2
            }
            
            result = self._make_request(
                self.compare_url,
                data=data,
                operation="compare_faces"
            )
            
            if result and 'confidence' in result:
                confidence = float(result['confidence'])
                logger.info(f"Face comparison confidence: {confidence}")
                return confidence
                
            error = ErrorResponse(
                status_code=500,
                detail="Face comparison failed: Invalid response format"
            )
            raise HTTPException(status_code=error.status_code, detail=error.dict())
            
        except HTTPException:
            raise
        except Exception as e:
            error = ErrorResponse(
                status_code=500,
                detail=f"Face comparison failed: {str(e)}"
            )
            logger.error(error.detail)
            raise HTTPException(status_code=error.status_code, detail=error.dict())