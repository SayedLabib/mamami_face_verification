import logging
from typing import Dict, List, Optional

from fastapi import HTTPException

from app.services.verification_system.face_verification.face_storage_manager import FaceStorageManager
from app.services.verification_system.face_verification.face_verification_schema import FaceVerificationMatch, ErrorResponse, VerificationResponse
from app.services.verification_system.api_manager.faceplusplus_manager import FacePlusPlusManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceVerification:
    def __init__(self):
        self.confidence_threshold = 80.0  # 90% confidence threshold
        self.fpp_manager = FacePlusPlusManager()
        self.storage_manager = FaceStorageManager()

    async def compare_with_stored_faces(self, face_token: str) -> List[FaceVerificationMatch]:
        """Compare the new face token with all stored face tokens"""
        try:
            matches = []
            stored_faces = self.storage_manager.get_all_stored_faces()
            
            for session_id, face_tokens in stored_faces.items():
                for stored_token in face_tokens:
                    try:
                        # Compare the new face token with the stored one
                        confidence = await self.fpp_manager.compare_faces(face_token, stored_token)
                        
                        # If confidence is above threshold, add to matches
                        if confidence >= self.confidence_threshold:
                            matches.append(FaceVerificationMatch(
                                confidence=confidence,
                                face_token=stored_token
                            ))
                            
                    except Exception as e:
                        logger.warning(f"Error comparing faces: {str(e)}")
                        continue
            
            # Sort matches by confidence in descending order
            matches.sort(key=lambda x: x.confidence, reverse=True)
            return matches
            
        except Exception as e:
            logger.error(f"Error during face comparison: {str(e)}")
            error = ErrorResponse(status_code=500, detail=f"Error during face comparison: {str(e)}")
            raise HTTPException(status_code=error.status_code, detail=error.dict())

    async def save_face_token(self, face_token: str, session_id: str = "default") -> bool:
        """Save face token to in-memory storage"""
        try:
            saved = self.storage_manager.save_face_token(face_token, session_id)
            
            if saved:
                logger.info(f"Successfully saved face token to session {session_id}")
                return True
            else:
                logger.error("Failed to save face token to storage")
                return False
                
        except Exception as e:
            logger.error(f"Error saving face token: {str(e)}")
            return False

    async def verify_face(self, image_data: bytes, session_id: str = "default") -> Dict:
        """Main verification flow - detect face, compare with stored faces, save if new"""
        try:
            # Detect face in the uploaded image
            face_token = await self.fpp_manager.detect_face(image_data)
            if not face_token:
                error = ErrorResponse(status_code=400, detail="No face detected in image")
                raise HTTPException(status_code=error.status_code, detail=error.dict())

            logger.info(f"Face detected with token: {face_token}")
            
            # Compare with all stored faces
            matches = await self.compare_with_stored_faces(face_token)
            
            if matches:
                # Found potential duplicates
                best_match = matches[0]
                logger.info(f"Duplicate face found with confidence: {best_match.confidence}")
                
                return VerificationResponse(
                    status="duplicate_found",
                    message="Potential duplicate face detected",
                    is_duplicate=True,
                    face_token=face_token,
                    confidence=best_match.confidence,
                    matches=matches
                ).dict()
            
            # No duplicates found, save the new face token
            if not await self.save_face_token(face_token, session_id):
                error = ErrorResponse(status_code=500, detail="Failed to save face token")
                raise HTTPException(status_code=error.status_code, detail=error.dict())

            logger.info("New unique face detected and saved")
            return VerificationResponse(
                status="success",
                message="New face detected and saved successfully",
                is_duplicate=False,
                face_token=face_token
            ).dict()
            
        except Exception as e:
            logger.error(f"Error in face verification: {str(e)}")
            error = ErrorResponse(status_code=500, detail=str(e))
            raise HTTPException(status_code=error.status_code, detail=error.dict())

    async def get_stored_faces_count(self) -> int:
        """Get the total number of stored face tokens"""
        try:
            return self.storage_manager.get_total_face_count()
        except Exception as e:
            logger.error(f"Error getting stored faces count: {str(e)}")
            return 0

    async def clear_stored_faces(self, session_id: str = None) -> bool:
        """Clear stored face tokens for a specific session or all sessions"""
        try:
            return self.storage_manager.clear_session_data(session_id)
        except Exception as e:
            logger.error(f"Error clearing stored faces: {str(e)}")
            return False