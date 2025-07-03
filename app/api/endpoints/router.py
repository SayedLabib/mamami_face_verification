from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from typing import Dict, Any, Optional, List
import base64
import logging
import json
import os
import uuid
import socket
import httpx
from urllib.parse import urlparse
from datetime import datetime

from app.models.models import (
    EnrollmentResponse, 
    VerificationResponse
)
from app.core.config import settings
from app.api.dependencies import get_face_recognition_service, get_qdrant_service
from app.services.service import FaceRecognitionService, QdrantService

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/extract-and-store", response_model=EnrollmentResponse)
async def extract_and_store(
    image: UploadFile = File(...),
    user_id: Optional[str] = Form(None),
    full_name: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    additional_info: Optional[str] = Form(None),
    face_service: FaceRecognitionService = Depends(get_face_recognition_service)
):
    """
    API endpoint to extract a face from an ID/passport image and store its embedding
    
    Takes user information and ID/passport image, detects face,
    generates embedding, and stores in vector database
    """
    try:
        # Validate image file
        if not image.content_type.startswith('image/'):
            logger.warning(f"Invalid file type uploaded: {image.content_type}")
            return EnrollmentResponse(
                success=False,
                message=f"Invalid file type: {image.content_type}. Please upload an image file."
            )
            
        # Read and encode the image
        try:
            image_data = await image.read()
            if not image_data:
                return EnrollmentResponse(
                    success=False,
                    message="Empty image file uploaded"
                )
            encoded_image = base64.b64encode(image_data).decode('utf-8')
        except Exception as img_error:
            logger.error(f"Error processing uploaded image: {str(img_error)}")
            return EnrollmentResponse(
                success=False,
                message=f"Error processing uploaded image: {str(img_error)}"
            )
        
        # Parse additional info if provided
        additional_info_dict = {}
        if additional_info:
            try:
                additional_info_dict = json.loads(additional_info)
            except json.JSONDecodeError:
                additional_info_dict = {"info": additional_info}
        
        # Generate a UUID if user_id is not provided
        if user_id is None:
            user_id = str(uuid.uuid4())
        
        # Set default values if not provided
        full_name = full_name or "Unknown User"
        email = email or "unknown@example.com"
            
        # Process enrollment
        success, message, embedding_id = await face_service.enroll_user(
            user_id=user_id,
            image_data=encoded_image,
            full_name=full_name,
            email=email,
            additional_info=additional_info_dict
        )
        
        if not success:
            return EnrollmentResponse(
                success=False,
                message=message
            )
        
        return EnrollmentResponse(
            success=True,
            user_id=user_id,
            embedding_id=embedding_id,
            message=message
        )
        
    except Exception as e:
        logger.error(f"Error in extract and store endpoint: {str(e)}")
        return EnrollmentResponse(
            success=False,
            message=f"Error extracting and storing face: {str(e)}"
        )


@router.post("/verify-match", response_model=VerificationResponse)
async def verify_match(
    image: UploadFile = File(...),
    face_service: FaceRecognitionService = Depends(get_face_recognition_service)
):
    """
    API endpoint to check if a face matches any stored faces
    
    Takes an image file, detects face, generates embedding,
    and compares against stored face embeddings
    """
    try:
        # Validate image file
        if not image.content_type.startswith('image/'):
            logger.warning(f"Invalid file type uploaded: {image.content_type}")
            return VerificationResponse(
                success=False,
                verified=False,
                message=f"Invalid file type: {image.content_type}. Please upload an image file."
            )
            
        # Read and encode the image
        try:
            image_data = await image.read()
            if not image_data:
                return VerificationResponse(
                    success=False,
                    verified=False,
                    message="Empty image file uploaded"
                )
            encoded_image = base64.b64encode(image_data).decode('utf-8')
        except Exception as img_error:
            logger.error(f"Error processing uploaded image: {str(img_error)}")
            return VerificationResponse(
                success=False,
                verified=False,
                message=f"Error processing uploaded image: {str(img_error)}"
            )
        
        # Process verification
        success, verified, matches, top_match = await face_service.verify_face(encoded_image)
        
        if not success:
            return VerificationResponse(
                success=False,
                verified=False,
                message="Failed to process face verification"
            )
        
        return VerificationResponse(
            success=True,
            verified=verified,
            matches=matches,
            top_match=top_match,
            message="Face match found" if verified else "No face match found"
        )
        
    except Exception as e:
        logger.error(f"Error in verify match endpoint: {str(e)}")
        return VerificationResponse(
            success=False,
            verified=False,
            message=f"Error during face verification: {str(e)}"
        )


@router.get("/diagnostics")
async def system_diagnostics():
    """
    API endpoint to check system connectivity and settings
    
    Helps diagnose connection issues to external APIs and services
    """
    results = {
        "dns_check": [],
        "api_connectivity": {},
        "environment": {},
        "dns_config": {},
        "host_entries": {}
    }
    
    try:
        # Check DNS resolution
        hosts_to_check = ["api-us.faceplusplus.com", "google.com", "qdrant"]
        for host in hosts_to_check:
            try:
                ip = socket.gethostbyname(host)
                results["dns_check"].append({
                    "host": host,
                    "status": "success",
                    "ip": ip
                })
            except Exception as e:
                results["dns_check"].append({
                    "host": host,
                    "status": "failed",
                    "error": str(e)
                })
        
        # Check DNS configuration
        try:
            custom_resolv_path = '/app/dns_config/resolv.conf'
            if os.path.exists(custom_resolv_path):
                with open(custom_resolv_path, 'r') as f:
                    results["dns_config"]["custom_resolv"] = f.read()
            
            # Check if /etc/resolv.conf exists and is readable
            if os.path.exists('/etc/resolv.conf'):
                try:
                    with open('/etc/resolv.conf', 'r') as f:
                        results["dns_config"]["system_resolv"] = f.read()
                except:
                    results["dns_config"]["system_resolv"] = "File exists but cannot be read"
            else:
                results["dns_config"]["system_resolv"] = "File does not exist"
                
            # Check hosts file
            if os.path.exists('/etc/hosts'):
                try:
                    with open('/etc/hosts', 'r') as f:
                        hosts_content = f.read()
                        results["host_entries"]["content"] = hosts_content
                        results["host_entries"]["has_api_entry"] = "api-us.faceplusplus.com" in hosts_content
                except:
                    results["host_entries"]["error"] = "File exists but cannot be read"
            else:
                results["host_entries"]["error"] = "Hosts file does not exist"
        except Exception as e:
            results["dns_config"]["error"] = str(e)
        
        # Check API connectivity using our client with DNS settings
        api_url = settings.facepp_api_url
        
        # Import service class for creating client
        from app.services.service import FacePlusPlusService
        facepp_service = FacePlusPlusService()
        
        try:
            async with facepp_service._create_httpx_client() as client:
                # Face++ doesn't have a /health endpoint, just check the base API connection
                response = await client.get(api_url, headers={"User-Agent": "Diagnostics"})
                # API root will typically return 404 but that's OK for connectivity test
                results["api_connectivity"]["status"] = "success" if response.status_code in [200, 404] else "failed"
                results["api_connectivity"]["status_code"] = response.status_code
                results["api_connectivity"]["response"] = "Connection successful" if response.status_code in [200, 404] else str(response)
        except Exception as e:
            results["api_connectivity"]["status"] = "failed"
            results["api_connectivity"]["error"] = str(e)
            
            # Try direct connection as fallback
            try:
                direct_url = "https://api-us.faceplusplus.com"
                async with facepp_service._create_httpx_client() as client:
                    direct_response = await client.get(direct_url, headers={"User-Agent": "Diagnostics"})
                    results["api_connectivity"]["direct_ip_status"] = "success" if direct_response.status_code in [200, 404] else "failed"
                    results["api_connectivity"]["direct_ip_status_code"] = direct_response.status_code
            except Exception as e2:
                results["api_connectivity"]["direct_ip_error"] = str(e2)
        
        # Get environment variables (without secrets)
        results["environment"]["api_url"] = settings.facepp_api_url
        results["environment"]["fallback_enabled"] = settings.use_local_fallback
        results["environment"]["similarity_threshold"] = settings.similarity_threshold
        results["environment"]["docker"] = "/.dockerenv" in os.listdir("/")
        
        return results
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error running diagnostics: {str(e)}"
        }


@router.get("/health", status_code=200)
async def health_check(
    face_service: FaceRecognitionService = Depends(get_face_recognition_service),
    qdrant_service: QdrantService = Depends(get_qdrant_service)
):
    """
    Health check endpoint to verify all services are working
    
    Checks Face++ connectivity and Qdrant availability
    """
    health_status = {
        "status": "healthy",
        "services": {
            "face_api": {"status": "unknown"},
            "qdrant": {"status": "unknown"}
        },
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        # Check Qdrant service
        try:
            # Check if collection exists
            collections = qdrant_service.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if qdrant_service.collection_name in collection_names:
                health_status["services"]["qdrant"] = {
                    "status": "healthy",
                    "message": f"Collection '{qdrant_service.collection_name}' exists"
                }
            else:
                health_status["services"]["qdrant"] = {
                    "status": "warning",
                    "message": f"Collection '{qdrant_service.collection_name}' not found"
                }
                health_status["status"] = "degraded"
        except Exception as e:
            health_status["services"]["qdrant"] = {
                "status": "unhealthy",
                "message": str(e)
            }
            health_status["status"] = "degraded"
        
        # Check Face++ connectivity
        import asyncio
        try:
            # Use a short timeout for the health check
            async with face_service.facepp_service._create_httpx_client() as client:
                client.timeout = httpx.Timeout(5.0)
                response = await client.get(
                    f"https://api-us.faceplusplus.com/", 
                    headers={"User-Agent": "HealthCheck"}
                )
                
                if response.status_code == 200 or response.status_code == 404:
                    health_status["services"]["face_api"] = {
                        "status": "healthy",
                        "message": "Connection successful"
                    }
                else:
                    health_status["services"]["face_api"] = {
                        "status": "warning",
                        "message": f"Unexpected status code: {response.status_code}"
                    }
                    health_status["status"] = "degraded"
        except Exception as e:
            health_status["services"]["face_api"] = {
                "status": "unhealthy",
                "message": str(e)
            }
            health_status["status"] = "degraded"
            
        return health_status
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
