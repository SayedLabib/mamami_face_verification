import logging

from fastapi import APIRouter, HTTPException, UploadFile, File

from app.services.verification_system.face_verification.face_verification import FaceVerification
from app.services.verification_system.face_verification.face_verification_schema import VerificationResponse, ErrorResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/face",
    tags=["face-verification"],
    responses={
        404: {"model": ErrorResponse, "description": "Not found"},
        400: {"model": ErrorResponse, "description": "Bad request"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)

# Constants for file validation
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_TYPES = ["image/jpeg", "image/jpg", "image/png", "image/bmp", "image/tiff"]

def validate_file_upload(file: UploadFile, file_name: str) -> None:
    """Validate uploaded file size and type"""
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail=f"{file_name} must be an image file")
    
    # Additional validation for specific image types
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400, 
            detail=f"{file_name} must be one of: {', '.join(ALLOWED_TYPES)}"
        )

@router.post("/compare", 
    response_model=VerificationResponse,
    summary="Compare two face images",
    description="Compare two face images to check if they match"
)
async def compare_face_images(
    image1: UploadFile = File(..., description="First image for comparison"),
    image2: UploadFile = File(..., description="Second image for comparison")
) -> VerificationResponse:
    """
    Compare two face images to check if they match.
    
    - **image1**: First image file for comparison
    - **image2**: Second image file for comparison
    
    Returns:
    - **status**: "success" if comparison completed
    - **confidence**: Confidence score for the face match
    - **is_duplicate**: Boolean indicating if faces match above threshold
    """
    try:
        # Validate file types
        validate_file_upload(image1, "First image")
        validate_file_upload(image2, "Second image")
            
        # Read image data
        image1_data = await image1.read()
        image2_data = await image2.read()
        
        # Validate file sizes after reading
        if len(image1_data) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413, 
                detail=f"First image is too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        if len(image2_data) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413, 
                detail=f"Second image is too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        if not image1_data or not image2_data:
            raise HTTPException(status_code=400, detail="Empty image file(s)")
            
        # Create service instances
        face_verifier = FaceVerification()
        
        # Detect faces in both images
        logger.info("Detecting face in first image...")
        face_token1 = await face_verifier.fpp_manager.detect_face(image1_data)
        
        logger.info("Detecting face in second image...")
        face_token2 = await face_verifier.fpp_manager.detect_face(image2_data)
        
        if not face_token1:
            raise HTTPException(status_code=400, detail="No face detected in first image")
        if not face_token2:
            raise HTTPException(status_code=400, detail="No face detected in second image")
        
        # Compare the faces
        logger.info("Comparing faces...")
        confidence = await face_verifier.fpp_manager.compare_faces(face_token1, face_token2)
        is_duplicate = confidence >= face_verifier.confidence_threshold
        
        message = f"Faces match (confidence: {confidence:.2f}%)" if is_duplicate else f"Faces do not match (confidence: {confidence:.2f}%)"
        
        logger.info(f"Face comparison completed. Match: {is_duplicate}, Confidence: {confidence:.2f}%")
        
        return VerificationResponse(
            status="success",
            message=message,
            confidence=confidence,
            is_duplicate=is_duplicate,
            face_token=face_token1
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error during face comparison: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
