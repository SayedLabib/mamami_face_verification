import logging
from typing import Union

from fastapi import APIRouter, HTTPException, UploadFile, File, Query

from app.services.verification_system.face_verification.face_verification import FaceVerification
from app.services.verification_system.face_verification.face_verification_schema import VerificationResponse, ErrorResponse

# Add the OCR import with proper indentation (at module level)
try:
    from app.services.ocr_service.ocr_manager import OCRManager
    logging.getLogger(__name__).info("OCR Manager imported successfully")
except Exception as e:
    logging.getLogger(__name__).error(f"Failed to import OCR Manager: {str(e)}")
    OCRManager = None

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

@router.post("/upload",
    response_model=VerificationResponse,
    summary="Upload and verify face image",
    description="Upload a face image and check if it matches any previously stored faces"
)
async def upload_face_image(
    file: UploadFile = File(..., description="Face image file to upload and verify"),
    session_id: str = Query("default", description="Session ID for grouping faces")
) -> VerificationResponse:
    """
    Upload a face image and verify against stored faces.
    
    - **file**: Image file containing a face
    - **session_id**: Optional session identifier for grouping related faces
    
    Returns verification results including confidence scores and match status.
    """
    try:
        # Validate file type
        validate_file_upload(file, "Face image")
            
        # Read image data
        image_data = await file.read()
        if not image_data:
            raise HTTPException(status_code=400, detail="Empty image file")
            
        # Create face verification instance
        face_verifier = FaceVerification()
        
        # Process the uploaded face
        result = await face_verifier.verify_face(image_data, session_id)
        
        return VerificationResponse(**result)
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error processing face upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compare", 
    response_model=VerificationResponse,
    summary="Compare NID/Passport image with a face image",
    description="Compare a face image from NID/Passport document with another face image and extract name from the document"
)
async def compare_face_images(
    nid_passport_image: UploadFile = File(..., description="NID/Passport image (first image)"),
    face_image: UploadFile = File(..., description="Face image to compare with (second image)")
) -> VerificationResponse:
    """
    Compare NID/Passport image with a face image and extract name from the document.
    
    - **nid_passport_image**: NID/Passport image file (must contain both face and text)
    - **face_image**: Face image file to compare with
    
    Returns:
    - **status**: "success" if comparison completed
    - **confidence**: Confidence score for the face match
    - **is_duplicate**: Boolean indicating if faces match above threshold
    - **extracted_name**: Name extracted from the NID/Passport document
    """
    try:
        # Validate file types
        validate_file_upload(nid_passport_image, "NID/Passport image")
        validate_file_upload(face_image, "Face image")
            
        # Read image data
        nid_passport_data = await nid_passport_image.read()
        face_image_data = await face_image.read()
        
        # Validate file sizes after reading
        if len(nid_passport_data) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413, 
                detail=f"NID/Passport image is too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        if len(face_image_data) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413, 
                detail=f"Face image is too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        if not nid_passport_data or not face_image_data:
            raise HTTPException(status_code=400, detail="Empty image file(s)")
            
        # Create service instances
        face_verifier = FaceVerification()
        
        # Extract name from NID/Passport image with enhanced debugging
        extracted_name = None
        if OCRManager:
            try:
                logger.info("Creating OCR manager instance...")
                ocr_manager = OCRManager()
                logger.info("Extracting name from NID/Passport image...")
                extracted_name = ocr_manager.extract_name(nid_passport_data)
                logger.info(f"OCR extraction result: {extracted_name}")
            except Exception as e:
                logger.error(f"OCR extraction failed: {str(e)}")
                extracted_name = None
        else:
            logger.error("OCR Manager not available")
        
        # Detect faces in both images
        logger.info("Detecting face in NID/Passport image...")
        face_token1 = await face_verifier.fpp_manager.detect_face(nid_passport_data)
        
        logger.info("Detecting face in comparison image...")
        face_token2 = await face_verifier.fpp_manager.detect_face(face_image_data)
        
        if not face_token1:
            raise HTTPException(status_code=400, detail="No face detected in NID/Passport image")
        if not face_token2:
            raise HTTPException(status_code=400, detail="No face detected in comparison image")
        
        # Compare the faces
        logger.info("Comparing faces...")
        confidence = await face_verifier.fpp_manager.compare_faces(face_token1, face_token2)
        is_duplicate = confidence >= face_verifier.confidence_threshold
        
        message = f"Faces match (confidence: {confidence:.2f}%)" if is_duplicate else f"Faces do not match (confidence: {confidence:.2f}%)"
        
        logger.info(f"Face comparison completed. Match: {is_duplicate}, Confidence: {confidence:.2f}%, Extracted Name: {extracted_name}")
        
        return VerificationResponse(
            status="success",
            message=message,
            confidence=confidence,
            is_duplicate=is_duplicate,
            face_token=face_token1,
            extracted_name=extracted_name
        )
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error during face comparison: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats",
    summary="Get face verification statistics",
    description="Get statistics about stored faces and sessions"
)
async def get_face_stats():
    """Get statistics about stored faces and sessions"""
    try:
        face_verifier = FaceVerification()
        total_faces = await face_verifier.get_stored_faces_count()
        session_metadata = face_verifier.storage_manager.get_session_metadata()
        
        return {
            "status": "success",
            "total_faces": total_faces,
            "total_sessions": len(session_metadata),
            "sessions": session_metadata
        }
        
    except Exception as e:
        logger.error(f"Error getting face stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/clear",
    summary="Clear stored face data",
    description="Clear stored face tokens for a specific session or all sessions"
)
async def clear_face_data(
    session_id: str = Query(default=None, description="Session ID to clear (leave empty to clear all)")
):
    """Clear stored face data for a specific session or all sessions"""
    try:
        face_verifier = FaceVerification()
        success = await face_verifier.clear_stored_faces(session_id)
        
        if success:
            message = f"Cleared data for session {session_id}" if session_id else "Cleared all stored face data"
            return {"status": "success", "message": message}
        else:
            raise HTTPException(status_code=404, detail="Session not found" if session_id else "No data to clear")
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error clearing face data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/test-ocr",
    summary="Test OCR extraction (debugging)",
    description="Test OCR name extraction from NID/Passport image independently"
)
async def test_ocr_extraction(
    image: UploadFile = File(..., description="NID/Passport image for testing OCR")
):
    """Test OCR extraction independently for debugging purposes"""
    try:
        # Validate file type and size
        validate_file_upload(image, "Test image")
        
        image_data = await image.read()
        
        # Validate file size after reading
        if len(image_data) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413, 
                detail=f"Image is too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        if OCRManager:
            try:
                ocr_manager = OCRManager()
                
                # Get raw text and extracted name
                raw_text = ocr_manager.extract_text_from_image(image_data)
                extracted_name = ocr_manager.extract_name(image_data)
                
                return {
                    "status": "success",
                    "raw_text": raw_text,
                    "extracted_name": extracted_name,
                    "ocr_available": True,
                    "raw_text_length": min(len(raw_text), 2500) if raw_text else 0
                }
            except Exception as e:
                logger.error(f"OCR test failed: {str(e)}")
                return {
                    "status": "error",
                    "message": str(e),
                    "ocr_available": True,
                    "error_type": "ocr_processing_error"
                }
        else:
            return {
                "status": "error",
                "message": "OCR Manager not available - import failed",
                "ocr_available": False
            }
            
    except Exception as e:
        logger.error(f"Test OCR endpoint failed: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "error_type": "endpoint_error"
        }