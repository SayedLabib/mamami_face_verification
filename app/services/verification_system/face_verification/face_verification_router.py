import logging
from typing import Union

from fastapi import APIRouter, HTTPException, UploadFile, File, Query

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

@router.post("/upload", 
    response_model=VerificationResponse,
    summary="Upload a face image for duplicate detection",
    description="Upload a face image (NID/Passport) to detect and store if it's not a duplicate of previously uploaded images"
)
async def upload_face_image(
    file: UploadFile = File(...),
    session_id: str = Query(default="default", description="Session identifier for grouping faces")
) -> VerificationResponse:
    """
    Upload a face image for duplicate detection.
    
    - **file**: Image file containing a face (NID/Passport photo)
    - **session_id**: Optional session identifier for grouping faces
    
    Returns:
    - **status**: "success" if new face, "duplicate_found" if duplicate detected
    - **is_duplicate**: Boolean indicating if face is a duplicate
    - **confidence**: Confidence score for the best match (if duplicate)
    - **face_token**: Unique token for the detected face
    - **matches**: List of matching faces with confidence scores
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
            
        # Read image data
        image_data = await file.read()
        if not image_data:
            raise HTTPException(status_code=400, detail="Empty image file")
            
        # Create FaceVerification instance
        face_verifier = FaceVerification()
        
        # Process the image for face verification
        result = await face_verifier.verify_face(image_data, session_id)
        
        # Return verification response
        return VerificationResponse(
            status=result["status"],
            message=result["message"],
            confidence=result.get("confidence"),
            face_token=result["face_token"],
            is_duplicate=result["is_duplicate"],
            matches=result.get("matches")
        )
        
    except HTTPException as he:
        # Handle specific case for "No face detected in image"
        if "No face detected in image" in str(he.detail):
            return VerificationResponse(
                status="error",
                message="No face detected in image",
                is_duplicate=False,
                matches=None
            )
        # Re-raise other HTTP exceptions as they are already properly formatted
        raise he
    except Exception as e:
        logger.error(f"Error during face verification: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compare", 
    response_model=VerificationResponse,
    summary="Compare two face images directly",
    description="Compare two uploaded face images to check if they belong to the same person"
)
async def compare_face_images(
    file1: UploadFile = File(..., description="First face image"),
    file2: UploadFile = File(..., description="Second face image")
) -> VerificationResponse:
    """
    Compare two face images directly.
    
    - **file1**: First image file containing a face
    - **file2**: Second image file containing a face
    
    Returns:
    - **status**: "success" if comparison completed
    - **confidence**: Confidence score for the face match
    - **is_duplicate**: Boolean indicating if faces match above threshold
    """
    try:
        # Validate file types
        if not file1.content_type.startswith('image/') or not file2.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Both files must be images")
            
        # Read image data
        image_data1 = await file1.read()
        image_data2 = await file2.read()
        
        if not image_data1 or not image_data2:
            raise HTTPException(status_code=400, detail="Empty image file(s)")
            
        # Create FaceVerification instance
        face_verifier = FaceVerification()
        
        # Detect faces in both images
        face_token1 = await face_verifier.fpp_manager.detect_face(image_data1)
        face_token2 = await face_verifier.fpp_manager.detect_face(image_data2)
        
        if not face_token1:
            raise HTTPException(status_code=400, detail="No face detected in first image")
        if not face_token2:
            raise HTTPException(status_code=400, detail="No face detected in second image")
        
        # Compare the faces
        confidence = await face_verifier.fpp_manager.compare_faces(face_token1, face_token2)
        is_duplicate = confidence >= face_verifier.confidence_threshold
        
        message = "Faces match" if is_duplicate else "Faces do not match"
        
        return VerificationResponse(
            status="success",
            message=message,
            confidence=confidence,
            is_duplicate=is_duplicate,
            face_token=face_token1  # Return the first face token
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