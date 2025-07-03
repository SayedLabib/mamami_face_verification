from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid

class UserMetadata(BaseModel):
    """Simple user metadata"""
    id: str
    full_name: str
    email: str  # Changed from EmailStr to str to remove the dependency on email-validator
    additional_info: Dict[str, Any] = {}

class FaceEmbedding(BaseModel):
    """Model for face embedding data"""
    user_id: str
    embedding_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    embedding: List[float]
    metadata: UserMetadata
    created_at: datetime = Field(default_factory=datetime.now)

class FaceDetectionRequest(BaseModel):
    """Request model for face detection"""
    image_data: str  # Base64 encoded image

class FaceDetectionResponse(BaseModel):
    """Response model from face detection"""
    success: bool
    face_detected: bool
    face_box: Optional[Dict[str, int]] = None  # x, y, width, height
    message: Optional[str] = None
    cropped_face: Optional[str] = None  # Base64 encoded cropped face
    face_token: Optional[str] = None  # Face++ API face token

class FaceEmbeddingRequest(BaseModel):
    """Request model for face embedding generation"""
    image_data: str  # Base64 encoded image

class FaceEmbeddingResponse(BaseModel):
    """Response model from face embedding generation"""
    success: bool
    embedding: Optional[List[float]] = None
    message: Optional[str] = None

class EnrollmentRequest(BaseModel):
    """Request model for user enrollment"""
    user_id: str
    image_data: str  # Base64 encoded image of ID/Passport
    full_name: str
    email: str  # Changed from EmailStr to str
    additional_info: Dict[str, Any] = {}

class EnrollmentResponse(BaseModel):
    """Response model for user enrollment"""
    success: bool
    user_id: Optional[str] = None
    embedding_id: Optional[str] = None
    message: str

class VerificationRequest(BaseModel):
    """Request model for face verification"""
    image_data: str  # Base64 encoded image of face

class VerificationMatch(BaseModel):
    """Model for a single verification match"""
    user_id: str
    similarity_score: float
    full_name: str
    email: str  # Changed from EmailStr to str
    
class VerificationResponse(BaseModel):
    """Response model for face verification"""
    success: bool
    verified: bool
    matches: Optional[List[VerificationMatch]] = None
    top_match: Optional[VerificationMatch] = None
    message: str