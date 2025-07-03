from fastapi import Depends
from app.services.service import FaceRecognitionService, FacePlusPlusService, QdrantService


# Create singleton instances to avoid re-initialization on every request
_face_recognition_service = None
_facepp_service = None
_qdrant_service = None

async def get_face_recognition_service() -> FaceRecognitionService:
    """Dependency to get the face recognition service"""
    global _face_recognition_service
    if _face_recognition_service is None:
        _face_recognition_service = FaceRecognitionService()
    return _face_recognition_service


async def get_facepp_service() -> FacePlusPlusService:
    """Dependency to get the Face++ API service"""
    global _facepp_service
    if _facepp_service is None:
        _facepp_service = FacePlusPlusService()
    return _facepp_service


async def get_qdrant_service() -> QdrantService:
    """Dependency to get the Qdrant vector database service"""
    global _qdrant_service
    if _qdrant_service is None:
        _qdrant_service = QdrantService()
    return _qdrant_service