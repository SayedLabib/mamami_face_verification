import numpy as np
import logging
import base64
from io import BytesIO
from PIL import Image
import cv2
from app.models.models import FaceDetectionResponse, FaceEmbeddingResponse

# Set up logging
logger = logging.getLogger(__name__)

class LocalFaceService:
    """Service for local face detection and embedding generation"""
    
    def __init__(self):
        # Load face detection model (Haar cascade is lightweight)
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            logger.info("Successfully initialized local face detection model")
        except Exception as e:
            logger.error(f"Error initializing face detection model: {str(e)}")
            self.face_cascade = None
    
    def detect_face(self, image_data: str) -> FaceDetectionResponse:
        """
        Detect a face in the provided image using OpenCV
        
        Args:
            image_data: Base64 encoded image
            
        Returns:
            FaceDetectionResponse object with detection results
        """
        try:
            # Ensure we have the face cascade before proceeding
            if self.face_cascade is None:
                return FaceDetectionResponse(
                    success=False,
                    face_detected=False,
                    message="Face detection model not initialized"
                )
                
            # Handle various base64 formats (with or without MIME type prefix)
            if ',' in image_data:
                # Format like "data:image/jpeg;base64,/9j/4AAQ..."
                image_data = image_data.split(',', 1)[1]
            
            # Decode base64 image
            try:
                image_bytes = base64.b64decode(image_data)
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as decode_error:
                logger.error(f"Base64 decoding error: {str(decode_error)}")
                return FaceDetectionResponse(
                    success=False,
                    face_detected=False,
                    message=f"Failed to decode image: {str(decode_error)}"
                )
            
            if img is None:
                logger.error("Failed to decode image")
                return FaceDetectionResponse(
                    success=False,
                    face_detected=False,
                    message="Failed to decode image"
                )
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            if len(faces) == 0:
                logger.info("No faces detected in the image")
                return FaceDetectionResponse(
                    success=True,
                    face_detected=False,
                    message="No face detected in the image"
                )
            
            # Get the largest face (assuming it's the main one)
            largest_face = None
            largest_area = 0
            
            for (x, y, w, h) in faces:
                area = w * h
                if area > largest_area:
                    largest_area = area
                    largest_face = (x, y, w, h)
            
            x, y, w, h = largest_face
            face_img = img[y:y+h, x:x+w]
            
            # Encode cropped face to base64
            _, buffer = cv2.imencode('.jpg', face_img)
            cropped_face_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return FaceDetectionResponse(
                success=True,
                face_detected=True,
                face_box={"x": x, "y": y, "width": w, "height": h},
                cropped_face=cropped_face_base64
            )
            
        except Exception as e:
            logger.error(f"Error in local face detection: {str(e)}")
            return FaceDetectionResponse(
                success=False,
                face_detected=False,
                message=f"Error: {str(e)}"
            )
    
    def generate_embedding(self, image_data: str) -> FaceEmbeddingResponse:
        """
        Generate a simple face embedding (feature vector) using OpenCV
        
        Args:
            image_data: Base64 encoded image (can be full image or cropped face)
            
        Returns:
            FaceEmbeddingResponse object with embedding results
        """
        try:
            # Handle various base64 formats (with or without MIME type prefix)
            if ',' in image_data:
                # Format like "data:image/jpeg;base64,/9j/4AAQ..."
                image_data = image_data.split(',', 1)[1]
            
            # Decode base64 image
            try:
                image_bytes = base64.b64decode(image_data)
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as decode_error:
                logger.error(f"Base64 decoding error: {str(decode_error)}")
                return FaceEmbeddingResponse(
                    success=False,
                    message=f"Failed to decode image: {str(decode_error)}"
                )
                
            if img is None:
                logger.error("Failed to decode image")
                return FaceEmbeddingResponse(
                    success=False,
                    message="Failed to decode image"
                )
            
            # Resize to fixed size for consistent embeddings
            resized = cv2.resize(img, (128, 128))
            
            # Convert to grayscale
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            
            # Generate a simple embedding using HOG features
            # This is a simplified approach - production systems would use more advanced techniques
            hog = cv2.HOGDescriptor((128, 128), (16, 16), (8, 8), (8, 8), 9)
            embedding = hog.compute(gray).flatten().tolist()
            
            # Ensure the embedding matches the expected Qdrant vector size
            from app.core.config import settings
            target_size = settings.qdrant_vector_size
            
            # Adjust embedding size if needed
            if len(embedding) < target_size:
                # Pad with zeros if too small
                logger.info(f"Padding embedding from {len(embedding)} to {target_size} dimensions")
                embedding.extend([0.0] * (target_size - len(embedding)))
            elif len(embedding) > target_size:
                # Truncate if too large
                logger.info(f"Truncating embedding from {len(embedding)} to {target_size} dimensions")
                embedding = embedding[:target_size]
            
            # Normalize the embedding to unit length (common for face embeddings)
            embedding_array = np.array(embedding)
            norm = np.linalg.norm(embedding_array)
            if norm > 0:
                embedding_array = embedding_array / norm
            
            # Convert back to list
            normalized_embedding = embedding_array.tolist()
            
            logger.info(f"Generated embedding with {len(normalized_embedding)} dimensions")
            
            return FaceEmbeddingResponse(
                success=True,
                embedding=normalized_embedding
            )
            
        except Exception as e:
            logger.error(f"Error in local embedding generation: {str(e)}")
            return FaceEmbeddingResponse(
                success=False,
                message=f"Error: {str(e)}"
            )
