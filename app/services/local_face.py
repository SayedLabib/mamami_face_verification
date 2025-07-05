import numpy as np
import logging
import base64
from io import BytesIO
from PIL import Image
from app.models.models import FaceDetectionResponse, FaceEmbeddingResponse
import traceback

# Safely import cv2 with a fallback
try:
    import cv2
except ImportError:
    cv2 = None
    logging.warning("OpenCV (cv2) could not be imported. Local face detection may not work.")
    
# Safely import skimage
try:
    from skimage.feature import hog
    has_skimage = True
except ImportError:
    has_skimage = False
    logging.warning("scikit-image could not be imported. Some face features will be limited.")

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
        Generate a high-quality face embedding using local processing
        
        This method uses multiple approaches to create a discriminative face embedding:
        1. Facial feature extraction using OpenCV
        2. HOG features for texture and patterns
        3. LBP features for local texture patterns
        4. Color histograms for color distribution
        
        Args:
            image_data: Base64 encoded image (can be full image or cropped face)
            
        Returns:
            FaceEmbeddingResponse object with embedding results
        """
        try:
            # First detect face if it's not already cropped
            detection_result = self.detect_face(image_data)
            if detection_result.success and detection_result.face_detected and detection_result.cropped_face:
                # Use the cropped face for embedding generation
                image_data = detection_result.cropped_face
            
            # Handle various base64 formats
            if ',' in image_data:
                image_data = image_data.split(',', 1)[1]
            
            # Import required modules
            from app.core.config import settings
            
            # Check if cv2 is available
            if cv2 is None:
                logger.error("OpenCV (cv2) not available for embedding generation")
                return FaceEmbeddingResponse(
                    success=False,
                    message="OpenCV not available for embedding generation"
                )
            
            # Decode base64 image
            try:
                # Try both methods of decoding
                try:
                    # Method 1: Using OpenCV directly
                    image_bytes = base64.b64decode(image_data)
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                except Exception:
                    # Method 2: Using PIL as intermediate
                    img_bytes = BytesIO(base64.b64decode(image_data))
                    pil_img = Image.open(img_bytes)
                    img = np.array(pil_img)
                    # Convert RGB to BGR for OpenCV
                    if img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    elif img.shape[2] == 4:
                        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                
            except Exception as decode_error:
                logger.error(f"Base64 decoding error: {str(decode_error)}")
                logger.error(traceback.format_exc())
                return FaceEmbeddingResponse(
                    success=False,
                    message=f"Failed to decode image: {str(decode_error)}"
                )
                
            if img is None or img.size == 0:
                logger.error("Failed to decode image or empty image")
                return FaceEmbeddingResponse(
                    success=False,
                    message="Failed to decode image or empty image"
                )
            
            # STEP 1: Preprocess the image
            # Resize to fixed size for consistency
            try:
                img = cv2.resize(img, (128, 128))
            except Exception as e:
                logger.error(f"Error resizing image: {str(e)}")
                return FaceEmbeddingResponse(
                    success=False,
                    message=f"Error preprocessing image: {str(e)}"
                )
            
            # Convert to grayscale for feature extraction
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # STEP 2: Extract multiple feature types
            embedding_parts = []
            
            # 2.1: HOG features for shape and texture
            try:
                if has_skimage:
                    # Use scikit-image for HOG if available (better quality)
                    hog_features = hog(
                        gray,
                        orientations=9,
                        pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2),
                        visualize=False,
                        block_norm='L2-Hys'
                    )
                else:
                    # Fallback to OpenCV HOG
                    hog_descriptor = cv2.HOGDescriptor((128, 128), (16, 16), (8, 8), (8, 8), 9)
                    hog_features = hog_descriptor.compute(gray).flatten()
                    
                # Normalize HOG features
                hog_norm = np.linalg.norm(hog_features)
                if hog_norm > 0:
                    hog_features = hog_features / hog_norm
                    
                embedding_parts.append(hog_features)
            except Exception as e:
                logger.warning(f"Error extracting HOG features: {str(e)}")
                # Add zeros instead
                embedding_parts.append(np.zeros(81))  # Typical HOG feature size
            
            # 2.2: Local Binary Patterns for texture
            try:
                # Basic LBP implementation
                lbp = np.zeros_like(gray)
                for i in range(1, gray.shape[0]-1):
                    for j in range(1, gray.shape[1]-1):
                        # Get 3x3 neighborhood
                        neighborhood = gray[i-1:i+2, j-1:j+2]
                        center = neighborhood[1,1]
                        # Calculate binary pattern
                        code = 0
                        if neighborhood[0,0] >= center: code += 1
                        if neighborhood[0,1] >= center: code += 2
                        if neighborhood[0,2] >= center: code += 4
                        if neighborhood[1,2] >= center: code += 8
                        if neighborhood[2,2] >= center: code += 16
                        if neighborhood[2,1] >= center: code += 32
                        if neighborhood[2,0] >= center: code += 64
                        if neighborhood[1,0] >= center: code += 128
                        lbp[i,j] = code
                
                # Calculate LBP histogram (256 bins)
                lbp_hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
                lbp_hist = lbp_hist.astype(np.float32)
                
                # Normalize LBP histogram
                if np.sum(lbp_hist) > 0:
                    lbp_hist = lbp_hist / np.sum(lbp_hist)
                
                embedding_parts.append(lbp_hist)
            except Exception as e:
                logger.warning(f"Error extracting LBP features: {str(e)}")
                # Add zeros instead
                embedding_parts.append(np.zeros(256))
                
            # 2.3: Color histograms (in HSV space for better invariance)
            try:
                # Convert to HSV color space
                hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                
                # Calculate histogram for each channel
                hist_h = cv2.calcHist([hsv_img], [0], None, [32], [0, 180])
                hist_s = cv2.calcHist([hsv_img], [1], None, [32], [0, 256])
                hist_v = cv2.calcHist([hsv_img], [2], None, [32], [0, 256])
                
                # Normalize histograms
                if cv2.norm(hist_h) > 0:
                    hist_h = hist_h / cv2.norm(hist_h)
                if cv2.norm(hist_s) > 0:
                    hist_s = hist_s / cv2.norm(hist_s)
                if cv2.norm(hist_v) > 0:
                    hist_v = hist_v / cv2.norm(hist_v)
                    
                # Flatten and combine color histograms
                color_hist = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
                embedding_parts.append(color_hist)
            except Exception as e:
                logger.warning(f"Error extracting color features: {str(e)}")
                # Add zeros instead
                embedding_parts.append(np.zeros(96))  # 32*3 = 96
                
            # STEP 3: Combine all features into a single embedding vector
            combined_embedding = []
            for part in embedding_parts:
                combined_embedding.extend(part.tolist())
                
            # Get target vector size from settings
            target_size = settings.qdrant_vector_size
            
            # STEP 4: Adjust embedding size to match target
            if len(combined_embedding) > target_size:
                # If too long, use even sampling to reduce dimensions
                indices = np.round(np.linspace(0, len(combined_embedding) - 1, target_size)).astype(int)
                embedding = [combined_embedding[i] for i in indices]
                logger.info(f"Reduced embedding from {len(combined_embedding)} to {target_size} dimensions")
            elif len(combined_embedding) < target_size:
                # If too short, pad with zeros
                embedding = combined_embedding + [0.0] * (target_size - len(combined_embedding))
                logger.info(f"Padded embedding from {len(combined_embedding)} to {target_size} dimensions")
            else:
                embedding = combined_embedding
                
            # STEP 5: Final normalization to unit vector
            embedding_array = np.array(embedding, dtype=np.float32)
            norm = np.linalg.norm(embedding_array)
            if norm > 0:
                normalized_embedding = (embedding_array / norm).tolist()
            else:
                # Handle zero norm (shouldn't happen with our features)
                logger.warning("Zero norm embedding detected, using uniform vector")
                normalized_embedding = [1.0/np.sqrt(target_size)] * target_size
            
            logger.info(f"Generated local embedding with {len(normalized_embedding)} dimensions")
            
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
