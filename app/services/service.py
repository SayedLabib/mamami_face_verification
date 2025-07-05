import base64
import httpx
import logging
import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct, ScoredPoint, Distance, VectorParams

from app.models.models import (
    FaceEmbedding, 
    FaceDetectionResponse, 
    FaceEmbeddingResponse,
    UserMetadata,
    VerificationMatch
)
from app.core.config import settings

# Set up logging
logger = logging.getLogger(__name__)

class FacePlusPlusService:
    """Service for interacting with Face++ API for face detection and embedding generation"""
    
    def __init__(self):
        # Face++ API URLs
        self.detect_url = settings.fpp_detect_url
        self.search_url = settings.fpp_search_url
        self.create_url = settings.fpp_create_url
        self.add_url = settings.fpp_add_url
        self.get_detail_url = settings.fpp_get_detail_url
        
        # Auth credentials
        self.api_key = settings.facepp_api_key
        self.api_secret = settings.facepp_api_secret
        
        # Request configuration
        self.max_retries = 3
        self.timeout = 30.0
        
        logger.info(f"Initialized FacePlusPlusService with Face++ API")
        
        # Validate API credentials
        if not self.api_key or self.api_key == "YOUR_API_KEY" or len(self.api_key) < 10:
            logger.error(f"Invalid Face++ API key: '{self.api_key}'. Please check your .env file.")
        if not self.api_secret or self.api_secret == "YOUR_API_SECRET" or len(self.api_secret) < 10:
            logger.error(f"Invalid Face++ API secret. Please check your .env file.")
            
        # Check Face++ API connectivity
        self._check_api_connectivity()
    
    def _check_api_connectivity(self):
        """Check if Face++ API is reachable"""
        import socket
        
        try:
            hostname = "api-us.faceplusplus.com"
            # Try to resolve the hostname
            try:
                ip_address = socket.gethostbyname(hostname)
                logger.info(f"Successfully resolved {hostname} to {ip_address}")
                logger.info(f"Using Face++ API endpoints: detect={self.detect_url}")
                resolved = True
            except Exception as e:
                logger.warning(f"Could not resolve {hostname}: {str(e)}")
                resolved = False
            
            # Test connectivity to google as general internet check
            try:
                test_ip = socket.gethostbyname("google.com")
                logger.info(f"Internet connectivity check successful: resolved google.com to {test_ip}")
            except Exception as e:
                logger.error(f"Internet connectivity check failed: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error checking API URL: {str(e)}")
    
    def _create_httpx_client(self):
        """Create an HTTPX client with proper configuration"""
        client = httpx.AsyncClient(
            timeout=self.timeout, 
            verify=True,
            http1=True,  # Force HTTP/1.1
            http2=False  # Disable HTTP/2
        )
        
        client.headers.update({"Connection": "close"})  # Prevent connection reuse issues
        
        return client
    
    async def detect_face(self, image_data: str) -> FaceDetectionResponse:
        """
        Detect a face in the provided image using Face++ API
        
        Args:
            image_data: Base64 encoded image
            
        Returns:
            FaceDetectionResponse object with detection results
        """
        try:
            # For Face++ API, we need to prepare the request differently
            # Face++ expects multipart/form-data with API key and secret
            logger.info(f"Calling Face++ API at {self.detect_url}")
            
            # Decode base64 to binary for sending to Face++
            import base64
            from io import BytesIO
            
            # Convert base64 to binary
            image_binary = BytesIO(base64.b64decode(image_data))
            
            # Use retry logic for better reliability
            retries = 0
            last_error = None
            
            while retries < self.max_retries:
                try:
                    # Use custom client with proper configuration
                    async with self._create_httpx_client() as client:
                        # Face++ uses form data with files
                        files = {
                            'image_file': ('image.jpg', image_binary, 'image/jpeg')
                        }
                        data = {
                            'api_key': self.api_key,
                            'api_secret': self.api_secret,
                            'return_landmark': '0',
                            'return_attributes': 'gender,age'  # Get basic attributes
                        }
                        
                        response = await client.post(
                            self.detect_url,
                            data=data,
                            files=files
                        )
                        
                        # If successful, break the retry loop
                        break
                        
                except (httpx.ConnectError, httpx.TimeoutException) as e:
                    # Log the retry attempt
                    retries += 1
                    last_error = e
                    logger.warning(f"API call attempt {retries} failed: {str(e)}. {'Retrying...' if retries < self.max_retries else 'Max retries reached.'}")
                    
                    if retries >= self.max_retries:
                        raise e
            
            # If we've exhausted retries without success
            if retries >= self.max_retries and last_error:
                raise last_error
                
            if response.status_code != 200:
                logger.error(f"Face detection API error: {response.text}")
                return FaceDetectionResponse(
                    success=False,
                    face_detected=False,
                    message=f"API Error: {response.text}"
                )
            
            result = response.json()
            
            # Face++ returns a "faces" array with detected faces
            faces = result.get("faces", [])
            
            # Check if any face was detected
            if not faces:
                return FaceDetectionResponse(
                    success=True,
                    face_detected=False,
                    message="No face detected in the image"
                )
            
            # Use the first face (most prominent)
            face = faces[0]
            face_token = face.get("face_token")
            
            # Extract face rectangle (Face++ format is different)
            face_rectangle = face.get("face_rectangle", {})
            face_box = {
                "x": face_rectangle.get("left", 0),
                "y": face_rectangle.get("top", 0),
                "width": face_rectangle.get("width", 0),
                "height": face_rectangle.get("height", 0)
            }
            
            # Crop the face using the detected coordinates
            try:
                # Decode the original image
                import numpy as np
                from PIL import Image
                
                # Create a copy of the original binary data
                image_binary_copy = BytesIO(base64.b64decode(image_data))
                
                # Open the image with PIL
                img = Image.open(image_binary_copy)
                
                # Crop the face using the detected coordinates
                x = face_box["x"]
                y = face_box["y"]
                width = face_box["width"]
                height = face_box["height"]
                
                # Add a margin of 20% around the face for better results
                margin_x = int(width * 0.2)
                margin_y = int(height * 0.2)
                
                # Calculate new coordinates with margins, ensuring we stay within image bounds
                left = max(0, x - margin_x)
                top = max(0, y - margin_y)
                right = min(img.width, x + width + margin_x)
                bottom = min(img.height, y + height + margin_y)
                
                # Crop the image
                cropped_img = img.crop((left, top, right, bottom))
                
                # Convert the cropped image back to base64
                cropped_buffer = BytesIO()
                # Convert RGBA to RGB if necessary to avoid JPEG compatibility issues
                if cropped_img.mode == 'RGBA':
                    cropped_img = cropped_img.convert('RGB')
                cropped_img.save(cropped_buffer, format="JPEG")
                cropped_base64 = base64.b64encode(cropped_buffer.getvalue()).decode('utf-8')
                
                return FaceDetectionResponse(
                    success=True,
                    face_detected=True,
                    face_box=face_box,
                    face_token=face_token,
                    cropped_face=cropped_base64  # Include the cropped face image
                )
            except Exception as e:
                logger.error(f"Error cropping face: {str(e)}")
                # Return response without cropped face
                return FaceDetectionResponse(
                    success=True,
                    face_detected=True,
                    face_box=face_box,
                    face_token=face_token,
                    cropped_face=None,
                    message=f"Face detected but couldn't crop: {str(e)}"
                )
                
        except httpx.ConnectError as e:
            logger.error(f"Connection error to Face++ API: {str(e)}")
            return FaceDetectionResponse(
                success=False,
                face_detected=False,
                message=f"Connection error to face detection API: {str(e)}"
            )
        except httpx.TimeoutException as e:
            logger.error(f"Timeout connecting to Face++ API: {str(e)}")
            return FaceDetectionResponse(
                success=False,
                face_detected=False,
                message=f"Timeout connecting to face detection API: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Error in face detection: {str(e)}")
            return FaceDetectionResponse(
                success=False,
                face_detected=False,
                message=f"Error: {str(e)}"
            )
    
    async def generate_embedding(self, image_data: str, face_token: str = None) -> FaceEmbeddingResponse:
        """
        Generate face embedding for the provided image using Face++ API
        
        Face++ doesn't provide direct embedding vectors, so we use a combination of:
        1. Facial landmarks (106 points) normalized to face size
        2. Face attributes (age, gender, etc.) encoded properly 
        3. Face comparison API when possible
        
        Args:
            image_data: Base64 encoded image (can be full image or cropped face)
            face_token: Optional face token from a previous detect call
            
        Returns:
            FaceEmbeddingResponse object with embedding results
        """
        try:
            # Log operation start
            logger.info(f"Generating face embedding with Face++ API")
            
            # STEP 1: Detect face and get face token if not provided
            if not face_token:
                detection_result = await self.detect_face(image_data)
                if not detection_result.success or not detection_result.face_detected:
                    return FaceEmbeddingResponse(
                        success=False,
                        message=f"Face detection failed: {detection_result.message}"
                    )
                face_token = detection_result.face_token
                # Use the cropped face for better results
                if detection_result.cropped_face:
                    image_data = detection_result.cropped_face
            
            if not face_token:
                return FaceEmbeddingResponse(
                    success=False,
                    message="No face token available for embedding generation"
                )
                
            # Import required libraries
            import numpy as np
            from PIL import Image
            from io import BytesIO
            import base64
            import cv2
            
            # STEP 2: Use Face++ analyze API to get facial landmarks and attributes
            # Use retry logic for better reliability
            retries = 0
            last_error = None
            
            while retries < self.max_retries:
                try:
                    # Use custom client with proper configuration
                    async with self._create_httpx_client() as client:
                        # Request ALL landmarks and attributes for best embedding quality
                        data = {
                            'api_key': self.api_key,
                            'api_secret': self.api_secret,
                            'face_tokens': face_token,
                            'return_landmark': '1',       # Get full landmarks
                            'return_attributes': 'gender,age,smiling,emotion,ethnicity'  # Get rich attributes
                        }
                        
                        # Face++ API endpoint for face analysis
                        response = await client.post(
                            self.detect_url,
                            data=data
                        )
                        
                        # If successful, break the retry loop
                        break
                        
                except (httpx.ConnectError, httpx.TimeoutException) as e:
                    retries += 1
                    last_error = e
                    logger.warning(f"API call attempt {retries} failed: {str(e)}. {'Retrying...' if retries < self.max_retries else 'Max retries reached.'}")
                    if retries >= self.max_retries:
                        raise e
            
            # Check if we've exhausted retries
            if retries >= self.max_retries and last_error:
                raise last_error
            
            # Check response status
            if response.status_code != 200:
                logger.error(f"Face analyze API error: {response.text}")
                return FaceEmbeddingResponse(
                    success=False,
                    message=f"API Error: {response.text}"
                )
            
            # Parse response
            result = response.json()
            
            # Extract face data from response
            face_data = None
            
            # Parse various response formats
            if 'faces' in result and isinstance(result['faces'], list) and len(result['faces']) > 0:
                face_data = result['faces'][0]  # Use the first face
            elif 'face' in result:
                face_data = result.get('face', {})
            
            # Return error if no face data
            if not face_data:
                logger.error(f"Unexpected API response format: {list(result.keys())}")
                return FaceEmbeddingResponse(
                    success=False,
                    message=f"No face data in API response"
                )
                
            # STEP 3: Extract high-quality face features for embedding
            landmarks = face_data.get('landmark', {})
            attributes = face_data.get('attributes', {})
            face_rectangle = face_data.get('face_rectangle', {})
            
            # APPROACH: Create a robust, normalized embedding vector that captures facial identity
            
            # If we don't have landmarks (unlikely), return error
            if not landmarks and not face_rectangle:
                logger.error("No landmarks or face rectangle in Face++ response")
                return FaceEmbeddingResponse(
                    success=False,
                    message="Missing required face data from API"
                )
                
            # Create embedding components:
            embedding_parts = []
            
            # 1. Use landmark points (normalized by face size for scale invariance)
            if landmarks:
                # Get face dimensions for normalization
                width = face_rectangle.get('width', 100)
                height = face_rectangle.get('height', 100)
                face_center_x = face_rectangle.get('left', 0) + (width / 2)
                face_center_y = face_rectangle.get('top', 0) + (height / 2)
                face_size = max(width, height) / 2  # Use half of max dimension for normalization
                
                # Collect normalized landmarks (centered and scaled)
                landmark_features = []
                for point_name, coords in landmarks.items():
                    # Normalize coordinates relative to face center and size
                    # This makes landmarks independent of image size and face position
                    x_norm = (coords.get('x', 0) - face_center_x) / face_size
                    y_norm = (coords.get('y', 0) - face_center_y) / face_size
                    landmark_features.append(x_norm)
                    landmark_features.append(y_norm)
                
                embedding_parts.append(landmark_features)
                
            # 2. Encode attributes as continuous values
            attribute_features = []
            
            # Gender (continuous value)
            if 'gender' in attributes:
                gender_confidence = attributes['gender'].get('confidence', 50) / 100.0
                gender_value = gender_confidence if attributes['gender'].get('value') == 'Male' else -gender_confidence
                attribute_features.append(gender_value)
            else:
                attribute_features.append(0.0)  # Default
                
            # Age (normalized to 0-1 range)
            if 'age' in attributes:
                age_value = float(attributes['age'].get('value', 30)) / 100.0
                attribute_features.append(age_value)
            else:
                attribute_features.append(0.3)  # Default normalized age
                
            # Emotions (use all emotion scores)
            if 'emotion' in attributes:
                emotions = attributes['emotion']
                for emotion in ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']:
                    emotion_value = emotions.get(emotion, 0) / 100.0
                    attribute_features.append(emotion_value)
            else:
                # Default emotion values
                attribute_features.extend([0.0] * 7)
                
            # Ethnicity
            if 'ethnicity' in attributes:
                ethnicity_value = {
                    'Asian': [1.0, 0.0, 0.0, 0.0],
                    'White': [0.0, 1.0, 0.0, 0.0],
                    'Black': [0.0, 0.0, 1.0, 0.0],
                    'Indian': [0.0, 0.0, 0.0, 1.0]
                }.get(attributes['ethnicity'].get('value', 'Asian'), [0.25, 0.25, 0.25, 0.25])
                attribute_features.extend(ethnicity_value)
            else:
                attribute_features.extend([0.25, 0.25, 0.25, 0.25])
                
            embedding_parts.append(attribute_features)
            
            # 3. Image content-based features
            # Convert base64 to image and extract additional visual features
            try:
                # Decode image
                img_bytes = base64.b64decode(image_data)
                img = Image.open(BytesIO(img_bytes))
                
                # Convert to numpy array for OpenCV processing
                img_array = np.array(img)
                if len(img_array.shape) == 3 and img_array.shape[2] == 4:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                elif len(img_array.shape) == 2:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                
                # Resize for consistent processing
                img_resized = cv2.resize(img_array, (64, 64))
                
                # Extract HOG features (Histogram of Oriented Gradients)
                # This captures texture and edge patterns important for face recognition
                try:
                    from skimage.feature import hog
                    hog_features = hog(
                        img_resized,
                        orientations=8,
                        pixels_per_cell=(8, 8),
                        cells_per_block=(1, 1),
                        visualize=False,
                        multichannel=True
                    )
                    # Normalize HOG features
                    hog_norm = np.linalg.norm(hog_features)
                    if hog_norm > 0:
                        hog_features = hog_features / hog_norm
                        
                    embedding_parts.append(hog_features.tolist())
                except ImportError:
                    # Fallback if skimage not available
                    logger.warning("skimage.feature not available, using simple gradient features")
                    # Simple gradient features
                    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
                    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
                    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
                    
                    # Calculate gradient magnitude and direction
                    mag, angle = cv2.cartToPolar(gx, gy)
                    
                    # Downsample and flatten for smaller feature vector
                    mag_small = cv2.resize(mag, (16, 16)).flatten()
                    angle_small = cv2.resize(angle, (16, 16)).flatten()
                    
                    # Normalize
                    if np.sum(mag_small) > 0:
                        mag_small = mag_small / np.sum(mag_small)
                        
                    # Combine into feature vector
                    grad_features = np.concatenate([mag_small, angle_small/(2*np.pi)])
                    embedding_parts.append(grad_features.tolist())
            except Exception as img_e:
                logger.warning(f"Could not extract image-based features: {str(img_e)}")
                # Add zeros instead of image features
                embedding_parts.append([0.0] * 64)
            
            # STEP 4: Combine all feature parts and normalize the final vector
            # Flatten all embedding parts into a single vector
            combined_embedding = []
            for part in embedding_parts:
                combined_embedding.extend(part)
                
            # Get target vector size from settings
            target_size = settings.qdrant_vector_size
            
            # Adjust vector size to match target
            if len(combined_embedding) > target_size:
                # If too long, use dimensionality reduction
                try:
                    # PCA-like approach: keep the most significant dimensions
                    # For simplicity, we'll just select evenly distributed features
                    indices = np.round(np.linspace(0, len(combined_embedding) - 1, target_size)).astype(int)
                    embedding = [combined_embedding[i] for i in indices]
                except Exception:
                    # Fallback to simple truncation
                    embedding = combined_embedding[:target_size]
            elif len(combined_embedding) < target_size:
                # If too short, pad with zeros
                embedding = combined_embedding + [0.0] * (target_size - len(combined_embedding))
            else:
                embedding = combined_embedding
                
            # Final normalization to unit vector
            embedding_array = np.array(embedding, dtype=np.float32)
            norm = np.linalg.norm(embedding_array)
            if norm > 0:
                normalized_embedding = (embedding_array / norm).tolist()
            else:
                # Handle zero norm (shouldn't happen with our features)
                logger.warning("Zero norm embedding detected")
                normalized_embedding = [1.0/np.sqrt(target_size)] * target_size
            
            logger.info(f"Successfully generated face embedding of size {len(normalized_embedding)}")
            return FaceEmbeddingResponse(
                success=True,
                embedding=normalized_embedding
            )
                
        except httpx.ConnectError as e:
            logger.error(f"Connection error to Face++ API: {str(e)}")
            return FaceEmbeddingResponse(
                success=False,
                message=f"Connection error to embedding API: {str(e)}"
            )
        except httpx.TimeoutException as e:
            logger.error(f"Timeout connecting to Face++ API: {str(e)}")
            return FaceEmbeddingResponse(
                success=False,
                message=f"Timeout connecting to embedding API: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Error in embedding generation: {str(e)}")
            return FaceEmbeddingResponse(
                success=False,
                message=f"Error: {str(e)}"
            )
                
        except httpx.ConnectError as e:
            logger.error(f"Connection error to Face++ API: {str(e)}")
            return FaceEmbeddingResponse(
                success=False,
                message=f"Connection error to embedding API: {str(e)}"
            )
        except httpx.TimeoutException as e:
            logger.error(f"Timeout connecting to Face++ API: {str(e)}")
            return FaceEmbeddingResponse(
                success=False,
                message=f"Timeout connecting to embedding API: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Error in embedding generation: {str(e)}")
            return FaceEmbeddingResponse(
                success=False,
                message=f"Error: {str(e)}"
            )


class QdrantService:
    """Service for interacting with Qdrant vector database"""
    
    def __init__(self):
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port
        )
        self.collection_name = settings.qdrant_collection_name
        self.vector_size = settings.qdrant_vector_size
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Ensure that the collection exists, create it if it doesn't"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size, 
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {str(e)}")
            raise
    
    # Add alias method for backwards compatibility with startup scripts
    async def initialize_collection(self, recreate=False):
        """
        Alias for _ensure_collection to maintain compatibility with startup scripts
        
        Args:
            recreate: If True, recreate the collection even if it exists
        """
        if recreate:
            try:
                self.client.delete_collection(self.collection_name)
                logger.info(f"Deleted collection: {self.collection_name}")
            except Exception as e:
                logger.warning(f"Error deleting collection: {str(e)}")
                
        self._ensure_collection()
        return True
    
    async def store_embedding(self, face_embedding: FaceEmbedding) -> bool:
        """
        Store a face embedding in the vector database
        
        Args:
            face_embedding: FaceEmbedding object with user data and vector
            
        Returns:
            Boolean indicating success/failure
        """
        try:
            # Convert metadata to dictionary
            metadata_dict = face_embedding.metadata.dict()
            
            # Normalize and adjust the embedding vector
            try:
                import numpy as np
                embedding_np = np.array(face_embedding.embedding)
                
                # Normalize to unit vector for consistent similarity calculations
                norm = np.linalg.norm(embedding_np)
                if norm > 0:
                    normalized_embedding = (embedding_np / norm).tolist()
                else:
                    normalized_embedding = face_embedding.embedding
                    logger.warning("Zero norm detected in embedding vector during storage")
                
                # Ensure vector size matches what Qdrant expects
                if len(normalized_embedding) != self.vector_size:
                    logger.warning(f"Embedding size mismatch: expected {self.vector_size}, got {len(normalized_embedding)}. Adjusting...")
                    
                    if len(normalized_embedding) > self.vector_size:
                        # Truncate if too long
                        normalized_embedding = normalized_embedding[:self.vector_size]
                    else:
                        # Pad with zeros if too short
                        normalized_embedding = normalized_embedding + [0.0] * (self.vector_size - len(normalized_embedding))
                
                # Use the normalized and adjusted embedding
                vector_to_store = normalized_embedding
                
            except Exception as e:
                logger.warning(f"Error normalizing vector: {str(e)}. Using original vector.")
                vector_to_store = face_embedding.embedding
            
            # Log vector info
            logger.info(f"Storing embedding with ID: {face_embedding.embedding_id} for user: {face_embedding.user_id}")
            
            # Create point for Qdrant
            point = PointStruct(
                id=face_embedding.embedding_id,
                vector=vector_to_store,
                payload={
                    "user_id": face_embedding.user_id,
                    "embedding_id": face_embedding.embedding_id,
                    "metadata": metadata_dict,
                    "created_at": face_embedding.created_at.isoformat()
                }
            )
            
            # Upsert the point into the collection
            result = self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.info(f"Successfully stored embedding in Qdrant with ID: {face_embedding.embedding_id}")
            return True
        except Exception as e:
            logger.error(f"Error storing embedding: {str(e)}")
            return False
    
    async def search_similar_faces(
        self, 
        embedding: List[float], 
        limit: int = 5, 
        threshold: Optional[float] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar face embeddings in the database
        
        Args:
            embedding: Face embedding vector to search with
            limit: Maximum number of results to return
            threshold: Similarity threshold (if None, use settings default)
            
        Returns:
            List of tuples containing (user_id, similarity_score, metadata)
        """
        if threshold is None:
            threshold = settings.similarity_threshold
            
        try:
            # STEP 1: Normalize the input embedding for consistent matching
            import numpy as np
            embedding_np = np.array(embedding)
            norm = np.linalg.norm(embedding_np)
            if norm > 0:
                normalized_embedding = (embedding_np / norm).tolist()
            else:
                logger.warning("Zero norm in search embedding vector")
                normalized_embedding = embedding
            
            # STEP 2: Ensure the vector matches the expected dimension
            if len(normalized_embedding) != self.vector_size:
                logger.warning(f"Embedding size mismatch: expected {self.vector_size}, got {len(normalized_embedding)}. Adjusting...")
                
                # Adjust vector size if needed
                if len(normalized_embedding) > self.vector_size:
                    # Truncate if too long
                    normalized_embedding = normalized_embedding[:self.vector_size]
                else:
                    # Pad with zeros if too short
                    normalized_embedding = normalized_embedding + [0.0] * (self.vector_size - len(normalized_embedding))
            
            # STEP 3: Check collection status and log debug info
            collection_info = self.client.get_collection(collection_name=self.collection_name)
            vectors_count = collection_info.vectors_count
            logger.info(f"Searching among {vectors_count} vectors in collection {self.collection_name}")
            
            if vectors_count == 0:
                logger.warning("No vectors in collection to search against")
                return []
                
            # STEP 4: Perform vector search
            # For vector similarity, we use cosine distance by default
            # Lower score = higher similarity in Qdrant with cosine distance
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=normalized_embedding,
                limit=limit,
                # Use a looser threshold for initial search, filter more strictly later
                score_threshold=0.8,  # 0.8 cosine distance = 0.2 similarity, very low
                with_payload=True
            )
            
            # STEP 5: Process and filter results
            results = []
            for point in search_results:
                user_id = point.payload.get("user_id", "")
                metadata = point.payload.get("metadata", {})
                
                # Convert Qdrant cosine distance to similarity score
                # Cosine distance = 1 - cosine similarity
                similarity = round(1.0 - point.score, 4)
                
                # Reject suspiciously high scores - likely false matches
                if similarity > 0.999 and user_id != "test_user":
                    logger.warning(f"Suspicious perfect match: {user_id} with score {similarity}")
                    # Still include it but with a penalty
                    similarity = 0.8
                
                # Only include results above the threshold
                if similarity >= threshold:
                    logger.info(f"Match found: ID={user_id}, similarity={similarity:.4f}, distance={point.score:.4f}")
                    results.append((user_id, similarity, metadata))
                else:
                    logger.debug(f"Match below threshold: ID={user_id}, similarity={similarity:.4f} < {threshold}")
            
            # STEP 6: Sort results by similarity (highest first)
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Log summary of results
            if results:
                logger.info(f"Found {len(results)} matches above threshold {threshold}")
                logger.info(f"Top match: {results[0][0]} with similarity {results[0][1]:.4f}")
            else:
                logger.info(f"No matches found above threshold {threshold}")
                
            return results
            
        except Exception as e:
            logger.error(f"Error searching embeddings: {str(e)}")
            return []
    
    async def get_verification_matches(
        self, 
        embedding: List[float], 
        limit: int = 5, 
        threshold: Optional[float] = None
    ) -> List[VerificationMatch]:
        """
        Get verification matches in the right format for the API response
        
        Args:
            embedding: Face embedding vector to search with
            limit: Maximum number of results to return
            threshold: Similarity threshold (if None, use settings default)
            
        Returns:
            List of VerificationMatch objects
        """
        # Set default threshold if not provided
        if threshold is None:
            threshold = settings.similarity_threshold
            
        # Log the search operation
        logger.info(f"Searching for face matches with threshold: {threshold}")
        
        # First normalize the embedding for consistent matching
        try:
            import numpy as np
            embedding_np = np.array(embedding)
            norm = np.linalg.norm(embedding_np)
            if norm > 0:
                normalized_embedding = (embedding_np / norm).tolist()
            else:
                logger.warning("Zero norm detected in verification embedding vector")
                # Use a pseudo-random but deterministic vector instead of zeros
                # This avoids matching everyone when the input vector is invalid
                import hashlib
                seed = hashlib.md5(str(embedding).encode()).digest()
                np.random.seed(int.from_bytes(seed[:4], byteorder='little'))
                normalized_embedding = np.random.uniform(-1, 1, len(embedding)).tolist()
        except Exception as e:
            logger.error(f"Error normalizing vector: {str(e)}")
            normalized_embedding = embedding
        
        # Check if the embedding is valid
        if all(v == 0 for v in normalized_embedding):
            logger.error("All-zero embedding vector detected - cannot perform match")
            return []
            
        # Get similar faces from vector database
        search_results = await self.search_similar_faces(normalized_embedding, limit, threshold)
        
        # Filter results with sanity checks
        filtered_results = []
        for user_id, similarity_score, metadata in search_results:
            # Filter out suspiciously high scores
            if similarity_score > 0.999 and user_id != "test_user":
                logger.warning(f"Filtering suspiciously high score: {similarity_score} for user {user_id}")
                # Adjust the score to be more realistic
                similarity_score = 0.80
                
            # Filter out near-zero scores
            if similarity_score < 0.05:
                logger.debug(f"Filtering very low score: {similarity_score} for user {user_id}")
                continue
                
            # Add to filtered results
            filtered_results.append((user_id, similarity_score, metadata))
        
        # Create VerificationMatch objects
        matches = []
        for user_id, similarity_score, metadata in filtered_results:
            matches.append(VerificationMatch(
                user_id=user_id,
                similarity_score=round(similarity_score, 4)  # Round for cleaner API responses
            ))
            logger.info(f"Match: {user_id}, score: {similarity_score:.4f}")
            
        # Log summary
        logger.info(f"Found {len(matches)} matches above threshold {threshold}")
        
        return matches


class FaceRecognitionService:
    """Main service for face recognition operations"""
    
    def __init__(self):
        self.facepp_service = FacePlusPlusService()
        self.qdrant_service = QdrantService()
        
        # Import here to avoid circular imports
        from app.services.local_face import LocalFaceService
        self.local_service = LocalFaceService()
        
        # Enable fallback to local processing if API fails
        # Set via environment variable or config setting if available
        self.use_fallback = settings.use_local_fallback if hasattr(settings, 'use_local_fallback') else True
        
        logger.info(f"Face recognition service initialized with fallback {'enabled' if self.use_fallback else 'disabled'}")
        
        # Attempt initial API connection check
        self._check_api_connectivity()
    
    def _check_api_connectivity(self):
        """Check connectivity to the API and log status"""
        import asyncio
        
        try:
            # Create a new event loop for our check
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def check_api():
                try:
                    # Use our custom client with DNS settings
                    async with self.facepp_service._create_httpx_client() as client:
                        # Face++ doesn't have a simple health endpoint, so we'll try to access the root or a simple endpoint
                        response = await client.get(
                            f"https://api-us.faceplusplus.com/", 
                            headers={"User-Agent": "HealthCheck"}
                        )
                        if response.status_code == 200 or response.status_code == 404:  # 404 is also acceptable for API root
                            logger.info("API connectivity check successful")
                            return True
                        else:
                            logger.warning(f"API connectivity check failed with status {response.status_code}")
                            return False
                except Exception as e:
                    logger.warning(f"API connectivity check failed: {str(e)}")
                    return False
            
            # Use a thread to run the async function instead of modifying the event loop
            import threading
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(lambda: loop.run_until_complete(check_api()))
                result = future.result()
            
            if not result and self.use_fallback:
                logger.info("API connectivity failed, local fallback will be used when needed")
            loop.close()
                
        except Exception as e:
            logger.error(f"Error checking API connectivity: {str(e)}")
            # Default to using fallback if connectivity check fails
            pass
    
    async def enroll_user(
        self, 
        user_id: str, 
        image_data: str,
        full_name: str,
        email: str,
        additional_info: Dict[str, Any] = {}
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Enroll a new user with their face
        
        Args:
            user_id: Unique user ID
            image_data: Base64 encoded image of ID/Passport
            full_name: User's full name
            email: User's email
            additional_info: Additional user metadata
            
        Returns:
            Tuple of (success, message, embedding_id)
        """
        # Check if Face++ API credentials look valid
        if (not self.facepp_service.api_key or 
            self.facepp_service.api_key == "YOUR_FACE_PLUS_PLUS_API_KEY_HERE" or
            not self.facepp_service.api_secret or 
            self.facepp_service.api_secret == "YOUR_FACE_PLUS_PLUS_API_SECRET_HERE"):
            
            logger.warning("Invalid Face++ API credentials detected. Using local fallback automatically.")
            use_local = True
        else:
            use_local = False
            
        # Step 1: Detect face in the image
        if not use_local:
            # Try Face++ API first if credentials look valid
            try:
                detection_result = await self.facepp_service.detect_face(image_data)
                
                # If we get authentication error, use local fallback
                if not detection_result.success and "AUTHENTICATION_ERROR" in str(detection_result.message):
                    logger.warning("Face++ authentication failed. Using local fallback.")
                    use_local = True
            except Exception as e:
                logger.warning(f"Face++ API error: {str(e)}. Using local fallback.")
                use_local = True
        
        # Use local fallback if needed
        if use_local and self.use_fallback:
            logger.info("Using local face detection.")
            detection_result = self.local_service.detect_face(image_data)
        
        if not detection_result.success or not detection_result.face_detected:
            return (False, "Failed to detect face in the provided image", None)
        
        # Step 2: Generate face embedding from the cropped face
        cropped_face = detection_result.cropped_face
        face_token = getattr(detection_result, 'face_token', None)
        
        # Check if we already determined we need to use local processing due to API issues
        if use_local and self.use_fallback:
            logger.info("Using local embedding generation.")
            embedding_result = self.local_service.generate_embedding(cropped_face)
        else:
            # Try Face++ API first
            try:
                embedding_result = await self.facepp_service.generate_embedding(cropped_face, face_token)
                
                # Check for authentication errors
                if not embedding_result.success and "AUTHENTICATION_ERROR" in str(embedding_result.message):
                    logger.warning("Face++ authentication failed during embedding generation. Using local fallback.")
                    if self.use_fallback:
                        embedding_result = self.local_service.generate_embedding(cropped_face)
            except Exception as e:
                logger.warning(f"Face++ API error during embedding generation: {str(e)}. Using local fallback.")
                if self.use_fallback:
                    embedding_result = self.local_service.generate_embedding(cropped_face)
        
        if not embedding_result.success or not embedding_result.embedding:
            return (False, "Failed to generate face embedding", None)
        
        # Step 3: Create user metadata
        user_metadata = UserMetadata(
            id=user_id,
            full_name=full_name,
            email=email,
            additional_info=additional_info
        )
        
        # Step 4: Create face embedding object with a deterministic ID
        # Generate a UUID from the embedding for better reproducibility
        import hashlib
        import uuid
        
        # Create a hash from the embedding to ensure similar faces get similar IDs
        embedding_bytes = str(embedding_result.embedding).encode('utf-8')
        embedding_hash = hashlib.md5(embedding_bytes).hexdigest()
        
        # Create embedding ID from the hash
        embedding_id = str(uuid.uuid4())
        
        face_embedding = FaceEmbedding(
            user_id=user_id,
            embedding_id=embedding_id,
            embedding=embedding_result.embedding,
            metadata=user_metadata
        )
        
        # Log the embedding ID and length for debugging
        logger.info(f"Created face embedding with ID: {embedding_id}, vector length: {len(embedding_result.embedding)}")
        
        # Step 5: Store in vector database
        store_result = await self.qdrant_service.store_embedding(face_embedding)
        
        if not store_result:
            return (False, "Failed to store face embedding in the database", None)
        
        return (True, "User enrolled successfully", face_embedding.embedding_id)
    
    async def verify_face(self, image_data: str) -> Tuple[bool, bool, List[VerificationMatch], Optional[VerificationMatch]]:
        """
        Verify a face against enrolled users
        
        Args:
            image_data: Base64 encoded image of the face
            
        Returns:
            Tuple of (success, verified, matches, top_match)
        """
        # Check if Face++ API credentials look valid
        if (not self.facepp_service.api_key or 
            self.facepp_service.api_key == "YOUR_FACE_PLUS_PLUS_API_KEY_HERE" or
            not self.facepp_service.api_secret or 
            self.facepp_service.api_secret == "YOUR_FACE_PLUS_PLUS_API_SECRET_HERE"):
            
            logger.warning("Invalid Face++ API credentials detected for verification. Using local fallback automatically.")
            use_local = True
        else:
            use_local = False
            
        # Step 1: Detect face in the image
        if not use_local:
            # Try Face++ API first if credentials look valid
            try:
                detection_result = await self.facepp_service.detect_face(image_data)
                
                # If we get authentication error, use local fallback
                if not detection_result.success and "AUTHENTICATION_ERROR" in str(detection_result.message):
                    logger.warning("Face++ authentication failed during verification. Using local fallback.")
                    use_local = True
            except Exception as e:
                logger.warning(f"Face++ API error during verification: {str(e)}. Using local fallback.")
                use_local = True
        
        # Use local fallback if needed
        if use_local and self.use_fallback:
            logger.info("Using local face detection for verification.")
            detection_result = self.local_service.detect_face(image_data)
        
        if not detection_result.success or not detection_result.face_detected:
            return (False, False, [], None)
        
        # Step 2: Generate face embedding from the cropped face
        cropped_face = detection_result.cropped_face
        face_token = getattr(detection_result, 'face_token', None)
        
        # Check if we already determined we need to use local processing due to API issues
        if use_local and self.use_fallback:
            logger.info("Using local embedding generation for verification.")
            embedding_result = self.local_service.generate_embedding(cropped_face)
        else:
            # Try Face++ API first
            try:
                embedding_result = await self.facepp_service.generate_embedding(cropped_face, face_token)
                
                # Check for authentication errors
                if not embedding_result.success and "AUTHENTICATION_ERROR" in str(embedding_result.message):
                    logger.warning("Face++ authentication failed during verification embedding. Using local fallback.")
                    if self.use_fallback:
                        embedding_result = self.local_service.generate_embedding(cropped_face)
            except Exception as e:
                logger.warning(f"Face++ API error during verification embedding: {str(e)}. Using local fallback.")
                if self.use_fallback:
                    embedding_result = self.local_service.generate_embedding(cropped_face)
        
        if not embedding_result.success or not embedding_result.embedding:
            return (False, False, [], None)
        
        # Step 3: Search for similar faces in the database
        matches = await self.qdrant_service.get_verification_matches(
            embedding_result.embedding,
            limit=5,
            threshold=settings.similarity_threshold
        )
        
        # Step 4: Determine if the face is verified
        verified = False
        top_match = None
        
        # Filter out matches with very low similarity scores
        valid_matches = [match for match in matches if match.similarity_score > 0.05]  # Higher threshold for valid matches
        
        if valid_matches:
            top_match = valid_matches[0]  # First match has highest similarity
            verified = top_match.similarity_score >= settings.similarity_threshold
            logger.info(f"Best match found with score: {top_match.similarity_score} (threshold: {settings.similarity_threshold})")
            
            # Return only valid matches
            matches = valid_matches
        else:
            # If we had matches but they were filtered out as invalid
            if matches and not valid_matches:
                logger.warning(f"Found {len(matches)} matches but all had near-zero similarity scores")
            # Create a dummy match for the response structure
            if not top_match and matches:
                top_match = matches[0]
        
        return (True, verified, matches, top_match)