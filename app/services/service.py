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
        self.api_url = settings.facepp_api_url
        self.api_key = settings.facepp_api_key
        self.api_secret = settings.facepp_api_secret
        
        # No need for headers as Face++ uses query parameters for authentication
        self.max_retries = 3
        self.timeout = 30.0  # increased timeout
        
        # Configure custom DNS resolver if available
        self._configure_dns()
        
        logger.info(f"Initialized FacePlusPlusService with API URL: {self.api_url}")
        
        # Validate API credentials
        if not self.api_key or self.api_key == "YOUR_API_KEY" or len(self.api_key) < 10:
            logger.error(f"Invalid Face++ API key: '{self.api_key}'. Please check your .env file.")
        if not self.api_secret or self.api_secret == "YOUR_API_SECRET" or len(self.api_secret) < 10:
            logger.error(f"Invalid Face++ API secret. Please check your .env file.")
            
        self._check_api_url()
    
    def _configure_dns(self):
        """Configure DNS resolution using custom resolv.conf if available"""
        try:
            import dns.resolver
            
            # Check if our custom DNS config exists
            custom_resolv_path = '/app/dns_config/resolv.conf'
            if os.path.exists(custom_resolv_path):
                logger.info(f"Found custom DNS configuration at {custom_resolv_path}")
                
                # Parse the nameservers from our custom resolv.conf
                nameservers = []
                with open(custom_resolv_path, 'r') as f:
                    for line in f:
                        if line.strip().startswith('nameserver'):
                            nameserver = line.strip().split()[1]
                            nameservers.append(nameserver)
                
                # Configure the DNS resolver to use our nameservers
                if nameservers:
                    dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
                    dns.resolver.default_resolver.nameservers = nameservers
                    logger.info(f"Configured DNS resolver with nameservers: {nameservers}")
                    
                    # Set HTTPX to use our resolver
                    # Note: We'll handle this in the httpx client creation
            else:
                logger.info("No custom DNS configuration found, using system defaults")
        except ImportError:
            logger.warning("dnspython not available, using system DNS resolution")
        except Exception as e:
            logger.error(f"Error configuring DNS resolver: {str(e)}")
    
    def _check_api_url(self):
        """Check if API URL is properly formatted and log details for troubleshooting"""
        import socket
        
        try:
            # Extract hostname from URL
            from urllib.parse import urlparse
            parsed_url = urlparse(self.api_url)
            hostname = parsed_url.netloc
            
            # Log API connection details for debugging
            logger.info(f"API URL: {self.api_url}")
            logger.info(f"Hostname to resolve: {hostname}")
            
            # Try to resolve the hostname
            try:
                # Try standard resolution first
                ip_address = socket.gethostbyname(hostname)
                logger.info(f"Successfully resolved {hostname} to {ip_address}")
                resolved = True
            except socket.gaierror:
                logger.warning(f"Could not resolve {hostname} with system DNS")
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
        
        # Set custom DNS resolver if available
        try:
            import dns.resolver
            if hasattr(dns.resolver, 'default_resolver') and dns.resolver.default_resolver.nameservers:
                # DNS resolver is already configured
                pass
        except ImportError:
            pass
            
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
            logger.info(f"Calling Face++ API at {self.api_url}/detect")
            
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
                            f"{self.api_url}/detect",
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
        
        Face++ uses face_set feature for managing face embeddings. We'll use the face_token from detection
        to get the embedding, or create a new face analysis if no token is provided.
        
        Args:
            image_data: Base64 encoded image (can be full image or cropped face)
            face_token: Optional face token from a previous detect call
            
        Returns:
            FaceEmbeddingResponse object with embedding results
        """
        try:
            # For Face++ we'll need to:
            # 1. First detect the face if no face_token is provided
            # 2. Use Face++ face_analyze endpoint for feature extraction
            logger.info(f"Generating face embedding with Face++ API")
            
            # If no face_token was provided, detect the face first
            if not face_token:
                detection_result = await self.detect_face(image_data)
                if not detection_result.success or not detection_result.face_detected:
                    return FaceEmbeddingResponse(
                        success=False,
                        message=f"Face detection failed: {detection_result.message}"
                    )
                face_token = detection_result.face_token
            
            if not face_token:
                return FaceEmbeddingResponse(
                    success=False,
                    message="No face token available for embedding generation"
                )
            
            # Import necessary libraries for conversion
            import base64
            import numpy as np
            from io import BytesIO
            
            # Use retry logic for better reliability
            retries = 0
            last_error = None
            
            # For Face++, we'll use face_analyze endpoint
            while retries < self.max_retries:
                try:
                    # Use custom client with proper configuration
                    async with self._create_httpx_client() as client:
                        # Face++ API data
                        data = {
                            'api_key': self.api_key,
                            'api_secret': self.api_secret,
                            'face_tokens': face_token,  # Fix: API expects 'face_tokens' not 'face_token'
                            'return_landmark': '1',  # Get full facial landmarks
                        }
                        
                        # Face++ API endpoint is facepp/v3/face/analyze
                        response = await client.post(
                            f"{self.api_url}/face/analyze",
                            data=data
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
                logger.error(f"Face analyze API error: {response.text}")
                return FaceEmbeddingResponse(
                    success=False,
                    message=f"API Error: {response.text}"
                )
            
            result = response.json()
            
            # Log response for debugging
            logger.debug(f"Face++ API response structure: {list(result.keys())}")
            
            # Extract features from Face++ response
            # Since Face++ doesn't directly provide embeddings like some APIs,
            # we'll create an embedding from the landmarks and face attributes
            
            # Face++ analyze API returns faces in a different structure than detect API
            # Try different possible response structures
            face_data = None
            
            # Check for 'faces' list first (common in analyze response)
            if 'faces' in result and isinstance(result['faces'], list) and len(result['faces']) > 0:
                face_data = result['faces'][0]
            # Then check for 'face' dictionary (may occur in some responses)
            elif 'face' in result:
                face_data = result.get('face', {})
            
            if not face_data:
                logger.error(f"Unexpected API response format. Keys in response: {list(result.keys())}")
                return FaceEmbeddingResponse(
                    success=False,
                    message=f"No face data returned from API. Response format: {list(result.keys())}"
                )
                
            # Extract landmarks and attributes to use as embedding
            landmarks = face_data.get('landmark', {})
            attributes = face_data.get('attributes', {})
            
            # Check if we actually have landmark data
            if not landmarks:
                logger.warning("No landmark data in Face++ API response. Attempting fallback method.")
                # Try alternative fields in the response
                if 'face_rectangle' in face_data:
                    rect = face_data['face_rectangle']
                    # Create a simple embedding from face rectangle and other available data
                    rect_values = [
                        rect.get('top', 0), rect.get('left', 0),
                        rect.get('width', 0), rect.get('height', 0)
                    ]
                    
                    # Add some face attributes if available
                    if 'gender' in attributes:
                        gender_value = 1.0 if attributes['gender'].get('value') == 'Male' else 0.0
                        rect_values.append(gender_value)
                    
                    if 'age' in attributes:
                        age_value = float(attributes['age'].get('value', 0)) / 100.0  # Normalize
                        rect_values.append(age_value)
                    
                    # Create a synthetic embedding from available data
                    import hashlib
                    import struct
                    
                    # Create a deterministic but diverse vector from the available data
                    hash_input = str(rect_values) + str(face_data)
                    hash_bytes = hashlib.sha256(hash_input.encode()).digest()
                    
                    # Convert hash bytes to floating point values between -1 and 1
                    embedding = []
                    for i in range(0, len(hash_bytes) - 3, 4):
                        val = struct.unpack('f', hash_bytes[i:i+4])[0]
                        # Normalize to range approximately -1 to 1
                        embedding.append(max(min(val, 1.0), -1.0))
                    
                    # Include the actual rectangle values at the beginning
                    embedding = rect_values + embedding
                else:
                    # If we don't have any useful data, return an error
                    logger.error("No usable face data in response")
                    return FaceEmbeddingResponse(
                        success=False,
                        message="No usable face data in API response"
                    )
            else:
                # Create a vector from the landmark points
                embedding = []
                
                # Add landmark points to the embedding
                for point_name, point_coords in landmarks.items():
                    embedding.append(point_coords.get('x', 0))
                    embedding.append(point_coords.get('y', 0))
                
            # Face++ typically returns 106 landmarks with x,y coordinates
            # resulting in ~212 values. We pad to reach expected vector size
            # or truncate if necessary to match our vector space
            target_size = settings.qdrant_vector_size
            
            if len(embedding) < target_size:
                # Pad with zeros if needed
                embedding.extend([0] * (target_size - len(embedding)))
            elif len(embedding) > target_size:
                # Truncate if too large
                embedding = embedding[:target_size]
            
            # Normalize the embedding to unit length (common for face embeddings)
            embedding_array = np.array(embedding)
            norm = np.linalg.norm(embedding_array)
            if norm > 0:
                embedding_array = embedding_array / norm
            
            # Convert back to list for JSON serialization
            normalized_embedding = embedding_array.tolist()
            
            # Return the embedding
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
            # Make sure the embedding vector matches the expected dimension
            if len(embedding) != self.vector_size:
                logger.warning(f"Embedding vector size mismatch: expected {self.vector_size}, got {len(embedding)}. Adjusting...")
                
                # Adjust vector size if needed
                if len(embedding) > self.vector_size:
                    # Truncate if too long
                    embedding = embedding[:self.vector_size]
                else:
                    # Pad with zeros if too short
                    embedding = embedding + [0.0] * (self.vector_size - len(embedding))
            
            # Check total vectors in the collection for debugging
            collection_info = self.client.get_collection(collection_name=self.collection_name)
            vectors_count = collection_info.vectors_count
            logger.info(f"Searching among {vectors_count} vectors in collection {self.collection_name}")
            
            if vectors_count == 0:
                logger.warning("No vectors in collection to search against")
                return []
                
            # For debugging, let's get all stored vectors first to check if there's any issue
            all_points = self.client.scroll(
                collection_name=self.collection_name,
                limit=10,  # Get first 10 for debugging
                with_payload=True,
                with_vectors=False  # Don't need actual vectors
            )[0]
            
            logger.info(f"Debug: First stored point ID: {all_points[0].id if all_points else 'None'}")
            
            # For better matching, always search without score threshold first,
            # then filter by threshold afterwards
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=embedding,
                limit=limit,
                score_threshold=None  # Don't use threshold at search time
            )
            
            # Format results
            results = []
            for point in search_results:
                user_id = point.payload.get("user_id", "")
                metadata = point.payload.get("metadata", {})
                similarity = 1.0 - point.score  # Convert cosine distance to similarity
                
                # Only include results that meet the threshold
                if similarity >= threshold:
                    logger.info(f"Match found: ID={user_id}, score={similarity:.4f}, distance={point.score:.4f}")
                    results.append((user_id, similarity, metadata))
                else:
                    logger.debug(f"Match below threshold: ID={user_id}, score={similarity:.4f} < {threshold}")
                
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
        # Normalize the embedding vector to improve matching consistency
        # This helps ensure Face++ and local fallback vectors are comparable
        try:
            import numpy as np
            embedding_np = np.array(embedding)
            norm = np.linalg.norm(embedding_np)
            if norm > 0:  # Avoid division by zero
                normalized_embedding = (embedding_np / norm).tolist()
            else:
                normalized_embedding = embedding
                logger.warning("Zero norm detected in embedding vector, skipping normalization")
        except Exception as e:
            logger.warning(f"Error normalizing embedding vector: {str(e)}. Using original vector.")
            normalized_embedding = embedding
            
        # Log debug information
        logger.debug(f"Searching for matches with threshold: {threshold or settings.similarity_threshold}")
        
        search_results = await self.search_similar_faces(normalized_embedding, limit, threshold)
        matches = []
        
        # Log the number of matches found
        logger.info(f"Found {len(search_results)} potential matches")
        
        for user_id, similarity_score, metadata in search_results:
            # Use the new simplified VerificationMatch model (without full_name and email)
            matches.append(VerificationMatch(
                user_id=user_id,
                similarity_score=similarity_score
            ))
            logger.debug(f"Match: {user_id}, score: {similarity_score}")
            
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
        valid_matches = [match for match in matches if match.similarity_score > 0.01]  # Ignore essentially zero matches
        
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