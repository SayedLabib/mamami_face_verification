import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file, silent=True to avoid errors if no file
load_dotenv(find_dotenv(), override=True)

class Settings(BaseSettings):
    # App Information
    app_name: str = "Face Recognition API"
    admin_email: str = "admin@example.com"
    
    # Face++ API Configuration
    facepp_api_key: str
    facepp_api_secret: str
    # Face++ API URLs
    fpp_create_url: str = "https://api-us.faceplusplus.com/facepp/v3/faceset/create"
    fpp_detect_url: str = "https://api-us.faceplusplus.com/facepp/v3/detect"
    fpp_search_url: str = "https://api-us.faceplusplus.com/facepp/v3/search"
    fpp_add_url: str = "https://api-us.faceplusplus.com/facepp/v3/faceset/addface"
    fpp_get_detail_url: str = "https://api-us.faceplusplus.com/facepp/v3/faceset/getdetail"
    
    # Qdrant Configuration
    qdrant_host: str
    qdrant_port: int
    qdrant_collection_name: str
    qdrant_vector_size: int
    
    # Face Recognition Settings
    similarity_threshold: float
    use_local_fallback: bool = True  # Whether to use local face processing when API fails

# Check if we're running in Docker environment
def is_docker():
    path = '/proc/self/cgroup'
    return os.path.exists('/.dockerenv') or (os.path.isfile(path) and any('docker' in line for line in open(path)))

# Use service name as hostname when running in Docker
qdrant_host = "qdrant" if is_docker() else os.getenv("QDRANT_HOST", "localhost")

# Create settings instance with explicit values from environment variables
settings = Settings(
    facepp_api_key=os.getenv("FACEPP_API_KEY", ""),
    facepp_api_secret=os.getenv("FACEPP_API_SECRET", ""),
    # Face++ URLs are already set with defaults in the Settings class
    qdrant_host=qdrant_host,
    qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
    qdrant_collection_name=os.getenv("QDRANT_COLLECTION_NAME", "face_embeddings"),
    qdrant_vector_size=int(os.getenv("QDRANT_VECTOR_SIZE", "512")),
    similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.7")),
    use_local_fallback=os.getenv("USE_LOCAL_FALLBACK", "true").lower() == "true"
)