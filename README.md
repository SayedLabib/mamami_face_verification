# Face Recognition System with Qdrant Vector DB

This project is a robust face recognition system that uses Face++ API for face detection and embedding generation, with a local OpenCV-based fallback option. It uses Qdrant Vector Database for efficient similarity search of face embeddings.

## Architecture

The system consists of two main processes:

1. **User Enrollment**:
   - Takes a user's NID/Passport image and personal information
   - Detects face using Face++ API
   - Extracts face embedding (feature vector)
   - Stores embedding with user metadata in Qdrant Vector DB

2. **Face Verification**:
   - Takes a new image of a user's face
   - Detects face and extracts embedding
   - Searches for similar face embeddings in Qdrant DB
   - Verifies identity based on similarity threshold

## Project Structure

```
mamai_face_recognition_SMT/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── security.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── models.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── endpoints/
│   │   │   ├── __init__.py
│   │   │   └── router.py
│   │   └── dependencies.py
│   └── services/
│       ├── __init__.py
│       └── service.py
├── dns_config/  # DNS configuration for API access
├── .env
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Features

- **Face Detection**: Detects faces in ID documents and selfies
- **Face Embedding**: Extracts feature vectors for face comparison
- **Vector Database**: Efficient storage and similarity search of face vectors
- **User Enrollment**: Register users with face biometrics
- **Face Verification**: Verify identity through face comparison
- **API Endpoints**: Simple RESTful API for operations
- **Docker Support**: Easy deployment with Docker and Docker Compose
- **Local Fallback**: Falls back to OpenCV-based face processing if external API fails
- **DNS Resolution**: Robust DNS configuration for reliable API access
- **Diagnostics API**: Built-in system diagnostics for troubleshooting
- **Auto-Recovery**: Automatic retry logic for transient API failures

## Environment Setup

1. Create a `.env` file with the following configurations:

```
# Face++ API Configuration
FACEPP_API_URL=https://api-us.faceplusplus.com/facepp/v3
FACEPP_API_KEY=your_face_plus_plus_api_key
FACEPP_API_SECRET=your_face_plus_plus_api_secret

# Qdrant Configuration
QDRANT_HOST=qdrant  # Use 'localhost' for non-Docker setup
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=face_embeddings
QDRANT_VECTOR_SIZE=512  # Size of face embeddings derived from Face++ landmarks

# Application Settings
SIMILARITY_THRESHOLD=0.7  # Similarity threshold for face verification
USE_LOCAL_FALLBACK=true  # Use local face processing if API fails
LOG_LEVEL=INFO
```

## Running the Application

### Using Docker (Recommended)

1. Make sure Docker and Docker Compose are installed
2. Run the application using the provided start script:

```bash
# On Windows
start.bat

# On Linux/Mac
chmod +x startup.sh
./startup.sh
```

3. Access the API documentation at http://localhost:8000/docs

### Manual Setup

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
uvicorn app.main:app --reload
```

## Local Fallback Implementation

The system uses a robust approach to face recognition:

1. **Primary Method**: Uses the Face++ API for high-quality face detection and embedding generation
2. **Fallback Method**: If the primary API fails, falls back to local processing:
   - Face Detection: Uses OpenCV's Haar Cascade classifier
   - Face Embedding: Generates HOG (Histogram of Oriented Gradients) features

The fallback mechanism works transparently and ensures system resilience even when:
- External API is unavailable
- Network connectivity issues occur
- API rate limits are exceeded

You can enable/disable the fallback by setting `USE_LOCAL_FALLBACK=true|false` in the `.env` file.

## DNS Configuration & API Connectivity

The system is designed to work with both external API calls and local fallback:

1. **DNS Configuration**: Instead of modifying system files, we use:
   - Custom DNS resolvers in application code
   - Docker's `extra_hosts` for hostname resolution
   - Direct IP fallback in code when hostname resolution fails

2. **Testing DNS Resolution**: Use the included test script:
   ```bash
   python test_dns.py
   ```

## Troubleshooting

If you experience API connectivity issues:

1. Check the diagnostic endpoint at `/diagnostics`
2. Run the DNS test script inside the Docker container:
   ```bash
   docker-compose exec web bash /app/dns_test.sh
   ```
   This will test DNS resolution, API connectivity, and show current DNS configuration.
3. Check if the local fallback is working:
   ```bash
   docker-compose logs web | grep "fallback"
   ```
4. Ensure your API key is valid and has sufficient quota
5. If DNS issues persist, try the direct IP approach in `.env`:
   ```
   LUXAN_API_URL=http://104.21.23.75
   ```

## Testing

The project includes a test script to verify the entire system works correctly:

```bash
# Run the test script
python test_system.py --id-image path/to/id_image.jpg --face-image path/to/face_image.jpg
```

This script tests:
1. API connectivity and DNS resolution
2. User enrollment with ID/passport image
3. Face verification with a separate face image
4. Both primary API path and local fallback functionality

## API Endpoints

### 1. Extract Face and Store

```
POST /extract-and-store
```

This endpoint extracts a face from an ID/passport image and stores its embedding in the database.

Request:
- `user_id`: (Optional) Unique identifier for the user (auto-generated UUID if not provided)
- `full_name`: (Optional) User's full name (defaults to "Unknown User")
- `email`: (Optional) User's email address (defaults to "unknown@example.com")
- `image`: ID/Passport image file (multipart form data)
- `additional_info`: (Optional) Additional user metadata as JSON

Response:

### 2. Verify Face Match

```
POST /verify-match
```

This endpoint checks if a submitted face matches any stored faces.

Request:
- `image`: Face image file (multipart form data)

Response:
- `success`: Boolean indicating if processing was successful
- `verified`: Boolean indicating if the face matches any stored face
- `matches`: List of potential matches with similarity scores
- `top_match`: The best match if any is found
- `message`: Process result message

### 3. System Diagnostics

```
GET /diagnostics
```

This endpoint provides system diagnostics to help troubleshoot API connectivity issues.

Response:
- `dns_check`: Results of DNS resolution tests
- `api_connectivity`: Status of API connectivity
- `environment`: Current environment settings
```json
{
  "success": true,
  "user_id": "12345",
  "embedding_id": "abcdef-12345",
  "message": "User enrolled successfully"
}
```

### 2. Verify Face Match

```
POST /verify-match
```

This endpoint checks if a new face image matches any stored face embeddings.

Request:
- `image`: Face image file to verify (multipart form data)

Response:
```json
{
  "success": true,
  "verified": true,
  "matches": [
    {
      "user_id": "12345",
      "similarity_score": 0.92,
      "full_name": "John Doe",
      "email": "john@example.com"
    }
  ],
  "top_match": {
    "user_id": "12345",
    "similarity_score": 0.92,
    "full_name": "John Doe",
    "email": "john@example.com"
  },
  "message": "Face match found"
}
```

## Local Development Setup

1. **Clone the Repository**:
   ```
   git clone <repository-url>
   cd mamai_face_recognition_SMT
   ```

2. **Create a Virtual Environment**:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```

4. **Set up Environment Variables**:
   Create a `.env` file with required configurations (see Environment Setup section)

5. **Run the Application**:
   ```
   uvicorn app.main:app --reload
   ```

## Docker Deployment

1. **Create the `.env` file** with your configuration

2. **Build and Run with Docker Compose**:
   ```
   docker-compose up -d
   ```

3. **Access the API**:
   Open your browser and navigate to `http://localhost:8000/docs` for Swagger UI documentation

## Notes

- This system uses the Face++ API for face detection and embedding generation
- Adjust the similarity threshold based on testing and security requirements
- This application has no built-in authentication - add security measures if deploying to production