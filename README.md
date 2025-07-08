# Face Recognition Duplicate Detection System

A FastAPI-based system that uses Face++ API to prevent duplicate account creation by detecting and comparing face images from NID/Passport documents. The system uses in-memory storage to compare uploaded face images and detect duplicates.

## Features

- Face detection using Face++ API
- Direct face comparison between uploaded images
- In-memory storage for face tokens during session
- Multiple endpoints for different verification workflows
- Docker and Docker Compose support
- Nginx reverse proxy configuration
- RESTful API endpoints for face verification and comparison

## Prerequisites

- Python 3.10+
- Docker and Docker Compose (optional)
- Face++ API credentials

## Local Development Setup

1. Clone the repository
```sh
git clone < https://github.com/syeda-ai-dev/Duplicate-Account-Creation-Prevention-System.git >
cd Duplicate-Account-Creation-Prevention-System
```

2. Create and activate a virtual environment
```sh
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies
```sh
pip install -r requirements.txt
```

4. Create a .env file with your Face++ API configurations
```sh
FPP_API_KEY = 'your-face-plus-plus-api-key'
FPP_API_SECRET = 'your-face-plus-plus-secret-key'
FPP_CREATE = 'https://api-us.faceplusplus.com/facepp/v3/faceset/create'
FPP_DETECT = 'https://api-us.faceplusplus.com/facepp/v3/detect'
FPP_SEARCH = 'https://api-us.faceplusplus.com/facepp/v3/search'
FPP_ADD = 'https://api-us.faceplusplus.com/facepp/v3/faceset/addface'
FPP_GET_DETAIL = 'https://api-us.faceplusplus.com/facepp/v3/faceset/getdetail'
```

5. Run the application locally using Uvicorn
```sh
uvicorn com.mhire.app.main:app --reload
```
The API will be available at ```http://localhost:8000```, with Swagger UI: ```http://localhost:8000/docs```

## Docker Setup

1. Build and run using Docker Compose
```sh
docker-compose up --build -d
```
This will:

- Build the FastAPI application container
- Set up Nginx reverse proxy
- Expose the service on port 8080
- Access the API at ```http://your-ip-address:8080```, with Swagger UI: ```http://your-ip-address:8080/docs```

2. Stop the containers
```sh
docker-compose down
```

## API Endpoints

### Face Verification Endpoints:

1. **POST /api/v1/face/upload**
   - Upload a face image (NID/Passport) for duplicate detection
   - Returns duplicate status and confidence score if duplicate found
   - Saves new faces to in-memory storage if no duplicates found
   - Optional session_id parameter for grouping faces

2. **POST /api/v1/face/compare**
   - Compare two uploaded face images directly
   - Returns confidence score and match status
   - Useful for direct face-to-face comparison

3. **GET /api/v1/face/stats**
   - Get statistics about stored faces and sessions
   - Returns total faces, sessions, and metadata

4. **DELETE /api/v1/face/clear**
   - Clear stored face data for specific session or all sessions
   - Optional session_id parameter

### Job Description Endpoint:
- **POST /api/v1/job/description**: Generate structured job descriptions

## How It Works

1. **Face Upload**: Upload a face image from NID/Passport document
2. **Face Detection**: System detects and extracts face features using Face++ API
3. **Duplicate Check**: Compares the detected face with previously stored faces
4. **Confidence Scoring**: Returns confidence score for potential matches
5. **Storage**: Saves new unique faces to in-memory storage for future comparisons

## Configuration

The system uses environment variables for configuration:
- Face++ API credentials for face detection and comparison
- OpenAI API settings for job description generation (optional)
- Confidence threshold for duplicate detection (default: 90%)

## Security Features

- File type validation for uploaded images
- Rate limiting for API requests
- Error handling and logging
- Session-based face grouping