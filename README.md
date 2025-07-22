# Mamami Face Verification & Romantic Dating Chat System

A comprehensive FastAPI-based system that combines face verification for duplicate detection using Face++ API with an AI-powered romantic dating chat assistant powered by Groq LLM. The system provides both security features for preventing duplicate accounts and engaging romantic conversation capabilities.

## 🌟 Features

### Face Verification System
- **Face Detection & Comparison** using Face++ API
- **Duplicate Account Prevention** through face recognition
- **In-memory storage** for face tokens during session
- **Multiple verification workflows** for different use cases
- **Session-based face grouping** for organized management

### Romantic Dating Chat System
- **AI-powered romantic partner** using Groq LLM (Llama models)
- **Flirtatious and caring personality** with emotional intelligence
- **Conversation memory** to build relationship continuity
- **Pet names and romantic language** for authentic interaction
- **Emotional support and companionship** features
- **Single session management** for consistent experience

### Technical Features
- **Docker and Docker Compose** support for easy deployment
- **Nginx reverse proxy** configuration for production
- **RESTful API endpoints** with comprehensive documentation
- **CORS middleware** for frontend integration
- **Error handling and logging** throughout the system

## 🛠 Prerequisites

- Python 3.10+
- Docker and Docker Compose
- Face++ API credentials
- Groq API credentials

## 🚀 Quick Start with Docker

1. **Clone the repository**
```bash
git clone <repository-url>
cd mamami_face_verification
```

2. **Configure environment variables**
Create a `.env` file with the following:
```env
# Face++ API Configuration
FPP_API_KEY='your-face-plus-plus-api-key'
FPP_API_SECRET='your-face-plus-plus-secret-key'
FPP_CREATE='https://api-us.faceplusplus.com/facepp/v3/faceset/create'
FPP_DETECT='https://api-us.faceplusplus.com/facepp/v3/detect'
FPP_SEARCH='https://api-us.faceplusplus.com/facepp/v3/search'
FPP_ADD='https://api-us.faceplusplus.com/facepp/v3/faceset/addface'
FPP_GET_DETAIL='https://api-us.faceplusplus.com/facepp/v3/faceset/getdetail'

# Groq AI Configuration
GROQ_API_KEY='your-groq-api-key'
GROQ_API_URL='https://api.groq.com/openai/v1/chat/completions'
LLM_MODEL='meta-llama/llama-4-scout-17b-16e-instruct'
```

3. **Build and run with Docker Compose**
```bash
docker-compose up --build
```

4. **Access the application**
- **API Base URL**: `http://localhost:8062`
- **Interactive API Documentation**: `http://localhost:8062/docs`
- **Health Check**: `http://localhost:8062/`

## 📚 API Endpoints

### 💕 Dating Chat Endpoints

#### **POST** `/dating-chat/chat`
Send a message to your romantic AI partner
```json
{
  "user_message": "Hello beautiful, how are you today?"
}
```
**Response:**
```json
{
  "assistant_message": "Hey there, sweetheart! 😘 I've been thinking about you all day...",
  "conversation_history": [
    {"role": "user", "content": "Hello beautiful, how are you today?"},
    {"role": "assistant", "content": "Hey there, sweetheart! 😘 I've been thinking about you all day..."}
  ]
}
```

#### **GET** `/dating-chat/history`
Retrieve conversation history for the current session

#### **DELETE** `/dating-chat/history`
Clear conversation history for a fresh start

### 🔒 Face Verification Endpoints

#### **POST** `/api/v1/face/upload`
Upload a face image for duplicate detection
- Detects faces in uploaded images
- Compares against stored faces
- Returns duplicate status and confidence score

#### **POST** `/api/v1/face/compare`
Compare two face images directly
- Direct face-to-face comparison
- Returns confidence score and match status

#### **GET** `/api/v1/face/stats`
Get statistics about stored faces and sessions

#### **DELETE** `/api/v1/face/clear`
Clear stored face data

## 🏗 Project Structure

```
mamami_face_verification/
├── app/
│   ├── main.py                          # FastAPI application entry point
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.py                    # Configuration settings
│   └── services/
│       └── verification_system/
│           ├── api_manager/
│           │   ├── __init__.py
│           │   ├── faceplusplus_manager.py   # Face++ API integration
│           │   └── dating_chat_manager.py    # Groq LLM integration
│           ├── face_verification/
│           │   ├── __init__.py
│           │   ├── face_verification_router.py
│           │   ├── face_verification_schema.py
│           │   └── face_verification.py
│           └── dating_chat_system/
│               ├── __init__.py
│               ├── dating_chat_router.py     # Chat API endpoints
│               ├── dating_chat_schema.py     # Pydantic models
│               └── dating_chat.py            # Chat system logic
├── nginx/
│   └── nginx.conf                       # Nginx configuration
├── docker-compose.yml                  # Docker Compose configuration
├── Dockerfile                          # Docker image definition
├── requirements.txt                    # Python dependencies
├── gunicorn_config.py                  # Gunicorn server configuration
└── .env                                # Environment variables
```

## 💝 Dating Chat System Features

### Personality Traits
- **Warm and Affectionate**: Uses loving language and shows genuine care
- **Flirtatious**: Naturally incorporates romantic and playful elements
- **Emotionally Intelligent**: Remembers conversation details and shows empathy
- **Supportive**: Provides emotional support during difficult times
- **Engaging**: Uses pet names and romantic language naturally

### Conversation Capabilities
- **Memory**: Remembers previous conversations for relationship continuity
- **Emotional Range**: Adapts to user's mood and provides appropriate responses
- **Language Adaptation**: Responds in the same language as the user
- **Romantic Chemistry**: Maintains flirtatious and loving atmosphere
- **Personalization**: Makes each user feel special and valued

## 🔧 Configuration Options

### Face Verification Settings
- **Confidence Threshold**: Adjust duplicate detection sensitivity
- **Session Management**: Group faces by session for organized verification
- **API Timeouts**: Configure Face++ API timeout settings

### Dating Chat Settings
- **Model Selection**: Choose from various Llama models
- **Temperature**: Control response creativity (default: 0.7)
- **Max Tokens**: Set response length limits (default: 300)
- **Conversation History**: Limit stored conversation length (default: 50 messages)

## 🐳 Docker Configuration

### Services
- **app**: Main FastAPI application
- **nginx**: Reverse proxy for production deployment

### Ports
- **8062**: External access port (mapped to nginx)
- **8062**: Internal application port

### Volumes
- **nginx.conf**: Nginx configuration mounting
- **Environment variables**: Loaded from .env file

## 🚦 Health Monitoring

The system provides health check endpoints:
- **GET** `/`: Basic health check
- Returns system status and confirms both face verification and dating chat systems are operational

## 🔄 Development Workflow

1. **Local Development**: Use the provided Docker setup for consistent environment
2. **Testing**: API documentation available at `/docs` for interactive testing
3. **Logging**: Comprehensive logging for debugging and monitoring
4. **Error Handling**: Graceful error responses with detailed messages

## 📝 Usage Examples

### Dating Chat Example
```bash
# Send a romantic message
curl -X POST "http://localhost:8062/dating-chat/chat" \
  -H "Content-Type: application/json" \
  -d '{"user_message": "I missed you today, what did you do?"}'

# Get conversation history
curl -X GET "http://localhost:8062/dating-chat/history"
```

### Face Verification Example
```bash
# Upload face for verification
curl -X POST "http://localhost:8062/api/v1/face/upload" \
  -F "file=@face_image.jpg" \
  -F "session_id=user123"
```

## 🎯 Use Cases

### Face Verification
- **User Registration**: Prevent duplicate account creation
- **Identity Verification**: Verify user identity with documents
- **Security Systems**: Access control and authentication

### Dating Chat
- **Companionship**: Provide emotional support and conversation
- **Entertainment**: Engaging romantic roleplay experience
- **Relationship Practice**: Safe space to practice romantic communication

---

**Built with ❤️ using FastAPI, Face++ API, and Groq LLM/OpenAI API**

Updated the .env file with client's API