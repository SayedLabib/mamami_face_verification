from fastapi import FastAPI
from fastapi import status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from app.services.verification_system.face_verification.face_verification_router import router as face_router
from app.services.verification_system.dating_chat_system.dating_chat_router import router as chat_router


app = FastAPI(
    title="Mamami Face Verification & Romantic Dating Chat System",
    description="API for face verification, duplicate detection using Face++ API, and romantic AI partner chat assistant using Groq LLM",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(face_router)
app.include_router(chat_router)


@app.get("/", status_code=status.HTTP_200_OK, response_class=PlainTextResponse)
async def health_check():
    return "Mamami Face Verification & Romantic Dating Chat System is running and healthy"
