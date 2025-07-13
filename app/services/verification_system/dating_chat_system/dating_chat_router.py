from fastapi import APIRouter, HTTPException
from app.services.verification_system.dating_chat_system.dating_chat import DatingChatSystem
from app.services.verification_system.dating_chat_system.dating_chat_schema import ChatRequest, ChatResponse, ChatHistoryResponse

router = APIRouter(prefix="/dating-chat", tags=["Dating Chat"])
dating_chat_system = DatingChatSystem()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to the dating chat assistant"""
    # Use a single session for all users since user_id is removed
    user_id = "session"
    response = await dating_chat_system.get_chat_response(request.user_message, user_id)
    # Get the conversation history to include in response
    history = dating_chat_system.get_chat_history(user_id)
    return ChatResponse(assistant_message=response, conversation_history=history)

@router.get("/history", response_model=ChatHistoryResponse)
async def get_chat_history():
    """Get chat history for the current session"""
    history = dating_chat_system.get_chat_history("session")
    return ChatHistoryResponse(history=history)

@router.delete("/history")
async def clear_chat_history():
    """Clear chat history for the current session"""
    dating_chat_system.clear_chat_history("session")
    return {"message": "Chat history cleared"}