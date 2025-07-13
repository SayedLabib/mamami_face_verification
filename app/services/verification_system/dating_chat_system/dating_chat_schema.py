from pydantic import BaseModel
from typing import List, Dict, Optional

class ChatRequest(BaseModel):
    user_message: str

class ChatResponse(BaseModel):
    assistant_message: str
    conversation_history: Optional[List[Dict[str, str]]] = []

class ChatHistoryResponse(BaseModel):
    history: List[Dict[str, str]]