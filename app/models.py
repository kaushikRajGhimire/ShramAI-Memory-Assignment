from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class Message(BaseModel):
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ChatRequest(BaseModel):
    message: str
    user_id: str
    conversation_id: str 

class ChatResponse(BaseModel):
    response: str
    user_id: str
    conversation_id: str
    timestamp: datetime

class LoginRequest(BaseModel):
    user_id: str
    conversation_id: str

class LogoutRequest(BaseModel):
    user_id: str
    conversation_id: str