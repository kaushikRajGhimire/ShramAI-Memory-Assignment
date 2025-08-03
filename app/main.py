import os
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from database_init import initialize_databases
from memory_manager import MemoryManager
from agent import process_conversation
from models import ChatRequest, ChatResponse, LoginRequest, LogoutRequest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize memory manager
memory_manager = MemoryManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    logger.info("Starting up: Initializing databases...")
    await initialize_databases()
    logger.info("Startup complete!")
    yield
    logger.info("Shutting down: Closing connections...")
    memory_manager.close()
    logger.info("Shutdown complete!")

# Create FastAPI app
app = FastAPI(
    title="Conversational AI Microservice",
    description="A sophisticated conversational AI with memory management",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Conversational AI Microservice",
        "timestamp": datetime.utcnow()
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint"""
    try:
        logger.info(f"Chat request: user={request.user_id}, conv={request.conversation_id}, msg='{request.message[:50]}...'")
        
        # Process the conversation
        response = await process_conversation(
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            user_message=request.message
        )
        
        logger.info(f"Chat response: user={request.user_id}, response='{response[:50]}...'")
        
        return ChatResponse(
            response=response,
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/login")
async def login_endpoint(request: LoginRequest):
    """Handle user login - restore memory from MongoDB to Redis"""
    try:
        logger.info(f"Login request: user={request.user_id}, conv={request.conversation_id}")
        
        # Load short-term memory from MongoDB to Redis
        await memory_manager.load_short_term_on_login(
            user_id=request.user_id,
            conversation_id=request.conversation_id
        )
        
        return {
            "status": "success",
            "message": "Memory restored successfully",
            "user_id": request.user_id,
            "conversation_id": request.conversation_id,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Login endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Login failed")

@app.post("/logout")
async def logout_endpoint(request: LogoutRequest):
    """Handle user logout - save memory to MongoDB and clear Redis"""
    try:
        logger.info(f"Logout request: user={request.user_id}, conv={request.conversation_id}")
        
        # Save short-term memory to MongoDB
        await memory_manager.save_short_term_on_logout(
            user_id=request.user_id,
            conversation_id=request.conversation_id
        )
        
        # Clear Redis data
        await memory_manager.clear_redis_on_logout(
            user_id=request.user_id,
            conversation_id=request.conversation_id
        )
        
        return {
            "status": "success",
            "message": "Logout completed successfully",
            "user_id": request.user_id,
            "conversation_id": request.conversation_id,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Logout endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Logout failed")

@app.get("/memory/{user_id}/{conversation_id}")
async def get_memory_endpoint(user_id: str, conversation_id: str):
    """Get current memory state for debugging"""
    try:
        context = await memory_manager.get_context_for_search(user_id, conversation_id)
        
        return {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "memory_state": context,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Memory endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get memory state")

@app.get("/history/{user_id}")
async def get_history_endpoint(user_id: str, skip: int = 0, limit: int = 20):
    """Get paginated chat history from MongoDB"""
    try:
        cursor = memory_manager.chat_history.find(
            {"user_id": user_id}
        ).sort("timestamp", -1).skip(skip).limit(limit)
        
        messages = []
        async for doc in cursor:
            doc["_id"] = str(doc["_id"])
            messages.append(doc)
        
        total = await memory_manager.chat_history.count_documents({"user_id": user_id})
        
        return {
            "user_id": user_id,
            "messages": messages,
            "total": total,
            "skip": skip,
            "limit": limit,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"History endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get chat history")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)