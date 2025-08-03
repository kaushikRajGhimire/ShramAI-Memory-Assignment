import os
import json
import redis
import motor.motor_asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from models import Message

logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self):
        # Redis connection
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 0)),
            username=os.getenv("REDIS_USERNAME", "default"),
            password=os.getenv("REDIS_PASSWORD"),
            decode_responses=True
        )
        
        # MongoDB connection
        self.mongo_client = motor.motor_asyncio.AsyncIOMotorClient(
            os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        )
        self.db = self.mongo_client[os.getenv("MONGODB_DB", "conversational_ai")]
        self.short_term_memory = self.db["short_term_memory"]
        self.long_term_memory = self.db["long_term_memory"]
        self.chat_history = self.db["chat_history"]

    def _get_redis_key(self, key_type: str, user_id: str, conversation_id: str = None) -> str:
        """Generate Redis keys based on type"""
        if key_type == "short_term":
            return f"short_term:{user_id}:{conversation_id}"
        elif key_type == "slider_summary":
            return f"slider_summary:{user_id}:{conversation_id}"
        elif key_type == "long_term":
            return f"long_term:{user_id}"  # No conversation_id for long-term
        elif key_type == "chat_history":
            return f"chat_history:{user_id}:{conversation_id}"
        elif key_type == "message_count":
            return f"message_count:{user_id}:{conversation_id}"
        else:
            raise ValueError(f"Unknown key type: {key_type}")

    async def add_message(self, user_id: str, conversation_id: str, message: Message):
        """Add message and trigger memory management"""
        try:
            # Serialize message
            msg_data = {
                "role": message.role,
                "content": message.content,
                "timestamp": message.timestamp.isoformat()
            }
            msg_json = json.dumps(msg_data)
            
            # 1. Add to chat history (MongoDB - persistent)
            await self.chat_history.insert_one({
                "user_id": user_id,
                "conversation_id": conversation_id,
                "role": message.role,
                "content": message.content,
                "timestamp": message.timestamp
            })
            
            # 2. Add to Redis chat history buffer (sliding window of 8 messages)
            chat_key = self._get_redis_key("chat_history", user_id, conversation_id)
            self.redis_client.lpush(chat_key, msg_json)
            self.redis_client.ltrim(chat_key, 0, 7)  # Keep only last 8 messages
            
            # 3. Update short-term memory (last 3-4 messages)
            await self._update_short_term_memory(user_id, conversation_id)
            
            # 4. Increment message count and trigger actions
            count_key = self._get_redis_key("message_count", user_id, conversation_id)
            count = self.redis_client.incr(count_key)
            
            logger.info(f"Message count for {user_id}/{conversation_id}: {count}")
            
            # 5. Generate slider summary every 4th message
            if count % 4 == 0:
                await self._generate_slider_summary(user_id, conversation_id)
            
            # 6. Generate long-term memory every 8th message
            if count % 8 == 0:
                await self._generate_long_term_memory(user_id, conversation_id)
                
        except Exception as e:
            logger.error(f"Error adding message: {e}")
            raise

    async def _update_short_term_memory(self, user_id: str, conversation_id: str):
        """Update short-term memory with last 3-4 messages"""
        try:
            chat_key = self._get_redis_key("chat_history", user_id, conversation_id)
            short_key = self._get_redis_key("short_term", user_id, conversation_id)
            
            # Get last 4 messages from chat history
            recent_messages = self.redis_client.lrange(chat_key, 0, 3)
            
            # Clear and update short-term memory
            self.redis_client.delete(short_key)
            for msg_json in reversed(recent_messages):
                self.redis_client.rpush(short_key, msg_json)
                
        except Exception as e:
            logger.error(f"Error updating short-term memory: {e}")

    async def _generate_slider_summary(self, user_id: str, conversation_id: str):
        """Generate slider summary of older conversations every 4th message"""
        try:
            chat_key = self._get_redis_key("chat_history", user_id, conversation_id)
            
            # Get messages older than the last 4 (for summarization)
            older_messages = self.redis_client.lrange(chat_key, 4, -1)
            
            if not older_messages:
                return
            
            # Convert to conversation format
            conversation_text = []
            for msg_json in reversed(older_messages):
                try:
                    msg_data = json.loads(msg_json)
                    conversation_text.append(f"{msg_data['role']}: {msg_data['content']}")
                except:
                    continue
            
            if not conversation_text:
                return
            
            # Generate summary using LLM
            from agent import get_llm
            llm = get_llm()
            
            prompt = f"""Summarize the following conversation in 2-3 sentences, focusing on the key topics and important information:

{chr(10).join(conversation_text)}

Summary:"""
            
            response = await llm.ainvoke(prompt)
            summary = response.content.strip()
            
            # Store summary in Redis
            summary_key = self._get_redis_key("slider_summary", user_id, conversation_id)
            self.redis_client.set(summary_key, summary)
            
            logger.info(f"Generated slider summary for {user_id}/{conversation_id}")
            
        except Exception as e:
            logger.error(f"Error generating slider summary: {e}")

    async def _generate_long_term_memory(self, user_id: str, conversation_id: str):
        """Generate long-term memory points every 8th message"""
        try:
            chat_key = self._get_redis_key("chat_history", user_id, conversation_id)
            
            # Get last 8 messages for long-term memory extraction
            recent_messages = self.redis_client.lrange(chat_key, 0, 7)
            
            if not recent_messages:
                return
            
            # Convert to conversation format
            conversation_text = []
            for msg_json in reversed(recent_messages):
                try:
                    msg_data = json.loads(msg_json)
                    conversation_text.append(f"{msg_data['role']}: {msg_data['content']}")
                except:
                    continue
            
            if not conversation_text:
                return
            
            # Generate key points using LLM
            from agent import get_llm
            llm = get_llm()
            
            prompt = f"""Extract exactly 5 key points from the following conversation. Focus on important information, preferences, facts, and context that would be useful to remember for future conversations.

{chr(10).join(conversation_text)}

Please provide exactly 5 key points in the following format:
- Point 1
- Point 2  
- Point 3
- Point 4
- Point 5

Key Points:"""
            
            response = await llm.ainvoke(prompt)
            content = response.content.strip()
            
            # Extract points from response
            points = []
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('- ') or line.startswith('• '):
                    points.append(line[2:].strip())
            
            # Ensure we have exactly 5 points
            while len(points) < 5:
                points.append(f"Context point {len(points) + 1}")
            points = points[:5]
            
            # Store in Redis (append to existing)
            long_term_key = self._get_redis_key("long_term", user_id)
            for point in points:
                self.redis_client.rpush(long_term_key, point)
            
            # Store in MongoDB
            await self.long_term_memory.insert_one({
                "user_id": user_id,
                "key_points": points,
                "source_conversation_id": conversation_id,
                "created_at": datetime.utcnow()
            })
            
            logger.info(f"Generated long-term memory for {user_id}")
            
        except Exception as e:
            logger.error(f"Error generating long-term memory: {e}")

    async def get_context_for_search(self, user_id: str, conversation_id: str) -> Dict[str, Any]:
        """Get all memory context for db_search tool"""
        try:
            context = {
                "short_term_messages": [],
                "slider_summary": "",
                "long_term_points": [],
                "recent_history": []
            }
            
            # Get short-term messages
            short_key = self._get_redis_key("short_term", user_id, conversation_id)
            short_messages = self.redis_client.lrange(short_key, 0, -1)
            for msg_json in short_messages:
                try:
                    context["short_term_messages"].append(json.loads(msg_json))
                except:
                    continue
            
            # Get slider summary
            summary_key = self._get_redis_key("slider_summary", user_id, conversation_id)
            context["slider_summary"] = self.redis_client.get(summary_key) or ""
            
            # Get long-term memory points
            long_term_key = self._get_redis_key("long_term", user_id)
            context["long_term_points"] = self.redis_client.lrange(long_term_key, 0, -1)
            
            # Get recent chat history
            chat_key = self._get_redis_key("chat_history", user_id, conversation_id)
            recent_messages = self.redis_client.lrange(chat_key, 0, 7)
            for msg_json in reversed(recent_messages):
                try:
                    context["recent_history"].append(json.loads(msg_json))
                except:
                    continue
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting context: {e}")
            return {
                "short_term_messages": [],
                "slider_summary": "",
                "long_term_points": [],
                "recent_history": []
            }

    async def save_short_term_on_logout(self, user_id: str, conversation_id: str):
        """Save short-term memory to MongoDB on logout"""
        try:
            # Get short-term messages
            short_key = self._get_redis_key("short_term", user_id, conversation_id)
            messages = []
            for msg_json in self.redis_client.lrange(short_key, 0, -1):
                try:
                    messages.append(json.loads(msg_json))
                except:
                    continue
            
            # Get slider summary
            summary_key = self._get_redis_key("slider_summary", user_id, conversation_id)
            slider_summary = self.redis_client.get(summary_key) or ""
            
            # Save to MongoDB
            if messages or slider_summary:
                await self.short_term_memory.update_one(
                    {"user_id": user_id, "conversation_id": conversation_id},
                    {
                        "$set": {
                            "messages": messages,
                            "slider_summary": slider_summary,
                            "updated_at": datetime.utcnow()
                        }
                    },
                    upsert=True
                )
                
            logger.info(f"Saved short-term memory for {user_id}/{conversation_id}")
            
        except Exception as e:
            logger.error(f"Error saving short-term memory: {e}")

    async def load_short_term_on_login(self, user_id: str, conversation_id: str):
        """Load short-term memory from MongoDB on login"""
        try:
            # Get from MongoDB
            doc = await self.short_term_memory.find_one({
                "user_id": user_id,
                "conversation_id": conversation_id
            })
            
            if doc:
                # Restore short-term messages
                short_key = self._get_redis_key("short_term", user_id, conversation_id)
                self.redis_client.delete(short_key)
                for msg_data in doc.get("messages", []):
                    self.redis_client.rpush(short_key, json.dumps(msg_data, default=str))
                
                # Restore slider summary
                if doc.get("slider_summary"):
                    summary_key = self._get_redis_key("slider_summary", user_id, conversation_id)
                    self.redis_client.set(summary_key, doc["slider_summary"])
                
                logger.info(f"Loaded short-term memory for {user_id}/{conversation_id}")
                
        except Exception as e:
            logger.error(f"Error loading short-term memory: {e}")

    async def clear_redis_on_logout(self, user_id: str, conversation_id: str):
        """Clear Redis data on logout"""
        try:
            keys_to_clear = [
                self._get_redis_key("short_term", user_id, conversation_id),
                self._get_redis_key("slider_summary", user_id, conversation_id),
                self._get_redis_key("chat_history", user_id, conversation_id),
                self._get_redis_key("message_count", user_id, conversation_id)
            ]
            
            for key in keys_to_clear:
                self.redis_client.delete(key)
                
            logger.info(f"Cleared Redis data for {user_id}/{conversation_id}")
            
        except Exception as e:
            logger.error(f"Error clearing Redis data: {e}")

    def close(self):
        """Close connections"""
        try:
            self.redis_client.close()
            self.mongo_client.close()
        except Exception as e:
            logger.error(f"Error closing connections: {e}")