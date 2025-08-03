
import os
import redis
import motor.motor_asyncio
import logging
from pymongo.errors import OperationFailure

logger = logging.getLogger(__name__)

async def initialize_databases():
    """Initialize MongoDB collections and indexes, test Redis connection"""
    
    # Initialize MongoDB
    try:
        client = motor.motor_asyncio.AsyncIOMotorClient(
            os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        )
        db = client[os.getenv("MONGODB_DB", "conversational_ai")]
        
        # Create collections if they don't exist
        collections = ["short_term_memory", "long_term_memory", "chat_history"]
        
        existing_collections = await db.list_collection_names()
        for collection_name in collections:
            if collection_name not in existing_collections:
                await db.create_collection(collection_name)
                logger.info(f"Created collection: {collection_name}")
        
        # Create indexes 
        async def try_create_index(collection, index_spec, **options):
            """Try to create index, handle conflicts gracefully"""
            try:
                result = await collection.create_index(index_spec, **options)
                logger.info(f"Created index: {result}")
                return result
            except OperationFailure as e:
                if e.code == 86:  # IndexKeySpecsConflict
                    logger.warning(f"Index conflict detected: {e}")
                    logger.info("This usually means an index with same name but different options exists")
                    
                    return None
                elif "already exists" in str(e).lower():
                    logger.info(f"Index already exists (skipping): {e}")
                    return None
                else:
                    raise  
        
        # Create indexes
        logger.info("Creating indexes...")
        
        # Short-term memory indexes
        await try_create_index(
            db.short_term_memory, 
            [("user_id", 1), ("conversation_id", 1)]
            
        )
        await try_create_index(db.short_term_memory, [("updated_at", 1)])
        
        # Long-term memory indexes
        await try_create_index(db.long_term_memory, [("user_id", 1)])
        await try_create_index(db.long_term_memory, [("updated_at", 1)])
        
        # Chat history indexes
        await try_create_index(db.chat_history, [("user_id", 1), ("conversation_id", 1)])
        await try_create_index(db.chat_history, [("timestamp", -1)])
        
        logger.info("MongoDB initialization completed!")
        client.close()
        
    except Exception as e:
        logger.error(f"MongoDB initialization failed: {e}")
        raise
    
    # Initialize Redis
    try:
        redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 0)),
            username=os.getenv("REDIS_USERNAME", "default"),
            password=os.getenv("REDIS_PASSWORD"),
            decode_responses=True
        )
        
        # Test connection
        redis_client.ping()
        logger.info("Redis connection successful!")
        
        # Set up initial data
        redis_client.set("service_status", "initialized")
        redis_client.close()
        
    except Exception as e:
        logger.error(f"Redis initialization failed: {e}")
        raise