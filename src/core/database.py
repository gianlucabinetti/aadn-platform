"""
AADN Database Management
Database connections and initialization for AADN
"""

import asyncio
import logging
from typing import Dict, Optional, Any

import pymongo
import redis
from motor.motor_asyncio import AsyncIOMotorClient

from .config import get_settings

logger = logging.getLogger("aadn.database")

# Global database connections
_mongodb_client: Optional[AsyncIOMotorClient] = None
_mongodb_db = None
_redis_client: Optional[redis.Redis] = None


async def init_databases():
    """Initialize all database connections"""
    settings = get_settings()
    
    try:
        # Initialize MongoDB
        await init_mongodb(settings.mongodb_url, settings.mongodb_database)
        logger.info("MongoDB initialized successfully")
        
        # Initialize Redis
        await init_redis(settings.redis_url)
        logger.info("Redis initialized successfully")
        
        logger.info("All databases initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize databases: {e}")
        raise


async def init_mongodb(url: str, database_name: str):
    """Initialize MongoDB connection"""
    global _mongodb_client, _mongodb_db
    
    try:
        _mongodb_client = AsyncIOMotorClient(url)
        _mongodb_db = _mongodb_client[database_name]
        
        # Test connection
        await _mongodb_client.admin.command('ping')
        
        # Create indexes
        await create_mongodb_indexes()
        
    except Exception as e:
        logger.error(f"Failed to initialize MongoDB: {e}")
        raise


async def init_redis(url: str):
    """Initialize Redis connection"""
    global _redis_client
    
    try:
        _redis_client = redis.from_url(url, decode_responses=True)
        
        # Test connection
        _redis_client.ping()
        
    except Exception as e:
        logger.error(f"Failed to initialize Redis: {e}")
        raise


async def create_mongodb_indexes():
    """Create MongoDB indexes for optimal performance"""
    if not _mongodb_db:
        return
    
    try:
        # Interactions collection indexes
        interactions = _mongodb_db.interactions
        await interactions.create_index([("timestamp", -1)])
        await interactions.create_index([("source_ip", 1)])
        await interactions.create_index([("decoy_id", 1)])
        await interactions.create_index([("interaction_type", 1)])
        
        # Threats collection indexes
        threats = _mongodb_db.threats
        await threats.create_index([("created_at", -1)])
        await threats.create_index([("source_ip", 1)])
        await threats.create_index([("level", 1)])
        await threats.create_index([("category", 1)])
        
        # Alerts collection indexes
        alerts = _mongodb_db.alerts
        await alerts.create_index([("created_at", -1)])
        await alerts.create_index([("severity", 1)])
        await alerts.create_index([("rule_id", 1)])
        
        # Decoys collection indexes
        decoys = _mongodb_db.decoys
        await decoys.create_index([("name", 1)], unique=True)
        await decoys.create_index([("type", 1)])
        await decoys.create_index([("status", 1)])
        
        logger.info("MongoDB indexes created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create MongoDB indexes: {e}")


async def check_database_connections() -> Dict[str, bool]:
    """Check the status of all database connections"""
    status = {}
    
    # Check MongoDB
    try:
        if _mongodb_client:
            await _mongodb_client.admin.command('ping')
            status["mongodb"] = True
        else:
            status["mongodb"] = False
    except Exception:
        status["mongodb"] = False
    
    # Check Redis
    try:
        if _redis_client:
            _redis_client.ping()
            status["redis"] = True
        else:
            status["redis"] = False
    except Exception:
        status["redis"] = False
    
    return status


def get_mongodb_db():
    """Get MongoDB database instance"""
    if not _mongodb_db:
        raise RuntimeError("MongoDB not initialized. Call init_databases() first.")
    return _mongodb_db


def get_redis_client():
    """Get Redis client instance"""
    if not _redis_client:
        raise RuntimeError("Redis not initialized. Call init_databases() first.")
    return _redis_client


async def close_databases():
    """Close all database connections"""
    global _mongodb_client, _mongodb_db, _redis_client
    
    try:
        if _mongodb_client:
            _mongodb_client.close()
            _mongodb_client = None
            _mongodb_db = None
            logger.info("MongoDB connection closed")
        
        if _redis_client:
            _redis_client.close()
            _redis_client = None
            logger.info("Redis connection closed")
            
    except Exception as e:
        logger.error(f"Error closing database connections: {e}")


# Database utility functions
async def store_interaction(interaction_data: Dict[str, Any]) -> str:
    """Store an interaction in the database"""
    db = get_mongodb_db()
    result = await db.interactions.insert_one(interaction_data)
    return str(result.inserted_id)


async def get_interactions(
    limit: int = 100,
    skip: int = 0,
    source_ip: Optional[str] = None,
    decoy_id: Optional[str] = None,
    start_time: Optional[Any] = None,
    end_time: Optional[Any] = None
) -> list:
    """Get interactions from the database"""
    db = get_mongodb_db()
    
    # Build query
    query = {}
    if source_ip:
        query["source_ip"] = source_ip
    if decoy_id:
        query["decoy_id"] = decoy_id
    if start_time or end_time:
        query["timestamp"] = {}
        if start_time:
            query["timestamp"]["$gte"] = start_time
        if end_time:
            query["timestamp"]["$lte"] = end_time
    
    # Execute query
    cursor = db.interactions.find(query).sort("timestamp", -1).skip(skip).limit(limit)
    interactions = await cursor.to_list(length=limit)
    
    # Convert ObjectId to string
    for interaction in interactions:
        interaction["_id"] = str(interaction["_id"])
    
    return interactions


async def store_threat_analysis(threat_data: Dict[str, Any]) -> str:
    """Store a threat analysis in the database"""
    db = get_mongodb_db()
    result = await db.threats.insert_one(threat_data)
    return str(result.inserted_id)


async def store_alert(alert_data: Dict[str, Any]) -> str:
    """Store an alert in the database"""
    db = get_mongodb_db()
    result = await db.alerts.insert_one(alert_data)
    return str(result.inserted_id)


async def get_decoy_stats() -> Dict[str, Any]:
    """Get decoy statistics from the database"""
    db = get_mongodb_db()
    
    # Get total decoys by status
    pipeline = [
        {"$group": {"_id": "$status", "count": {"$sum": 1}}}
    ]
    status_counts = await db.decoys.aggregate(pipeline).to_list(length=None)
    
    # Get interaction counts by decoy
    pipeline = [
        {"$group": {"_id": "$decoy_id", "count": {"$sum": 1}}}
    ]
    interaction_counts = await db.interactions.aggregate(pipeline).to_list(length=None)
    
    return {
        "status_counts": {item["_id"]: item["count"] for item in status_counts},
        "interaction_counts": {item["_id"]: item["count"] for item in interaction_counts}
    }


# Cache utilities using Redis
async def cache_set(key: str, value: str, expire: int = 3600):
    """Set a value in Redis cache"""
    redis_client = get_redis_client()
    redis_client.setex(key, expire, value)


async def cache_get(key: str) -> Optional[str]:
    """Get a value from Redis cache"""
    redis_client = get_redis_client()
    return redis_client.get(key)


async def cache_delete(key: str):
    """Delete a value from Redis cache"""
    redis_client = get_redis_client()
    redis_client.delete(key) 