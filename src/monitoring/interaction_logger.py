"""
AADN Interaction Logger
Logs and manages decoy interactions
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from ..core.database import store_interaction, get_interactions as db_get_interactions
from ..core.logging_config import log_interaction as log_security_interaction

logger = logging.getLogger("aadn.monitoring")


async def log_interaction(
    decoy_id: str,
    source_ip: str,
    source_port: int,
    interaction_type: str,
    data: Dict[str, Any]
) -> str:
    """Log a decoy interaction"""
    
    interaction_data = {
        "decoy_id": decoy_id,
        "source_ip": source_ip,
        "source_port": source_port,
        "interaction_type": interaction_type,
        "data": data,
        "timestamp": datetime.utcnow()
    }
    
    try:
        # Store in database
        interaction_id = await store_interaction(interaction_data)
        
        # Log security event
        log_security_interaction(
            decoy_id=decoy_id,
            source_ip=source_ip,
            interaction_type=interaction_type,
            data=data
        )
        
        logger.info(f"Interaction logged: {interaction_id} from {source_ip}")
        return interaction_id
        
    except Exception as e:
        logger.error(f"Failed to log interaction: {e}")
        raise


async def get_interactions(
    limit: int = 100,
    skip: int = 0,
    source_ip: Optional[str] = None,
    decoy_id: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """Get interactions from the database"""
    
    try:
        interactions = await db_get_interactions(
            limit=limit,
            skip=skip,
            source_ip=source_ip,
            decoy_id=decoy_id,
            start_time=start_time,
            end_time=end_time
        )
        
        return interactions
        
    except Exception as e:
        logger.error(f"Failed to get interactions: {e}")
        return []


async def get_interaction_stats() -> Dict[str, Any]:
    """Get interaction statistics"""
    
    try:
        # Get recent interactions (last 24 hours)
        end_time = datetime.utcnow()
        start_time = end_time.replace(hour=0, minute=0, second=0, microsecond=0)
        
        recent_interactions = await get_interactions(
            start_time=start_time,
            end_time=end_time,
            limit=1000
        )
        
        # Calculate statistics
        total_interactions = len(recent_interactions)
        unique_ips = len(set(i.get("source_ip", "") for i in recent_interactions))
        
        # Group by interaction type
        type_counts = {}
        for interaction in recent_interactions:
            interaction_type = interaction.get("interaction_type", "unknown")
            type_counts[interaction_type] = type_counts.get(interaction_type, 0) + 1
        
        # Group by decoy
        decoy_counts = {}
        for interaction in recent_interactions:
            decoy_id = interaction.get("decoy_id", "unknown")
            decoy_counts[decoy_id] = decoy_counts.get(decoy_id, 0) + 1
        
        return {
            "total_interactions": total_interactions,
            "unique_source_ips": unique_ips,
            "interaction_types": type_counts,
            "decoy_interactions": decoy_counts,
            "time_period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get interaction stats: {e}")
        return {
            "total_interactions": 0,
            "unique_source_ips": 0,
            "interaction_types": {},
            "decoy_interactions": {},
            "error": str(e)
        } 