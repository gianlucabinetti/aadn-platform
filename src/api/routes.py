"""
AADN API Routes
Main API router with all endpoint definitions
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional

from ..core.exceptions import AADNException

# Create main API router
api_router = APIRouter()

# Import decoy manager and monitoring
from ..decoys.manager import decoy_manager
from ..decoys.models import (
    DecoyDeploymentRequest, DecoyUpdateRequest, DecoyListResponse,
    DecoyType, DecoyStatus
)
from ..monitoring.interaction_logger import interaction_logger


@api_router.get("/", summary="API Root", tags=["system"])
async def api_root():
    """API root endpoint"""
    return {
        "message": "AADN API v1.0",
        "description": "Adaptive AI-Driven Deception Network API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "decoys": "/decoys",
            "monitoring": "/monitoring",
            "intelligence": "/intelligence",
            "ai": "/ai",
            "alerts": "/alerts"
        }
    }


@api_router.get("/health", summary="Health Check", tags=["system"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "AADN API",
        "version": "1.0.0"
    }


@api_router.get("/status", summary="System Status", tags=["system"])
async def system_status():
    """Get system status and statistics"""
    try:
        # This will be implemented with actual system checks
        return {
            "status": "operational",
            "components": {
                "database": "healthy",
                "ai_engine": "healthy",
                "decoy_manager": "healthy",
                "monitoring": "healthy"
            },
            "statistics": {
                "active_decoys": 0,
                "total_interactions": 0,
                "active_alerts": 0,
                "threat_level": "low"
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system status: {str(e)}"
        )


# Decoy Management Endpoints

@api_router.get("/decoys", summary="List Decoys", tags=["decoys"], response_model=DecoyListResponse)
async def list_decoys(
    page: int = 1,
    page_size: int = 50,
    status: Optional[DecoyStatus] = None,
    type: Optional[DecoyType] = None
):
    """List all decoys with pagination and filtering"""
    try:
        return await decoy_manager.list_decoys(
            page=page,
            page_size=page_size,
            status_filter=status,
            type_filter=type
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list decoys: {str(e)}"
        )


@api_router.post("/decoys", summary="Create Decoy", tags=["decoys"])
async def create_decoy(request: DecoyDeploymentRequest):
    """Create and deploy a new decoy"""
    try:
        decoy = await decoy_manager.create_decoy(request)
        return {"message": "Decoy created successfully", "decoy": decoy}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create decoy: {str(e)}"
        )


@api_router.get("/decoys/{decoy_id}", summary="Get Decoy", tags=["decoys"])
async def get_decoy(decoy_id: str):
    """Get a specific decoy by ID"""
    try:
        decoy = await decoy_manager.get_decoy(decoy_id)
        return decoy
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Decoy not found: {str(e)}"
        )


@api_router.put("/decoys/{decoy_id}", summary="Update Decoy", tags=["decoys"])
async def update_decoy(decoy_id: str, request: DecoyUpdateRequest):
    """Update a decoy configuration"""
    try:
        decoy = await decoy_manager.update_decoy(decoy_id, request)
        return {"message": "Decoy updated successfully", "decoy": decoy}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update decoy: {str(e)}"
        )


@api_router.post("/decoys/{decoy_id}/start", summary="Start Decoy", tags=["decoys"])
async def start_decoy(decoy_id: str):
    """Start a decoy service"""
    try:
        success = await decoy_manager.start_decoy(decoy_id)
        return {"message": "Decoy started successfully", "success": success}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start decoy: {str(e)}"
        )


@api_router.post("/decoys/{decoy_id}/stop", summary="Stop Decoy", tags=["decoys"])
async def stop_decoy(decoy_id: str):
    """Stop a decoy service"""
    try:
        success = await decoy_manager.stop_decoy(decoy_id)
        return {"message": "Decoy stopped successfully", "success": success}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop decoy: {str(e)}"
        )


@api_router.delete("/decoys/{decoy_id}", summary="Delete Decoy", tags=["decoys"])
async def delete_decoy(decoy_id: str):
    """Delete a decoy"""
    try:
        success = await decoy_manager.delete_decoy(decoy_id)
        return {"message": "Decoy deleted successfully", "success": success}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete decoy: {str(e)}"
        )


@api_router.get("/decoys/{decoy_id}/health", summary="Decoy Health Check", tags=["decoys"])
async def check_decoy_health(decoy_id: str):
    """Perform health check on a decoy"""
    try:
        health = await decoy_manager.health_check_decoy(decoy_id)
        return health
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )


@api_router.get("/decoys/{decoy_id}/stats", summary="Decoy Statistics", tags=["decoys"])
async def get_decoy_stats(decoy_id: str):
    """Get decoy interaction statistics"""
    try:
        stats = await decoy_manager.get_decoy_stats(decoy_id)
        return stats
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get decoy stats: {str(e)}"
        )


# Monitoring Endpoints

@api_router.get("/monitoring", summary="Monitoring Overview", tags=["monitoring"])
async def monitoring_overview():
    """Get monitoring overview"""
    try:
        stats = await interaction_logger.get_interaction_stats()
        top_attackers = await interaction_logger.get_top_attackers(limit=5)
        
        return {
            "stats": stats,
            "top_attackers": top_attackers,
            "status": "active"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get monitoring overview: {str(e)}"
        )


@api_router.get("/monitoring/interactions", summary="Get Interactions", tags=["monitoring"])
async def get_interactions(
    decoy_id: Optional[str] = None,
    source_ip: Optional[str] = None,
    interaction_type: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """Get interaction logs with filtering"""
    try:
        interactions = await interaction_logger.get_interactions(
            decoy_id=decoy_id,
            source_ip=source_ip,
            interaction_type=interaction_type,
            limit=limit,
            offset=offset
        )
        return {"interactions": interactions, "total": len(interactions)}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get interactions: {str(e)}"
        )


@api_router.get("/monitoring/stats", summary="Interaction Statistics", tags=["monitoring"])
async def get_interaction_stats(
    decoy_id: Optional[str] = None,
    time_range_hours: int = 24
):
    """Get interaction statistics"""
    try:
        stats = await interaction_logger.get_interaction_stats(
            decoy_id=decoy_id,
            time_range_hours=time_range_hours
        )
        return stats
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get interaction stats: {str(e)}"
        )


@api_router.get("/monitoring/attackers", summary="Top Attackers", tags=["monitoring"])
async def get_top_attackers(
    limit: int = 10,
    time_range_hours: int = 24
):
    """Get top attacking IP addresses"""
    try:
        attackers = await interaction_logger.get_top_attackers(
            limit=limit,
            time_range_hours=time_range_hours
        )
        return {"attackers": attackers}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get top attackers: {str(e)}"
        )


# Placeholder endpoints for future implementation

@api_router.get("/intelligence", summary="Threat Intelligence", tags=["intelligence"])
async def threat_intelligence():
    """Get threat intelligence summary"""
    return {
        "threats": [],
        "indicators": [],
        "message": "Threat intelligence endpoints will be implemented in Phase 2"
    }


@api_router.get("/ai", summary="AI Status", tags=["ai"])
async def ai_status():
    """Get AI/ML system status"""
    return {
        "models": [],
        "training_status": "idle",
        "message": "AI/ML endpoints will be implemented in Phase 2"
    }


@api_router.get("/alerts", summary="Active Alerts", tags=["alerts"])
async def active_alerts():
    """Get active security alerts"""
    return {
        "alerts": [],
        "total": 0,
        "message": "Alert management endpoints will be implemented in Phase 1"
    } 