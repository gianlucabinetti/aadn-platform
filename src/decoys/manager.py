"""
AADN Decoy Manager
Handles decoy lifecycle management, deployment, and monitoring
"""

import asyncio
import logging
import socket
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import uuid4

from ..core.database import get_mongodb_db
from ..core.exceptions import DecoyError, DecoyNotFoundError, DecoyDeploymentError
from ..core.logging_config import get_decoy_logger, log_system_event, SecurityEvent
from .models import (
    Decoy, DecoyType, DecoyStatus, DecoyConfiguration, DecoyMetadata,
    DecoyDeploymentRequest, DecoyUpdateRequest, DecoyListResponse,
    DecoyHealthCheck, DEFAULT_CONFIGURATIONS
)
from .services import DecoyServiceFactory

logger = get_decoy_logger()


class DecoyManager:
    """Manages decoy lifecycle and operations"""
    
    def __init__(self):
        self.service_factory = DecoyServiceFactory()
        self._running_decoys: Dict[str, Any] = {}
        self._health_check_tasks: Dict[str, asyncio.Task] = {}
    
    async def create_decoy(
        self,
        request: DecoyDeploymentRequest,
        user_id: str = "system"
    ) -> Decoy:
        """Create a new decoy"""
        try:
            logger.info(f"Creating decoy: {request.name} of type {request.type}")
            
            # Generate configuration if not provided
            if not request.configuration:
                if request.type in DEFAULT_CONFIGURATIONS:
                    configuration = DEFAULT_CONFIGURATIONS[request.type].copy()
                else:
                    raise DecoyDeploymentError(f"No default configuration for decoy type: {request.type}")
            else:
                configuration = request.configuration
            
            # Auto-assign host if not provided
            if not request.host:
                request.host = await self._get_available_host()
            
            # Auto-assign port if needed
            if hasattr(configuration, 'port') and not await self._is_port_available(request.host, configuration.port):
                configuration.port = await self._get_available_port(request.host)
            
            # Create decoy instance
            decoy = Decoy(
                name=request.name,
                type=request.type,
                host=request.host,
                configuration=configuration,
                metadata=request.metadata or DecoyMetadata()
            )
            
            # Save to database
            db = get_mongodb_db()
            collection = db.decoys
            
            decoy_dict = decoy.dict()
            await collection.insert_one(decoy_dict)
            
            # Log system event
            log_system_event(
                SecurityEvent.CONFIG_CHANGE,
                f"Decoy deployed: {decoy.name}",
                decoy_id=decoy.id,
                user_id=user_id
            )
            
            # Auto-start if requested
            if request.auto_start:
                await self.start_decoy(decoy.id, user_id)
            
            logger.info(f"Decoy created successfully: {decoy.id}")
            return decoy
            
        except Exception as e:
            logger.error(f"Failed to create decoy: {e}")
            raise DecoyDeploymentError(f"Failed to create decoy: {str(e)}")
    
    async def get_decoy(self, decoy_id: str) -> Decoy:
        """Get a decoy by ID"""
        try:
            db = get_mongodb_db()
            collection = db.decoys
            
            decoy_data = await collection.find_one({"id": decoy_id})
            if not decoy_data:
                raise DecoyNotFoundError(decoy_id)
            
            return Decoy(**decoy_data)
            
        except DecoyNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to get decoy {decoy_id}: {e}")
            raise DecoyError(f"Failed to get decoy: {str(e)}")
    
    async def list_decoys(
        self,
        page: int = 1,
        page_size: int = 50,
        status_filter: Optional[DecoyStatus] = None,
        type_filter: Optional[DecoyType] = None
    ) -> DecoyListResponse:
        """List decoys with pagination and filtering"""
        try:
            db = get_mongodb_db()
            collection = db.decoys
            
            # Build filter query
            query = {}
            if status_filter:
                query["status"] = status_filter
            if type_filter:
                query["type"] = type_filter
            
            # Calculate pagination
            skip = (page - 1) * page_size
            
            # Get total count
            total = await collection.count_documents(query)
            
            # Get decoys
            cursor = collection.find(query).skip(skip).limit(page_size).sort("created_at", -1)
            decoy_data = await cursor.to_list(length=page_size)
            
            decoys = [Decoy(**data) for data in decoy_data]
            
            return DecoyListResponse(
                decoys=decoys,
                total=total,
                page=page,
                page_size=page_size,
                has_next=skip + page_size < total
            )
            
        except Exception as e:
            logger.error(f"Failed to list decoys: {e}")
            raise DecoyError(f"Failed to list decoys: {str(e)}")
    
    async def update_decoy(
        self,
        decoy_id: str,
        request: DecoyUpdateRequest,
        user_id: str = "system"
    ) -> Decoy:
        """Update a decoy"""
        try:
            # Get existing decoy
            decoy = await self.get_decoy(decoy_id)
            
            # Update fields
            update_data = {}
            if request.name:
                update_data["name"] = request.name
            if request.configuration:
                update_data["configuration"] = request.configuration.dict()
            if request.metadata:
                update_data["metadata"] = request.metadata.dict()
            if request.status:
                update_data["status"] = request.status
            
            update_data["updated_at"] = datetime.utcnow()
            
            # Update in database
            db = get_mongodb()
            collection = db[Collections.DECOYS]
            
            await collection.update_one(
                {"id": decoy_id},
                {"$set": update_data}
            )
            
            # Get updated decoy
            updated_decoy = await self.get_decoy(decoy_id)
            
            # Log audit event
            log_audit_event(
                AuditEvent.CONFIG_CHANGE,
                user_id,
                {"decoy_id": decoy_id, "changes": list(update_data.keys())}
            )
            
            logger.info(f"Decoy updated successfully: {decoy_id}")
            return updated_decoy
            
        except DecoyNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to update decoy {decoy_id}: {e}")
            raise DecoyError(f"Failed to update decoy: {str(e)}")
    
    async def start_decoy(self, decoy_id: str, user_id: str = "system") -> bool:
        """Start a decoy service"""
        try:
            decoy = await self.get_decoy(decoy_id)
            
            if decoy.status == DecoyStatus.ACTIVE:
                logger.warning(f"Decoy {decoy_id} is already active")
                return True
            
            logger.info(f"Starting decoy: {decoy_id}")
            
            # Update status to deploying
            await self._update_decoy_status(decoy_id, DecoyStatus.DEPLOYING)
            
            # Create and start the service
            service = self.service_factory.create_service(decoy.type, decoy.configuration)
            
            # Start the service
            await service.start(decoy.host, decoy.configuration.port)
            
            # Store running service
            self._running_decoys[decoy_id] = {
                "service": service,
                "decoy": decoy,
                "started_at": datetime.utcnow()
            }
            
            # Update status to active
            await self._update_decoy_status(decoy_id, DecoyStatus.ACTIVE, deployed_at=datetime.utcnow())
            
            # Start health check monitoring
            await self._start_health_monitoring(decoy_id)
            
            logger.info(f"Decoy started successfully: {decoy_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start decoy {decoy_id}: {e}")
            await self._update_decoy_status(decoy_id, DecoyStatus.ERROR, error=str(e))
            raise DecoyDeploymentError(f"Failed to start decoy: {str(e)}")
    
    async def stop_decoy(self, decoy_id: str, user_id: str = "system") -> bool:
        """Stop a decoy service"""
        try:
            decoy = await self.get_decoy(decoy_id)
            
            if decoy.status not in [DecoyStatus.ACTIVE, DecoyStatus.ERROR]:
                logger.warning(f"Decoy {decoy_id} is not running")
                return True
            
            logger.info(f"Stopping decoy: {decoy_id}")
            
            # Update status to terminating
            await self._update_decoy_status(decoy_id, DecoyStatus.TERMINATING)
            
            # Stop health monitoring
            await self._stop_health_monitoring(decoy_id)
            
            # Stop the service if running
            if decoy_id in self._running_decoys:
                service_info = self._running_decoys[decoy_id]
                service = service_info["service"]
                
                await service.stop()
                del self._running_decoys[decoy_id]
            
            # Update status to inactive
            await self._update_decoy_status(decoy_id, DecoyStatus.INACTIVE)
            
            # Log audit event
            log_audit_event(
                AuditEvent.DECOY_REMOVED,
                user_id,
                {"decoy_id": decoy_id}
            )
            
            logger.info(f"Decoy stopped successfully: {decoy_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop decoy {decoy_id}: {e}")
            raise DecoyError(f"Failed to stop decoy: {str(e)}")
    
    async def delete_decoy(self, decoy_id: str, user_id: str = "system") -> bool:
        """Delete a decoy"""
        try:
            # Stop the decoy first
            await self.stop_decoy(decoy_id, user_id)
            
            # Delete from database
            db = get_mongodb()
            collection = db[Collections.DECOYS]
            
            result = await collection.delete_one({"id": decoy_id})
            if result.deleted_count == 0:
                raise DecoyNotFoundError(decoy_id)
            
            logger.info(f"Decoy deleted successfully: {decoy_id}")
            return True
            
        except DecoyNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to delete decoy {decoy_id}: {e}")
            raise DecoyError(f"Failed to delete decoy: {str(e)}")
    
    async def health_check_decoy(self, decoy_id: str) -> DecoyHealthCheck:
        """Perform health check on a decoy"""
        try:
            decoy = await self.get_decoy(decoy_id)
            
            start_time = datetime.utcnow()
            
            # Check if service is responsive
            is_healthy = await self._check_service_health(decoy.host, decoy.configuration.port)
            
            response_time = (datetime.utcnow() - start_time).total_seconds()
            
            health_check = DecoyHealthCheck(
                decoy_id=decoy_id,
                status="healthy" if is_healthy else "unhealthy",
                response_time=response_time,
                details={
                    "host": decoy.host,
                    "port": decoy.configuration.port,
                    "service_type": decoy.type
                }
            )
            
            # Update last health check time
            await self._update_decoy_health(decoy_id, health_check)
            
            return health_check
            
        except Exception as e:
            logger.error(f"Health check failed for decoy {decoy_id}: {e}")
            return DecoyHealthCheck(
                decoy_id=decoy_id,
                status="error",
                response_time=0.0,
                errors=[str(e)]
            )
    
    async def get_decoy_stats(self, decoy_id: str) -> Dict[str, Any]:
        """Get decoy interaction statistics"""
        try:
            decoy = await self.get_decoy(decoy_id)
            
            # Get interaction data from database
            db = get_mongodb()
            interactions_collection = db[Collections.INTERACTIONS]
            
            # Count total interactions
            total_interactions = await interactions_collection.count_documents({"decoy_id": decoy_id})
            
            # Count unique source IPs
            unique_sources = len(await interactions_collection.distinct("source_ip", {"decoy_id": decoy_id}))
            
            # Get recent interactions
            recent_interactions = await interactions_collection.find(
                {"decoy_id": decoy_id}
            ).sort("timestamp", -1).limit(10).to_list(length=10)
            
            return {
                "decoy_id": decoy_id,
                "total_interactions": total_interactions,
                "unique_sources": unique_sources,
                "recent_interactions": recent_interactions,
                "status": decoy.status,
                "uptime": self._calculate_uptime(decoy)
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats for decoy {decoy_id}: {e}")
            raise DecoyError(f"Failed to get decoy stats: {str(e)}")
    
    # Private helper methods
    
    async def _update_decoy_status(
        self,
        decoy_id: str,
        status: DecoyStatus,
        deployed_at: Optional[datetime] = None,
        error: Optional[str] = None
    ):
        """Update decoy status in database"""
        update_data = {
            "status": status,
            "updated_at": datetime.utcnow()
        }
        
        if deployed_at:
            update_data["deployed_at"] = deployed_at
        if error:
            update_data["last_error"] = error
        
        db = get_mongodb()
        collection = db[Collections.DECOYS]
        
        await collection.update_one(
            {"id": decoy_id},
            {"$set": update_data}
        )
    
    async def _update_decoy_health(self, decoy_id: str, health_check: DecoyHealthCheck):
        """Update decoy health information"""
        update_data = {
            "health_status": health_check.status,
            "last_health_check": health_check.timestamp,
            "updated_at": datetime.utcnow()
        }
        
        db = get_mongodb()
        collection = db[Collections.DECOYS]
        
        await collection.update_one(
            {"id": decoy_id},
            {"$set": update_data}
        )
    
    async def _get_available_host(self) -> str:
        """Get an available host IP for decoy deployment"""
        # For now, return localhost. In production, this would implement
        # intelligent host selection based on network topology
        return "127.0.0.1"
    
    async def _get_available_port(self, host: str, start_port: int = 8000) -> int:
        """Find an available port on the given host"""
        for port in range(start_port, start_port + 1000):
            if await self._is_port_available(host, port):
                return port
        
        raise DecoyDeploymentError("No available ports found")
    
    async def _is_port_available(self, host: str, port: int) -> bool:
        """Check if a port is available on the given host"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            sock.close()
            return result != 0
        except Exception:
            return False
    
    async def _check_service_health(self, host: str, port: int) -> bool:
        """Check if a service is responsive"""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=5.0
            )
            writer.close()
            await writer.wait_closed()
            return True
        except Exception:
            return False
    
    async def _start_health_monitoring(self, decoy_id: str):
        """Start health monitoring for a decoy"""
        if decoy_id in self._health_check_tasks:
            return
        
        async def health_monitor():
            while decoy_id in self._running_decoys:
                try:
                    await self.health_check_decoy(decoy_id)
                    await asyncio.sleep(60)  # Check every minute
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Health monitoring error for decoy {decoy_id}: {e}")
                    await asyncio.sleep(60)
        
        task = asyncio.create_task(health_monitor())
        self._health_check_tasks[decoy_id] = task
    
    async def _stop_health_monitoring(self, decoy_id: str):
        """Stop health monitoring for a decoy"""
        if decoy_id in self._health_check_tasks:
            task = self._health_check_tasks[decoy_id]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            del self._health_check_tasks[decoy_id]
    
    def _calculate_uptime(self, decoy: Decoy) -> float:
        """Calculate decoy uptime in seconds"""
        if decoy.deployed_at and decoy.status == DecoyStatus.ACTIVE:
            return (datetime.utcnow() - decoy.deployed_at).total_seconds()
        return 0.0


# Global decoy manager instance
decoy_manager = DecoyManager() 