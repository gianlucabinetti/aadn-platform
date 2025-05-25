"""
AADN Decoy Models
Data models for decoy management
"""

from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class DecoyType(str, Enum):
    """Decoy service types"""
    SSH = "ssh"
    HTTP = "http"
    FTP = "ftp"
    TELNET = "telnet"
    SMB = "smb"
    MYSQL = "mysql"


class DecoyStatus(str, Enum):
    """Decoy status"""
    INACTIVE = "inactive"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    TERMINATING = "terminating"
    ERROR = "error"


class DecoyConfiguration(BaseModel):
    """Decoy configuration"""
    port: int
    interaction_level: str = "medium"
    custom_responses: Dict[str, Any] = Field(default_factory=dict)
    enable_logging: bool = True


class DecoyMetadata(BaseModel):
    """Decoy metadata"""
    description: Optional[str] = None
    tags: list = Field(default_factory=list)
    environment: str = "development"


class Decoy(BaseModel):
    """Decoy model"""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    type: DecoyType
    host: str
    configuration: DecoyConfiguration
    metadata: DecoyMetadata = Field(default_factory=DecoyMetadata)
    status: DecoyStatus = DecoyStatus.INACTIVE
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    deployed_at: Optional[datetime] = None
    health_status: Optional[str] = None
    last_health_check: Optional[datetime] = None
    last_error: Optional[str] = None


class DecoyDeploymentRequest(BaseModel):
    """Request to deploy a new decoy"""
    name: str
    type: DecoyType
    host: Optional[str] = None
    configuration: Optional[DecoyConfiguration] = None
    metadata: Optional[DecoyMetadata] = None
    auto_start: bool = True


class DecoyUpdateRequest(BaseModel):
    """Request to update a decoy"""
    name: Optional[str] = None
    configuration: Optional[DecoyConfiguration] = None
    metadata: Optional[DecoyMetadata] = None
    status: Optional[DecoyStatus] = None


class DecoyListResponse(BaseModel):
    """Response for listing decoys"""
    decoys: list[Decoy]
    total: int
    page: int
    page_size: int
    has_next: bool


class DecoyHealthCheck(BaseModel):
    """Decoy health check result"""
    decoy_id: str
    status: str
    response_time: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = Field(default_factory=dict)
    errors: list = Field(default_factory=list)


# Default configurations for different decoy types
DEFAULT_CONFIGURATIONS = {
    DecoyType.SSH: DecoyConfiguration(port=2222),
    DecoyType.HTTP: DecoyConfiguration(port=8080),
    DecoyType.FTP: DecoyConfiguration(port=2121),
    DecoyType.TELNET: DecoyConfiguration(port=2323),
    DecoyType.SMB: DecoyConfiguration(port=445),
    DecoyType.MYSQL: DecoyConfiguration(port=3306),
} 