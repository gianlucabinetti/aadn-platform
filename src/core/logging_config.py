"""
AADN Logging Configuration
Centralized logging setup for the AADN system
"""

import logging
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any

from rich.console import Console
from rich.logging import RichHandler

from .config import get_settings


class SecurityEvent(str, Enum):
    """Security event types for logging"""
    DECOY_INTERACTION = "decoy_interaction"
    THREAT_DETECTED = "threat_detected"
    ALERT_SENT = "alert_sent"
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    CONFIG_CHANGE = "config_change"
    AUTH_FAILURE = "auth_failure"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"


# Global console for rich output
console = Console()

# Logger instances
_loggers: Dict[str, logging.Logger] = {}


def setup_logging():
    """Setup logging configuration for AADN"""
    settings = get_settings()
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler with Rich
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=True,
        markup=True,
        rich_tracebacks=True
    )
    console_handler.setLevel(getattr(logging, settings.log_level.upper()))
    
    # Console formatter
    console_formatter = logging.Formatter(
        fmt="%(message)s",
        datefmt="[%X]"
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if log file is specified
    if settings.log_file:
        file_handler = logging.FileHandler(settings.log_file)
        file_handler.setLevel(logging.DEBUG)
        
        file_formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Security events file handler
    security_handler = logging.FileHandler(log_dir / "security.log")
    security_handler.setLevel(logging.INFO)
    
    security_formatter = logging.Formatter(
        fmt="%(asctime)s - SECURITY - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    security_handler.setFormatter(security_formatter)
    
    # Create security logger
    security_logger = logging.getLogger("aadn.security")
    security_logger.addHandler(security_handler)
    security_logger.setLevel(logging.INFO)
    security_logger.propagate = False
    
    # Create component-specific loggers
    _create_component_loggers()


def _create_component_loggers():
    """Create loggers for different AADN components"""
    components = [
        "aadn.api",
        "aadn.decoys",
        "aadn.ai",
        "aadn.intelligence",
        "aadn.monitoring",
        "aadn.core"
    ]
    
    for component in components:
        logger = logging.getLogger(component)
        _loggers[component] = logger


def get_logger(name: str = "aadn") -> logging.Logger:
    """Get a logger instance"""
    if name in _loggers:
        return _loggers[name]
    
    logger = logging.getLogger(name)
    _loggers[name] = logger
    return logger


def get_api_logger() -> logging.Logger:
    """Get API logger"""
    return get_logger("aadn.api")


def get_decoy_logger() -> logging.Logger:
    """Get decoy logger"""
    return get_logger("aadn.decoys")


def get_ai_logger() -> logging.Logger:
    """Get AI logger"""
    return get_logger("aadn.ai")


def get_intelligence_logger() -> logging.Logger:
    """Get intelligence logger"""
    return get_logger("aadn.intelligence")


def get_monitoring_logger() -> logging.Logger:
    """Get monitoring logger"""
    return get_logger("aadn.monitoring")


def log_security_event(
    event_type: SecurityEvent,
    data: Dict[str, Any],
    severity: str = "INFO",
    source_ip: Optional[str] = None,
    user_id: Optional[str] = None
):
    """Log a security event"""
    security_logger = logging.getLogger("aadn.security")
    
    event_data = {
        "event_type": event_type,
        "timestamp": datetime.utcnow().isoformat(),
        "severity": severity,
        "data": data
    }
    
    if source_ip:
        event_data["source_ip"] = source_ip
    
    if user_id:
        event_data["user_id"] = user_id
    
    # Format message for logging
    message = f"{event_type} | {severity}"
    if source_ip:
        message += f" | IP: {source_ip}"
    
    # Add key data points to message
    if "decoy_id" in data:
        message += f" | Decoy: {data['decoy_id']}"
    if "threat_level" in data:
        message += f" | Threat: {data['threat_level']}"
    if "alert_id" in data:
        message += f" | Alert: {data['alert_id']}"
    
    # Log the event
    log_level = getattr(logging, severity.upper(), logging.INFO)
    security_logger.log(log_level, message, extra={"event_data": event_data})


def log_interaction(
    decoy_id: str,
    source_ip: str,
    interaction_type: str,
    data: Dict[str, Any]
):
    """Log a decoy interaction"""
    log_security_event(
        SecurityEvent.DECOY_INTERACTION,
        {
            "decoy_id": decoy_id,
            "interaction_type": interaction_type,
            "interaction_data": data
        },
        severity="INFO",
        source_ip=source_ip
    )


def log_threat_detection(
    threat_id: str,
    threat_level: str,
    source_ip: str,
    details: Dict[str, Any]
):
    """Log a threat detection"""
    log_security_event(
        SecurityEvent.THREAT_DETECTED,
        {
            "threat_id": threat_id,
            "threat_level": threat_level,
            "details": details
        },
        severity="WARNING" if threat_level in ["medium", "high"] else "CRITICAL",
        source_ip=source_ip
    )


def log_system_event(event_type: SecurityEvent, message: str, **kwargs):
    """Log a system event"""
    log_security_event(
        event_type,
        {"message": message, **kwargs},
        severity="INFO"
    ) 