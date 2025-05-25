"""
AADN Configuration Management
Centralized configuration for the AADN system
"""

import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field


class AADNSettings(BaseSettings):
    """AADN system configuration"""
    
    # API Settings
    api_host: str = Field(default="0.0.0.0", env="AADN_API_HOST")
    api_port: int = Field(default=8000, env="AADN_API_PORT")
    debug: bool = Field(default=False, env="AADN_DEBUG")
    
    # Database Settings
    mongodb_url: str = Field(default="mongodb://localhost:27017", env="MONGODB_URL")
    mongodb_database: str = Field(default="aadn", env="MONGODB_DATABASE")
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # Security Settings
    secret_key: str = Field(default="your-secret-key-change-this", env="AADN_SECRET_KEY")
    allowed_hosts: List[str] = Field(default=["*"], env="AADN_ALLOWED_HOSTS")
    
    # Decoy Settings
    default_decoy_host: str = Field(default="127.0.0.1", env="AADN_DECOY_HOST")
    decoy_timeout: int = Field(default=30, env="AADN_DECOY_TIMEOUT")
    max_connections_per_decoy: int = Field(default=100, env="AADN_MAX_CONNECTIONS")
    
    # AI Settings
    enable_ai_analysis: bool = Field(default=True, env="AADN_ENABLE_AI")
    threat_analysis_interval: int = Field(default=300, env="AADN_ANALYSIS_INTERVAL")
    
    # Alert Settings
    smtp_host: Optional[str] = Field(default=None, env="SMTP_HOST")
    smtp_port: int = Field(default=587, env="SMTP_PORT")
    smtp_user: Optional[str] = Field(default=None, env="SMTP_USER")
    smtp_password: Optional[str] = Field(default=None, env="SMTP_PASSWORD")
    smtp_tls: bool = Field(default=True, env="SMTP_TLS")
    
    slack_webhook_url: Optional[str] = Field(default=None, env="SLACK_WEBHOOK_URL")
    
    # Logging Settings
    log_level: str = Field(default="INFO", env="AADN_LOG_LEVEL")
    log_file: Optional[str] = Field(default=None, env="AADN_LOG_FILE")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
_settings: Optional[AADNSettings] = None


def get_settings() -> AADNSettings:
    """Get the global settings instance"""
    global _settings
    if _settings is None:
        _settings = AADNSettings()
    return _settings


def reload_settings():
    """Reload settings from environment"""
    global _settings
    _settings = AADNSettings()
    return _settings 