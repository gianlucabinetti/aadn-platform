"""
AADN Exception Classes
Custom exceptions for the AADN system
"""

from typing import Any, Dict, Optional


class AADNException(Exception):
    """Base exception for AADN"""
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        self.detail = message
        super().__init__(self.message)


class DecoyError(AADNException):
    """Base decoy exception"""
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message, status_code)


class DecoyNotFoundError(DecoyError):
    """Decoy not found exception"""
    def __init__(self, decoy_id: str):
        super().__init__(f"Decoy not found: {decoy_id}", 404)


class DecoyDeploymentError(DecoyError):
    """Decoy deployment exception"""
    def __init__(self, message: str):
        super().__init__(f"Decoy deployment failed: {message}", 500)


class MonitoringError(AADNException):
    """Monitoring system exception"""
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message, status_code)


class AIError(AADNException):
    """AI system exception"""
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message, status_code)


class ConfigurationError(AADNException):
    """Configuration exception"""
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message, status_code)


class DatabaseError(AADNException):
    """Raised when there's a database error"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=500, details=details)


class AIModelError(AADNException):
    """Raised when there's an error with AI/ML models"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=500, details=details)


class ModelNotFoundError(AIModelError):
    """Raised when an AI model is not found"""
    
    def __init__(self, model_name: str, details: Optional[Dict[str, Any]] = None):
        message = f"AI model '{model_name}' not found"
        super().__init__(message, details=details)


class ModelTrainingError(AIModelError):
    """Raised when model training fails"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details=details)


class InteractionError(AADNException):
    """Raised when there's an error processing interactions"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=500, details=details)


class ThreatIntelligenceError(AADNException):
    """Raised when there's an error with threat intelligence operations"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=500, details=details)


class AuthenticationError(AADNException):
    """Raised when authentication fails"""
    
    def __init__(self, message: str = "Authentication failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=401, details=details)


class AuthorizationError(AADNException):
    """Raised when authorization fails"""
    
    def __init__(self, message: str = "Access denied", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=403, details=details)


class ValidationError(AADNException):
    """Raised when data validation fails"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=422, details=details)


class NetworkError(AADNException):
    """Raised when there's a network-related error"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=500, details=details)


class IntegrationError(AADNException):
    """Raised when there's an error with external integrations"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=500, details=details)


class RateLimitError(AADNException):
    """Raised when rate limits are exceeded"""
    
    def __init__(self, message: str = "Rate limit exceeded", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=429, details=details)


class ResourceNotFoundError(AADNException):
    """Raised when a requested resource is not found"""
    
    def __init__(self, resource_type: str, resource_id: str, details: Optional[Dict[str, Any]] = None):
        message = f"{resource_type} with ID '{resource_id}' not found"
        super().__init__(message, status_code=404, details=details)


class ResourceConflictError(AADNException):
    """Raised when there's a conflict with a resource"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=409, details=details)


class ServiceUnavailableError(AADNException):
    """Raised when a service is temporarily unavailable"""
    
    def __init__(self, service_name: str, details: Optional[Dict[str, Any]] = None):
        message = f"Service '{service_name}' is temporarily unavailable"
        super().__init__(message, status_code=503, details=details) 