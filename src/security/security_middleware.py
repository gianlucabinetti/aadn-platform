#!/usr/bin/env python3
"""
AADN Security Middleware
Comprehensive security layer protecting against common vulnerabilities
"""

import re
import time
import hashlib
import secrets
import logging
from typing import Dict, List, Optional, Set, Any
from datetime import datetime, timedelta
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
import ipaddress
from urllib.parse import unquote
import html

logger = logging.getLogger(__name__)

class SecurityConfig:
    """Security configuration settings"""
    
    # Rate limiting
    RATE_LIMIT_REQUESTS = 100
    RATE_LIMIT_WINDOW = 60  # seconds
    RATE_LIMIT_BURST = 20
    
    # CSRF protection
    CSRF_TOKEN_LENGTH = 32
    CSRF_TOKEN_EXPIRY = 3600  # seconds
    
    # Content Security Policy
    CSP_POLICY = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self' https:; "
        "connect-src 'self'; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self'"
    )
    
    # Security headers
    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        "Content-Security-Policy": CSP_POLICY
    }
    
    # Input validation
    MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_HEADER_SIZE = 8192  # 8KB
    MAX_URL_LENGTH = 2048
    
    # Blocked patterns (potential attacks)
    BLOCKED_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # XSS
        r'javascript:',  # XSS
        r'on\w+\s*=',  # Event handlers
        r'union\s+select',  # SQL injection
        r'drop\s+table',  # SQL injection
        r'insert\s+into',  # SQL injection
        r'delete\s+from',  # SQL injection
        r'update\s+.*\s+set',  # SQL injection
        r'exec\s*\(',  # Command injection
        r'system\s*\(',  # Command injection
        r'eval\s*\(',  # Code injection
        r'\.\./',  # Path traversal
        r'\.\.\\',  # Path traversal
        r'/etc/passwd',  # File access
        r'/proc/',  # System access
        r'cmd\.exe',  # Windows command execution
        r'powershell',  # PowerShell execution
    ]
    
    # Trusted IP ranges (for internal services)
    TRUSTED_IPS = [
        "127.0.0.0/8",  # Localhost
        "10.0.0.0/8",   # Private network
        "172.16.0.0/12", # Private network
        "192.168.0.0/16" # Private network
    ]

class RateLimiter:
    """Advanced rate limiting with multiple strategies"""
    
    def __init__(self):
        self.requests: Dict[str, List[float]] = {}
        self.blocked_ips: Dict[str, float] = {}
        self.suspicious_ips: Set[str] = set()
    
    def is_allowed(self, client_ip: str, endpoint: str = "") -> bool:
        """Check if request is allowed based on rate limiting"""
        now = time.time()
        key = f"{client_ip}:{endpoint}"
        
        # Check if IP is temporarily blocked
        if client_ip in self.blocked_ips:
            if now < self.blocked_ips[client_ip]:
                return False
            else:
                del self.blocked_ips[client_ip]
        
        # Initialize request history for this key
        if key not in self.requests:
            self.requests[key] = []
        
        # Clean old requests
        self.requests[key] = [
            req_time for req_time in self.requests[key]
            if now - req_time < SecurityConfig.RATE_LIMIT_WINDOW
        ]
        
        # Check rate limit
        if len(self.requests[key]) >= SecurityConfig.RATE_LIMIT_REQUESTS:
            # Block IP for 5 minutes
            self.blocked_ips[client_ip] = now + 300
            self.suspicious_ips.add(client_ip)
            logger.warning(f"Rate limit exceeded for IP {client_ip}, blocking for 5 minutes")
            return False
        
        # Add current request
        self.requests[key].append(now)
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics"""
        now = time.time()
        active_blocks = sum(1 for block_time in self.blocked_ips.values() if now < block_time)
        
        return {
            "active_blocks": active_blocks,
            "suspicious_ips": len(self.suspicious_ips),
            "total_tracked_keys": len(self.requests)
        }

class CSRFProtection:
    """CSRF token generation and validation"""
    
    def __init__(self):
        self.tokens: Dict[str, Dict[str, Any]] = {}
    
    def generate_token(self, session_id: str) -> str:
        """Generate CSRF token for session"""
        token = secrets.token_urlsafe(SecurityConfig.CSRF_TOKEN_LENGTH)
        expires_at = time.time() + SecurityConfig.CSRF_TOKEN_EXPIRY
        
        self.tokens[token] = {
            "session_id": session_id,
            "expires_at": expires_at,
            "created_at": time.time()
        }
        
        return token
    
    def validate_token(self, token: str, session_id: str) -> bool:
        """Validate CSRF token"""
        if not token or token not in self.tokens:
            return False
        
        token_data = self.tokens[token]
        now = time.time()
        
        # Check expiry
        if now > token_data["expires_at"]:
            del self.tokens[token]
            return False
        
        # Check session match
        if token_data["session_id"] != session_id:
            return False
        
        return True
    
    def cleanup_expired(self):
        """Remove expired tokens"""
        now = time.time()
        expired_tokens = [
            token for token, data in self.tokens.items()
            if now > data["expires_at"]
        ]
        
        for token in expired_tokens:
            del self.tokens[token]

class InputValidator:
    """Input validation and sanitization"""
    
    @staticmethod
    def sanitize_input(data: str) -> str:
        """Sanitize input data"""
        if not isinstance(data, str):
            return str(data)
        
        # HTML escape
        data = html.escape(data)
        
        # Remove null bytes
        data = data.replace('\x00', '')
        
        # Normalize unicode
        data = data.encode('utf-8', 'ignore').decode('utf-8')
        
        return data
    
    @staticmethod
    def validate_input(data: str, max_length: int = 1000) -> bool:
        """Validate input against malicious patterns"""
        if not data:
            return True
        
        # Check length
        if len(data) > max_length:
            return False
        
        # Check for malicious patterns
        data_lower = data.lower()
        for pattern in SecurityConfig.BLOCKED_PATTERNS:
            if re.search(pattern, data_lower, re.IGNORECASE):
                logger.warning(f"Blocked malicious pattern: {pattern} in input: {data[:100]}")
                return False
        
        return True
    
    @staticmethod
    def validate_json_input(data: Dict[str, Any]) -> bool:
        """Validate JSON input recursively"""
        def _validate_value(value):
            if isinstance(value, str):
                return InputValidator.validate_input(value)
            elif isinstance(value, dict):
                return all(_validate_value(v) for v in value.values())
            elif isinstance(value, list):
                return all(_validate_value(item) for item in value)
            return True
        
        return _validate_value(data)

class IPWhitelist:
    """IP address whitelisting and blacklisting"""
    
    def __init__(self):
        self.blacklisted_ips: Set[str] = set()
        self.whitelisted_networks = [
            ipaddress.ip_network(net) for net in SecurityConfig.TRUSTED_IPS
        ]
    
    def is_trusted_ip(self, ip_str: str) -> bool:
        """Check if IP is in trusted networks"""
        try:
            ip = ipaddress.ip_address(ip_str)
            return any(ip in network for network in self.whitelisted_networks)
        except ValueError:
            return False
    
    def is_blacklisted(self, ip_str: str) -> bool:
        """Check if IP is blacklisted"""
        return ip_str in self.blacklisted_ips
    
    def blacklist_ip(self, ip_str: str):
        """Add IP to blacklist"""
        self.blacklisted_ips.add(ip_str)
        logger.warning(f"IP {ip_str} added to blacklist")

class SecurityMiddleware:
    """Main security middleware class"""
    
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.csrf_protection = CSRFProtection()
        self.ip_whitelist = IPWhitelist()
        self.request_log: List[Dict[str, Any]] = []
        
    async def __call__(self, request: Request, call_next):
        """Main middleware function"""
        start_time = time.time()
        client_ip = self._get_client_ip(request)
        
        try:
            # Security checks
            security_check = await self._perform_security_checks(request, client_ip)
            if security_check:
                return security_check
            
            # Process request
            response = await call_next(request)
            
            # Add security headers
            self._add_security_headers(response)
            
            # Log request
            self._log_request(request, client_ip, response.status_code, time.time() - start_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Security middleware error: {e}")
            self._log_request(request, client_ip, 500, time.time() - start_time, str(e))
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error"}
            )
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address"""
        # Check for forwarded headers (reverse proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct connection
        if request.client:
            return request.client.host
        
        return "unknown"
    
    async def _perform_security_checks(self, request: Request, client_ip: str) -> Optional[Response]:
        """Perform comprehensive security checks"""
        
        # 1. IP blacklist check
        if self.ip_whitelist.is_blacklisted(client_ip):
            logger.warning(f"Blocked blacklisted IP: {client_ip}")
            return JSONResponse(
                status_code=403,
                content={"error": "Access denied"}
            )
        
        # 2. Rate limiting
        endpoint = str(request.url.path)
        if not self.rate_limiter.is_allowed(client_ip, endpoint):
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded"}
            )
        
        # 3. Request size validation
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > SecurityConfig.MAX_REQUEST_SIZE:
            logger.warning(f"Request too large from {client_ip}: {content_length} bytes")
            return JSONResponse(
                status_code=413,
                content={"error": "Request too large"}
            )
        
        # 4. URL validation
        if len(str(request.url)) > SecurityConfig.MAX_URL_LENGTH:
            logger.warning(f"URL too long from {client_ip}")
            return JSONResponse(
                status_code=414,
                content={"error": "URL too long"}
            )
        
        # 5. Header validation
        for header_name, header_value in request.headers.items():
            if len(header_value) > SecurityConfig.MAX_HEADER_SIZE:
                logger.warning(f"Header too large from {client_ip}: {header_name}")
                return JSONResponse(
                    status_code=400,
                    content={"error": "Header too large"}
                )
        
        # 6. URL pattern validation
        url_path = unquote(str(request.url.path))
        if not InputValidator.validate_input(url_path, max_length=SecurityConfig.MAX_URL_LENGTH):
            logger.warning(f"Malicious URL pattern from {client_ip}: {url_path}")
            self.ip_whitelist.blacklist_ip(client_ip)
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid request"}
            )
        
        # 7. Query parameter validation
        for param_name, param_value in request.query_params.items():
            if not InputValidator.validate_input(param_value):
                logger.warning(f"Malicious query parameter from {client_ip}: {param_name}={param_value}")
                return JSONResponse(
                    status_code=400,
                    content={"error": "Invalid query parameter"}
                )
        
        # 8. JSON body validation (for POST/PUT requests)
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                if "application/json" in request.headers.get("content-type", ""):
                    body = await request.body()
                    if body:
                        import json
                        json_data = json.loads(body)
                        if not InputValidator.validate_json_input(json_data):
                            logger.warning(f"Malicious JSON input from {client_ip}")
                            return JSONResponse(
                                status_code=400,
                                content={"error": "Invalid input data"}
                            )
            except Exception as e:
                logger.warning(f"JSON validation error from {client_ip}: {e}")
                return JSONResponse(
                    status_code=400,
                    content={"error": "Invalid JSON"}
                )
        
        return None
    
    def _add_security_headers(self, response: Response):
        """Add security headers to response"""
        for header_name, header_value in SecurityConfig.SECURITY_HEADERS.items():
            response.headers[header_name] = header_value
    
    def _log_request(self, request: Request, client_ip: str, status_code: int, 
                    response_time: float, error: str = None):
        """Log request for security monitoring"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "client_ip": client_ip,
            "method": request.method,
            "path": str(request.url.path),
            "status_code": status_code,
            "response_time": response_time,
            "user_agent": request.headers.get("User-Agent", ""),
            "referer": request.headers.get("Referer", ""),
            "error": error
        }
        
        self.request_log.append(log_entry)
        
        # Keep only last 1000 entries
        if len(self.request_log) > 1000:
            self.request_log = self.request_log[-1000:]
        
        # Log suspicious activity
        if status_code in [400, 403, 429] or error:
            logger.warning(f"Suspicious request: {log_entry}")
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics"""
        now = time.time()
        last_hour = now - 3600
        
        recent_requests = [
            req for req in self.request_log
            if datetime.fromisoformat(req["timestamp"]).timestamp() > last_hour
        ]
        
        blocked_requests = [
            req for req in recent_requests
            if req["status_code"] in [400, 403, 429]
        ]
        
        return {
            "total_requests_last_hour": len(recent_requests),
            "blocked_requests_last_hour": len(blocked_requests),
            "rate_limiter_stats": self.rate_limiter.get_stats(),
            "blacklisted_ips": len(self.ip_whitelist.blacklisted_ips),
            "csrf_tokens_active": len(self.csrf_protection.tokens)
        }
    
    def cleanup(self):
        """Cleanup expired data"""
        self.csrf_protection.cleanup_expired()

# Global security middleware instance
security_middleware = SecurityMiddleware() 