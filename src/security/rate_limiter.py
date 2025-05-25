#!/usr/bin/env python3
"""
Advanced Rate Limiting and DDoS Protection
Enterprise-grade API protection system
"""

import time
import asyncio
from typing import Dict, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import hashlib
import ipaddress
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

class ThreatLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RateLimitRule:
    requests_per_minute: int
    requests_per_hour: int
    burst_limit: int
    block_duration: int  # seconds
    threat_level: ThreatLevel

class AdvancedRateLimiter:
    def __init__(self):
        # Rate limiting storage
        self.request_counts = defaultdict(lambda: {"minute": deque(), "hour": deque()})
        self.blocked_ips = {}  # ip -> unblock_time
        self.suspicious_ips = defaultdict(int)
        self.whitelist = set()
        self.blacklist = set()
        
        # Rate limit rules by endpoint type
        self.rules = {
            "auth": RateLimitRule(5, 20, 10, 300, ThreatLevel.HIGH),
            "api": RateLimitRule(60, 1000, 100, 60, ThreatLevel.MEDIUM),
            "health": RateLimitRule(120, 2000, 200, 30, ThreatLevel.LOW),
            "default": RateLimitRule(30, 500, 50, 120, ThreatLevel.MEDIUM)
        }
        
        # DDoS detection patterns
        self.ddos_patterns = {
            "rapid_requests": {"threshold": 100, "window": 60},
            "distributed_attack": {"threshold": 50, "window": 300},
            "resource_exhaustion": {"threshold": 1000, "window": 3600}
        }
        
        # Geolocation-based rules (simplified)
        self.geo_rules = {
            "high_risk_countries": ["CN", "RU", "KP", "IR"],
            "allowed_countries": ["US", "CA", "GB", "DE", "FR", "AU", "JP"]
        }
    
    def get_client_identifier(self, request: Request) -> str:
        """Get unique client identifier with multiple fallbacks"""
        # Try to get real IP from headers (for load balancers/proxies)
        real_ip = (
            request.headers.get("X-Forwarded-For", "").split(",")[0].strip() or
            request.headers.get("X-Real-IP", "") or
            request.headers.get("CF-Connecting-IP", "") or  # Cloudflare
            request.client.host if request.client else "unknown"
        )
        
        # Create composite identifier
        user_agent = request.headers.get("User-Agent", "")
        user_agent_hash = hashlib.md5(user_agent.encode()).hexdigest()[:8]
        
        return f"{real_ip}:{user_agent_hash}"
    
    def get_endpoint_type(self, path: str) -> str:
        """Determine endpoint type for rate limiting rules"""
        if "/auth/" in path:
            return "auth"
        elif "/health" in path:
            return "health"
        elif "/api/" in path:
            return "api"
        else:
            return "default"
    
    def is_ip_whitelisted(self, ip: str) -> bool:
        """Check if IP is in whitelist"""
        try:
            ip_obj = ipaddress.ip_address(ip)
            # Add common safe ranges
            safe_ranges = [
                ipaddress.ip_network("127.0.0.0/8"),  # Localhost
                ipaddress.ip_network("10.0.0.0/8"),   # Private
                ipaddress.ip_network("172.16.0.0/12"), # Private
                ipaddress.ip_network("192.168.0.0/16") # Private
            ]
            
            for safe_range in safe_ranges:
                if ip_obj in safe_range:
                    return True
                    
            return ip in self.whitelist
        except:
            return False
    
    def is_ip_blacklisted(self, ip: str) -> bool:
        """Check if IP is in blacklist"""
        return ip in self.blacklist
    
    def detect_ddos_patterns(self, client_id: str, current_time: float) -> Optional[str]:
        """Detect DDoS attack patterns"""
        ip = client_id.split(":")[0]
        
        # Check for rapid requests from single IP
        minute_requests = len([t for t in self.request_counts[client_id]["minute"] 
                              if current_time - t < 60])
        
        if minute_requests > self.ddos_patterns["rapid_requests"]["threshold"]:
            return "rapid_requests"
        
        # Check for distributed attack (many IPs with moderate traffic)
        recent_ips = set()
        for cid in self.request_counts:
            if any(current_time - t < 300 for t in self.request_counts[cid]["minute"]):
                recent_ips.add(cid.split(":")[0])
        
        if len(recent_ips) > self.ddos_patterns["distributed_attack"]["threshold"]:
            return "distributed_attack"
        
        return None
    
    def calculate_threat_score(self, client_id: str, request: Request) -> int:
        """Calculate threat score based on multiple factors"""
        score = 0
        ip = client_id.split(":")[0]
        
        # IP reputation
        if self.is_ip_blacklisted(ip):
            score += 100
        
        # Suspicious activity history
        score += self.suspicious_ips[ip] * 10
        
        # User agent analysis
        user_agent = request.headers.get("User-Agent", "").lower()
        suspicious_agents = ["bot", "crawler", "scanner", "curl", "wget", "python"]
        if any(agent in user_agent for agent in suspicious_agents):
            score += 20
        
        # Missing common headers
        if not request.headers.get("Accept"):
            score += 10
        if not request.headers.get("Accept-Language"):
            score += 10
        
        # Request patterns
        if request.method in ["PUT", "DELETE", "PATCH"]:
            score += 5
        
        return min(score, 100)  # Cap at 100
    
    async def check_rate_limit(self, request: Request) -> Tuple[bool, Optional[dict]]:
        """
        Check if request should be rate limited
        Returns: (is_allowed, error_response)
        """
        current_time = time.time()
        client_id = self.get_client_identifier(request)
        ip = client_id.split(":")[0]
        endpoint_type = self.get_endpoint_type(request.url.path)
        rule = self.rules[endpoint_type]
        
        # Check if IP is blocked
        if ip in self.blocked_ips:
            if current_time < self.blocked_ips[ip]:
                return False, {
                    "error": "IP_BLOCKED",
                    "message": "Your IP has been temporarily blocked due to suspicious activity",
                    "unblock_time": self.blocked_ips[ip],
                    "threat_level": rule.threat_level.value
                }
            else:
                # Unblock expired IPs
                del self.blocked_ips[ip]
        
        # Check whitelist/blacklist
        if self.is_ip_whitelisted(ip):
            return True, None
        
        if self.is_ip_blacklisted(ip):
            return False, {
                "error": "IP_BLACKLISTED",
                "message": "Your IP has been blacklisted",
                "threat_level": ThreatLevel.CRITICAL.value
            }
        
        # Clean old entries
        self._cleanup_old_entries(client_id, current_time)
        
        # Add current request
        self.request_counts[client_id]["minute"].append(current_time)
        self.request_counts[client_id]["hour"].append(current_time)
        
        # Count recent requests
        minute_count = len(self.request_counts[client_id]["minute"])
        hour_count = len(self.request_counts[client_id]["hour"])
        
        # Check rate limits
        if minute_count > rule.requests_per_minute:
            self._handle_rate_limit_violation(client_id, "minute", rule)
            return False, {
                "error": "RATE_LIMIT_EXCEEDED",
                "message": f"Too many requests per minute. Limit: {rule.requests_per_minute}",
                "retry_after": 60,
                "threat_level": rule.threat_level.value
            }
        
        if hour_count > rule.requests_per_hour:
            self._handle_rate_limit_violation(client_id, "hour", rule)
            return False, {
                "error": "RATE_LIMIT_EXCEEDED",
                "message": f"Too many requests per hour. Limit: {rule.requests_per_hour}",
                "retry_after": 3600,
                "threat_level": rule.threat_level.value
            }
        
        # Check for burst traffic
        recent_requests = [t for t in self.request_counts[client_id]["minute"] 
                          if current_time - t < 10]  # Last 10 seconds
        
        if len(recent_requests) > rule.burst_limit:
            self._handle_rate_limit_violation(client_id, "burst", rule)
            return False, {
                "error": "BURST_LIMIT_EXCEEDED",
                "message": f"Too many requests in short time. Burst limit: {rule.burst_limit}",
                "retry_after": 10,
                "threat_level": rule.threat_level.value
            }
        
        # DDoS detection
        ddos_pattern = self.detect_ddos_patterns(client_id, current_time)
        if ddos_pattern:
            self._handle_ddos_detection(client_id, ddos_pattern)
            return False, {
                "error": "DDOS_DETECTED",
                "message": "DDoS attack pattern detected",
                "pattern": ddos_pattern,
                "threat_level": ThreatLevel.CRITICAL.value
            }
        
        # Threat scoring
        threat_score = self.calculate_threat_score(client_id, request)
        if threat_score > 80:
            self.suspicious_ips[ip] += 1
            return False, {
                "error": "HIGH_THREAT_SCORE",
                "message": "Request blocked due to high threat score",
                "threat_score": threat_score,
                "threat_level": ThreatLevel.HIGH.value
            }
        
        return True, None
    
    def _cleanup_old_entries(self, client_id: str, current_time: float):
        """Remove old entries from rate limiting counters"""
        # Clean minute entries (older than 60 seconds)
        while (self.request_counts[client_id]["minute"] and 
               current_time - self.request_counts[client_id]["minute"][0] > 60):
            self.request_counts[client_id]["minute"].popleft()
        
        # Clean hour entries (older than 3600 seconds)
        while (self.request_counts[client_id]["hour"] and 
               current_time - self.request_counts[client_id]["hour"][0] > 3600):
            self.request_counts[client_id]["hour"].popleft()
    
    def _handle_rate_limit_violation(self, client_id: str, limit_type: str, rule: RateLimitRule):
        """Handle rate limit violations"""
        ip = client_id.split(":")[0]
        current_time = time.time()
        
        # Increase suspicion score
        self.suspicious_ips[ip] += 1
        
        # Block IP if too many violations
        if self.suspicious_ips[ip] >= 5:
            self.blocked_ips[ip] = current_time + rule.block_duration
            
        # Log the violation (in production, send to SIEM)
        print(f"Rate limit violation: {ip} - {limit_type} - Score: {self.suspicious_ips[ip]}")
    
    def _handle_ddos_detection(self, client_id: str, pattern: str):
        """Handle DDoS attack detection"""
        ip = client_id.split(":")[0]
        current_time = time.time()
        
        # Immediate block for DDoS
        self.blocked_ips[ip] = current_time + 3600  # 1 hour block
        self.suspicious_ips[ip] += 10
        
        # Log critical security event
        print(f"DDoS ATTACK DETECTED: {ip} - Pattern: {pattern}")
    
    def add_to_whitelist(self, ip: str):
        """Add IP to whitelist"""
        self.whitelist.add(ip)
    
    def add_to_blacklist(self, ip: str):
        """Add IP to blacklist"""
        self.blacklist.add(ip)
    
    def get_stats(self) -> dict:
        """Get rate limiting statistics"""
        current_time = time.time()
        active_ips = len([ip for ip, unblock_time in self.blocked_ips.items() 
                         if current_time < unblock_time])
        
        return {
            "active_clients": len(self.request_counts),
            "blocked_ips": active_ips,
            "suspicious_ips": len(self.suspicious_ips),
            "whitelist_size": len(self.whitelist),
            "blacklist_size": len(self.blacklist),
            "total_violations": sum(self.suspicious_ips.values())
        }

# Global rate limiter instance
rate_limiter = AdvancedRateLimiter()

async def rate_limit_middleware(request: Request, call_next):
    """FastAPI middleware for rate limiting"""
    is_allowed, error_response = await rate_limiter.check_rate_limit(request)
    
    if not is_allowed:
        return JSONResponse(
            status_code=429,
            content=error_response,
            headers={"Retry-After": str(error_response.get("retry_after", 60))}
        )
    
    response = await call_next(request)
    
    # Add rate limit headers
    client_id = rate_limiter.get_client_identifier(request)
    endpoint_type = rate_limiter.get_endpoint_type(request.url.path)
    rule = rate_limiter.rules[endpoint_type]
    
    current_time = time.time()
    rate_limiter._cleanup_old_entries(client_id, current_time)
    
    minute_count = len(rate_limiter.request_counts[client_id]["minute"])
    hour_count = len(rate_limiter.request_counts[client_id]["hour"])
    
    response.headers["X-RateLimit-Limit-Minute"] = str(rule.requests_per_minute)
    response.headers["X-RateLimit-Remaining-Minute"] = str(max(0, rule.requests_per_minute - minute_count))
    response.headers["X-RateLimit-Limit-Hour"] = str(rule.requests_per_hour)
    response.headers["X-RateLimit-Remaining-Hour"] = str(max(0, rule.requests_per_hour - hour_count))
    
    return response 