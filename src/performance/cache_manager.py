#!/usr/bin/env python3
"""
AADN Cache Manager
High-performance caching system for improved response times
"""

import json
import time
import hashlib
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import threading
from collections import OrderedDict

@dataclass
class CacheEntry:
    key: str
    value: Any
    created_at: datetime
    expires_at: datetime
    access_count: int = 0
    last_accessed: datetime = None

class MemoryCache:
    """High-performance in-memory cache with TTL and LRU eviction"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_requests": 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            self._stats["total_requests"] += 1
            
            if key not in self._cache:
                self._stats["misses"] += 1
                return None
            
            entry = self._cache[key]
            
            # Check if expired
            if datetime.now() > entry.expires_at:
                del self._cache[key]
                self._stats["misses"] += 1
                return None
            
            # Update access stats
            entry.access_count += 1
            entry.last_accessed = datetime.now()
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            
            self._stats["hits"] += 1
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache"""
        with self._lock:
            ttl = ttl or self.default_ttl
            now = datetime.now()
            expires_at = now + timedelta(seconds=ttl)
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                expires_at=expires_at,
                last_accessed=now
            )
            
            self._cache[key] = entry
            self._cache.move_to_end(key)
            
            # Evict if over max size
            while len(self._cache) > self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._stats["evictions"] += 1
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            hit_rate = 0
            if self._stats["total_requests"] > 0:
                hit_rate = self._stats["hits"] / self._stats["total_requests"]
            
            return {
                **self._stats,
                "hit_rate": hit_rate,
                "cache_size": len(self._cache),
                "max_size": self.max_size
            }
    
    def cleanup_expired(self) -> int:
        """Remove expired entries"""
        with self._lock:
            now = datetime.now()
            expired_keys = [
                key for key, entry in self._cache.items()
                if now > entry.expires_at
            ]
            
            for key in expired_keys:
                del self._cache[key]
            
            return len(expired_keys)

class CacheManager:
    """Main cache manager with multiple cache types"""
    
    def __init__(self):
        # Different caches for different data types
        self.api_cache = MemoryCache(max_size=500, default_ttl=60)  # API responses
        self.stats_cache = MemoryCache(max_size=100, default_ttl=30)  # Statistics
        self.threat_cache = MemoryCache(max_size=1000, default_ttl=300)  # Threat data
        self.user_cache = MemoryCache(max_size=200, default_ttl=900)  # User sessions
        
        # Start cleanup thread
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """Start background thread for cache cleanup"""
        def cleanup_worker():
            while True:
                time.sleep(60)  # Cleanup every minute
                self.cleanup_all()
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def cleanup_all(self):
        """Cleanup expired entries from all caches"""
        caches = [self.api_cache, self.stats_cache, self.threat_cache, self.user_cache]
        total_cleaned = sum(cache.cleanup_expired() for cache in caches)
        return total_cleaned
    
    def get_cache_key(self, prefix: str, *args) -> str:
        """Generate cache key from arguments"""
        key_data = f"{prefix}:{':'.join(str(arg) for arg in args)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    # API Cache methods
    def get_api_response(self, endpoint: str, params: Dict = None) -> Optional[Any]:
        """Get cached API response"""
        key = self.get_cache_key("api", endpoint, json.dumps(params or {}, sort_keys=True))
        return self.api_cache.get(key)
    
    def cache_api_response(self, endpoint: str, params: Dict, response: Any, ttl: int = 60):
        """Cache API response"""
        key = self.get_cache_key("api", endpoint, json.dumps(params or {}, sort_keys=True))
        self.api_cache.set(key, response, ttl)
    
    # Stats Cache methods
    def get_stats(self, stats_type: str) -> Optional[Any]:
        """Get cached statistics"""
        return self.stats_cache.get(f"stats:{stats_type}")
    
    def cache_stats(self, stats_type: str, data: Any, ttl: int = 30):
        """Cache statistics data"""
        self.stats_cache.set(f"stats:{stats_type}", data, ttl)
    
    # Threat Cache methods
    def get_threat_data(self, threat_id: str) -> Optional[Any]:
        """Get cached threat data"""
        return self.threat_cache.get(f"threat:{threat_id}")
    
    def cache_threat_data(self, threat_id: str, data: Any, ttl: int = 300):
        """Cache threat analysis data"""
        self.threat_cache.set(f"threat:{threat_id}", data, ttl)
    
    # User Cache methods
    def get_user_session(self, user_id: str) -> Optional[Any]:
        """Get cached user session data"""
        return self.user_cache.get(f"user:{user_id}")
    
    def cache_user_session(self, user_id: str, session_data: Any, ttl: int = 900):
        """Cache user session data"""
        self.user_cache.set(f"user:{user_id}", session_data, ttl)
    
    def invalidate_user_session(self, user_id: str):
        """Invalidate user session cache"""
        self.user_cache.delete(f"user:{user_id}")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        return {
            "api_cache": self.api_cache.get_stats(),
            "stats_cache": self.stats_cache.get_stats(),
            "threat_cache": self.threat_cache.get_stats(),
            "user_cache": self.user_cache.get_stats(),
            "total_memory_usage": self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> str:
        """Estimate total memory usage"""
        total_entries = (
            len(self.api_cache._cache) +
            len(self.stats_cache._cache) +
            len(self.threat_cache._cache) +
            len(self.user_cache._cache)
        )
        # Rough estimate: 1KB per entry
        estimated_mb = (total_entries * 1024) / (1024 * 1024)
        return f"{estimated_mb:.2f} MB"

# Global cache manager instance
cache_manager = CacheManager()

# Decorator for caching function results
def cached(cache_type: str = "api", ttl: int = 60):
    """Decorator to cache function results"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key
            key_data = f"{func.__name__}:{args}:{sorted(kwargs.items())}"
            cache_key = hashlib.md5(str(key_data).encode()).hexdigest()
            
            # Try to get from cache
            if cache_type == "api":
                cached_result = cache_manager.api_cache.get(cache_key)
            elif cache_type == "stats":
                cached_result = cache_manager.stats_cache.get(cache_key)
            elif cache_type == "threat":
                cached_result = cache_manager.threat_cache.get(cache_key)
            else:
                cached_result = cache_manager.user_cache.get(cache_key)
            
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            
            if cache_type == "api":
                cache_manager.api_cache.set(cache_key, result, ttl)
            elif cache_type == "stats":
                cache_manager.stats_cache.set(cache_key, result, ttl)
            elif cache_type == "threat":
                cache_manager.threat_cache.set(cache_key, result, ttl)
            else:
                cache_manager.user_cache.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator 