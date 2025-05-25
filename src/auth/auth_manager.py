#!/usr/bin/env python3
"""
AADN Authentication Manager
Complete user authentication and authorization system
"""

import hashlib
import secrets
import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from dataclasses import dataclass
from enum import Enum
import json
import os

class UserRole(Enum):
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"

@dataclass
class User:
    id: str
    username: str
    email: str
    password_hash: str
    role: UserRole
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None

@dataclass
class Session:
    token: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    ip_address: str
    user_agent: str

class AuthManager:
    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Session] = {}
        self.max_failed_attempts = 5
        self.lockout_duration = timedelta(minutes=30)
        self.session_duration = timedelta(hours=8)
        
        # Create default admin user
        self._create_default_admin()
    
    def _create_default_admin(self):
        """Create default admin user for initial setup"""
        admin_id = "admin_001"
        if admin_id not in self.users:
            password_hash = bcrypt.hashpw("admin123!".encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            admin_user = User(
                id=admin_id,
                username="admin",
                email="admin@aadn.local",
                password_hash=password_hash,
                role=UserRole.ADMIN,
                created_at=datetime.utcnow()
            )
            self.users[admin_id] = admin_user
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    def create_user(self, username: str, email: str, password: str, role: UserRole) -> User:
        """Create a new user"""
        user_id = f"user_{secrets.token_urlsafe(8)}"
        password_hash = self.hash_password(password)
        
        user = User(
            id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            role=role,
            created_at=datetime.utcnow()
        )
        
        self.users[user_id] = user
        return user
    
    def authenticate(self, username: str, password: str, ip_address: str, user_agent: str) -> Optional[str]:
        """Authenticate user and return session token"""
        user = self._find_user_by_username(username)
        
        if not user:
            return None
        
        # Check if account is locked
        if user.locked_until and datetime.utcnow() < user.locked_until:
            return None
        
        # Verify password
        if not self.verify_password(password, user.password_hash):
            user.failed_login_attempts += 1
            
            # Lock account after max failed attempts
            if user.failed_login_attempts >= self.max_failed_attempts:
                user.locked_until = datetime.utcnow() + self.lockout_duration
            
            return None
        
        # Reset failed attempts on successful login
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.utcnow()
        
        # Create session
        return self._create_session(user.id, ip_address, user_agent)
    
    def _find_user_by_username(self, username: str) -> Optional[User]:
        """Find user by username"""
        for user in self.users.values():
            if user.username == username:
                return user
        return None
    
    def _create_session(self, user_id: str, ip_address: str, user_agent: str) -> str:
        """Create a new session"""
        token = jwt.encode({
            'user_id': user_id,
            'exp': datetime.utcnow() + self.session_duration,
            'iat': datetime.utcnow(),
            'ip': ip_address
        }, self.secret_key, algorithm='HS256')
        
        session = Session(
            token=token,
            user_id=user_id,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + self.session_duration,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.sessions[token] = session
        return token
    
    def validate_session(self, token: str, ip_address: str = None) -> Optional[User]:
        """Validate session token and return user"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            user_id = payload.get('user_id')
            
            # Check if session exists
            if token not in self.sessions:
                return None
            
            session = self.sessions[token]
            
            # Check if session is expired
            if datetime.utcnow() > session.expires_at:
                del self.sessions[token]
                return None
            
            # Optional IP validation
            if ip_address and session.ip_address != ip_address:
                return None
            
            # Return user
            return self.users.get(user_id)
            
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def logout(self, token: str):
        """Logout user by invalidating session"""
        if token in self.sessions:
            del self.sessions[token]
    
    def get_user_permissions(self, user: User) -> List[str]:
        """Get user permissions based on role"""
        permissions = {
            UserRole.ADMIN: [
                "read:all", "write:all", "delete:all", 
                "manage:users", "manage:system", "manage:decoys",
                "view:analytics", "manage:alerts"
            ],
            UserRole.ANALYST: [
                "read:decoys", "write:decoys", "read:analytics",
                "read:alerts", "write:alerts", "read:threats"
            ],
            UserRole.VIEWER: [
                "read:decoys", "read:analytics", "read:alerts"
            ]
        }
        return permissions.get(user.role, [])
    
    def has_permission(self, user: User, permission: str) -> bool:
        """Check if user has specific permission"""
        user_permissions = self.get_user_permissions(user)
        return permission in user_permissions or "write:all" in user_permissions
    
    def get_active_sessions(self, user_id: str) -> List[Session]:
        """Get all active sessions for a user"""
        return [
            session for session in self.sessions.values()
            if session.user_id == user_id and datetime.utcnow() < session.expires_at
        ]
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        current_time = datetime.utcnow()
        expired_tokens = [
            token for token, session in self.sessions.items()
            if current_time > session.expires_at
        ]
        
        for token in expired_tokens:
            del self.sessions[token]

# Global auth manager instance
auth_manager = AuthManager()

# FastAPI dependency functions
from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """FastAPI dependency to get current authenticated user"""
    token = credentials.credentials
    user = auth_manager.validate_session(token)
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "role": user.role.value,
        "last_login": user.last_login.isoformat() if user.last_login else None
    }

def authenticate_user(username: str, password: str, ip_address: str = "127.0.0.1", user_agent: str = "") -> dict:
    """Authenticate user and return result"""
    token = auth_manager.authenticate(username, password, ip_address, user_agent)
    
    if token:
        user = auth_manager._find_user_by_username(username)
        return {
            "success": True,
            "token": token,
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "role": user.role.value
            },
            "session_id": token[:8]  # First 8 chars as session ID
        }
    else:
        return {
            "success": False,
            "message": "Invalid credentials or account locked"
        } 