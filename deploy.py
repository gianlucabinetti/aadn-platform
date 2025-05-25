#!/usr/bin/env python3
"""
AADN Deployment Script
Automated deployment helper for various platforms
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List

def run_command(cmd: str, cwd: str = None) -> bool:
    """Run a shell command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {cmd}")
            return True
        else:
            print(f"❌ {cmd}")
            print(f"   Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {cmd}")
        print(f"   Exception: {e}")
        return False

def create_github_repo():
    """Guide for creating GitHub repository"""
    print("\n🐙 GitHub Repository Setup")
    print("=" * 40)
    print("1. Go to https://github.com/new")
    print("2. Repository name: aadn-cybersecurity-platform")
    print("3. Description: 'Enterprise-grade AI-driven cybersecurity deception network'")
    print("4. Make it Public (for portfolio)")
    print("5. Don't initialize with README (we have one)")
    print("\nThen run these commands:")
    print("git init")
    print("git add .")
    print("git commit -m 'Initial AADN implementation - Enterprise cybersecurity platform'")
    print("git branch -M main")
    print("git remote add origin https://github.com/YOUR_USERNAME/aadn-cybersecurity-platform.git")
    print("git push -u origin main")

def create_vercel_config():
    """Create Vercel configuration for frontend deployment"""
    vercel_config = {
        "version": 2,
        "name": "aadn-dashboard",
        "builds": [
            {
                "src": "web/package.json",
                "use": "@vercel/static-build",
                "config": {
                    "distDir": "dist"
                }
            }
        ],
        "routes": [
            {
                "src": "/api/(.*)",
                "dest": "https://your-backend-url.com/api/$1"
            },
            {
                "src": "/(.*)",
                "dest": "/web/dist/$1"
            }
        ]
    }
    
    with open("vercel.json", "w") as f:
        json.dump(vercel_config, f, indent=2)
    
    print("✅ Created vercel.json")

def create_railway_config():
    """Create Railway configuration for backend deployment"""
    railway_config = {
        "deploy": {
            "startCommand": "python main_production.py",
            "healthcheckPath": "/health"
        }
    }
    
    with open("railway.json", "w") as f:
        json.dump(railway_config, f, indent=2)
    
    print("✅ Created railway.json")

def create_docker_config():
    """Create Docker configuration"""
    dockerfile_content = """# AADN Production Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 aadn && chown -R aadn:aadn /app
USER aadn

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Start application
CMD ["python", "main_production.py"]
"""
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)
    
    # Create .dockerignore
    dockerignore_content = """node_modules
.git
.gitignore
README.md
Dockerfile
.dockerignore
.env
*.log
"""
    
    with open(".dockerignore", "w") as f:
        f.write(dockerignore_content)
    
    print("✅ Created Dockerfile and .dockerignore")

def create_env_template():
    """Create environment template"""
    env_template = """# AADN Environment Configuration
# Copy this file to .env and update values

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false

# Database Configuration (optional)
MONGODB_URL=mongodb://localhost:27017
REDIS_URL=redis://localhost:6379

# Security
SECRET_KEY=your-secret-key-here

# Logging
LOG_LEVEL=INFO

# External Services (optional)
SLACK_WEBHOOK_URL=
EMAIL_SMTP_SERVER=
EMAIL_USERNAME=
EMAIL_PASSWORD=
"""
    
    with open(".env.template", "w") as f:
        f.write(env_template)
    
    print("✅ Created .env.template")

def create_deployment_docs():
    """Create deployment documentation"""
    docs_content = """# AADN Deployment Guide

## Quick Deploy Options

### 1. Frontend (React Dashboard)
- **Vercel**: Connect GitHub repo, auto-deploy from main branch
- **Netlify**: Drag & drop `web/dist` folder
- **GitHub Pages**: Enable in repo settings

### 2. Backend (FastAPI API)
- **Railway**: Connect GitHub repo, auto-deploy
- **Render**: Connect GitHub repo, use `python main_production.py`
- **Heroku**: Use Procfile: `web: python main_production.py`

### 3. Full Stack
- **Docker**: `docker build -t aadn . && docker run -p 8000:8000 aadn`
- **AWS/GCP/Azure**: Use container services

## Environment Variables

Copy `.env.template` to `.env` and configure:
- `API_HOST`: Server host (0.0.0.0 for production)
- `API_PORT`: Server port (8000)
- `DEBUG`: false for production

## Database Setup (Optional)

For production with real databases:
1. MongoDB Atlas (free tier)
2. Redis Cloud (free tier)
3. Update connection strings in `.env`

## Custom Domain

1. Purchase domain
2. Configure DNS to point to your deployment
3. Enable SSL/HTTPS
4. Update CORS settings in `main_production.py`

## Monitoring

- Health check: `GET /health`
- Metrics: `GET /api/v1/stats`
- Logs: Check platform-specific logging

## Security Checklist

- [ ] Change default SECRET_KEY
- [ ] Configure CORS properly
- [ ] Enable HTTPS
- [ ] Set up monitoring/alerting
- [ ] Regular security updates
"""
    
    with open("DEPLOYMENT.md", "w") as f:
        f.write(docs_content)
    
    print("✅ Created DEPLOYMENT.md")

def main():
    """Main deployment setup function"""
    print("🚀 AADN Deployment Setup")
    print("=" * 50)
    
    print("\n📋 Creating deployment configurations...")
    create_vercel_config()
    create_railway_config()
    create_docker_config()
    create_env_template()
    create_deployment_docs()
    
    print("\n🔧 Deployment configurations created!")
    print("\n📚 Next Steps:")
    print("1. Run system check: python check_system.py")
    print("2. Set up GitHub repository (see instructions above)")
    print("3. Choose deployment platform:")
    print("   - Frontend: Vercel/Netlify")
    print("   - Backend: Railway/Render")
    print("   - Full: Docker/Cloud platforms")
    print("4. Configure environment variables")
    print("5. Deploy and test!")
    
    create_github_repo()

if __name__ == "__main__":
    main() 