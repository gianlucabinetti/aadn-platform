#!/usr/bin/env python3
"""
AADN Sample Decoy Deployment Script
Deploys sample decoys for testing and demonstration
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.core.config import get_settings
from src.core.database import init_databases
from src.core.logging_config import setup_logging
from src.decoys.manager import decoy_manager
from src.decoys.models import (
    DecoyDeploymentRequest, DecoyType, DecoyConfiguration,
    DecoyMetadata, InteractionLevel
)

logger = logging.getLogger(__name__)


async def deploy_sample_decoys():
    """Deploy a set of sample decoys for testing"""
    
    sample_decoys = [
        {
            "name": "SSH Honeypot",
            "type": DecoyType.SSH,
            "configuration": DecoyConfiguration(
                port=2222,
                interaction_level=InteractionLevel.MEDIUM,
                custom_banner="SSH-2.0-OpenSSH_8.0",
                response_delay=0.2
            ),
            "metadata": DecoyMetadata(
                tags=["ssh", "linux", "demo"],
                environment="demo",
                purpose="SSH attack detection",
                notes="Sample SSH honeypot for demonstration"
            )
        },
        {
            "name": "HTTP Web Server",
            "type": DecoyType.HTTP,
            "configuration": DecoyConfiguration(
                port=8080,
                interaction_level=InteractionLevel.MEDIUM,
                custom_banner="Apache/2.4.41 (Ubuntu)",
                response_delay=0.1
            ),
            "metadata": DecoyMetadata(
                tags=["http", "web", "demo"],
                environment="demo",
                purpose="Web attack detection",
                notes="Sample HTTP server honeypot"
            )
        },
        {
            "name": "FTP Server",
            "type": DecoyType.FTP,
            "configuration": DecoyConfiguration(
                port=2121,
                interaction_level=InteractionLevel.LOW,
                custom_banner="220 ProFTPD 1.3.6 Server ready",
                response_delay=0.3
            ),
            "metadata": DecoyMetadata(
                tags=["ftp", "file-transfer", "demo"],
                environment="demo",
                purpose="FTP attack detection",
                notes="Sample FTP honeypot"
            )
        },
        {
            "name": "Telnet Service",
            "type": DecoyType.TELNET,
            "configuration": DecoyConfiguration(
                port=2323,
                interaction_level=InteractionLevel.MEDIUM,
                response_delay=0.5
            ),
            "metadata": DecoyMetadata(
                tags=["telnet", "remote-access", "demo"],
                environment="demo",
                purpose="Telnet attack detection",
                notes="Sample Telnet honeypot"
            )
        }
    ]
    
    deployed_decoys = []
    
    for decoy_config in sample_decoys:
        try:
            logger.info(f"Deploying {decoy_config['name']}...")
            
            request = DecoyDeploymentRequest(
                name=decoy_config["name"],
                type=decoy_config["type"],
                configuration=decoy_config["configuration"],
                metadata=decoy_config["metadata"],
                auto_start=True
            )
            
            decoy = await decoy_manager.create_decoy(request, user_id="demo_script")
            deployed_decoys.append(decoy)
            
            logger.info(f"Successfully deployed {decoy.name} (ID: {decoy.id}) on port {decoy.configuration.port}")
            
        except Exception as e:
            logger.error(f"Failed to deploy {decoy_config['name']}: {e}")
    
    return deployed_decoys


async def show_deployment_summary(decoys):
    """Show summary of deployed decoys"""
    print("\n" + "="*60)
    print("AADN Sample Decoys Deployment Summary")
    print("="*60)
    
    if not decoys:
        print("No decoys were successfully deployed.")
        return
    
    print(f"Successfully deployed {len(decoys)} decoys:\n")
    
    for decoy in decoys:
        print(f"• {decoy.name}")
        print(f"  Type: {decoy.type}")
        print(f"  Host: {decoy.host}")
        print(f"  Port: {decoy.configuration.port}")
        print(f"  Status: {decoy.status}")
        print(f"  ID: {decoy.id}")
        print()
    
    print("You can now test the decoys by connecting to them:")
    print()
    
    for decoy in decoys:
        if decoy.type == DecoyType.SSH:
            print(f"  SSH: ssh -p {decoy.configuration.port} user@{decoy.host}")
        elif decoy.type == DecoyType.HTTP:
            print(f"  HTTP: curl http://{decoy.host}:{decoy.configuration.port}")
        elif decoy.type == DecoyType.FTP:
            print(f"  FTP: ftp {decoy.host} {decoy.configuration.port}")
        elif decoy.type == DecoyType.TELNET:
            print(f"  Telnet: telnet {decoy.host} {decoy.configuration.port}")
    
    print("\nMonitor interactions via the API:")
    print("  GET http://localhost:8000/api/v1/monitoring/interactions")
    print("  GET http://localhost:8000/api/v1/monitoring/stats")
    print("\nView the dashboard at: http://localhost:3000")
    print("\nAPI documentation: http://localhost:8000/docs")


async def main():
    """Main deployment function"""
    setup_logging()
    logger.info("Starting AADN sample decoy deployment...")
    
    try:
        # Initialize database connections
        await init_databases()
        logger.info("Database connections initialized")
        
        # Deploy sample decoys
        decoys = await deploy_sample_decoys()
        
        # Show summary
        await show_deployment_summary(decoys)
        
        logger.info("Sample decoy deployment completed!")
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 