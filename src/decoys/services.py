"""
AADN Decoy Services
Service implementations for different decoy types
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any

from .models import DecoyType, DecoyConfiguration

logger = logging.getLogger("aadn.decoys")


class BaseDecoyService(ABC):
    """Base class for all decoy services"""
    
    def __init__(self, config: DecoyConfiguration):
        self.config = config
        self.is_running = False
        self.server = None
    
    @abstractmethod
    async def start(self, host: str, port: int):
        """Start the decoy service"""
        pass
    
    @abstractmethod
    async def stop(self):
        """Stop the decoy service"""
        pass
    
    @abstractmethod
    async def handle_connection(self, reader, writer):
        """Handle incoming connections"""
        pass


class SSHDecoyService(BaseDecoyService):
    """SSH honeypot service"""
    
    async def start(self, host: str, port: int):
        """Start SSH decoy service"""
        self.server = await asyncio.start_server(
            self.handle_connection, host, port
        )
        self.is_running = True
        logger.info(f"SSH decoy started on {host}:{port}")
    
    async def stop(self):
        """Stop SSH decoy service"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        self.is_running = False
        logger.info("SSH decoy stopped")
    
    async def handle_connection(self, reader, writer):
        """Handle SSH connection"""
        try:
            # Send SSH banner
            writer.write(b"SSH-2.0-OpenSSH_8.0\r\n")
            await writer.drain()
            
            # Read client data
            data = await reader.read(1024)
            logger.info(f"SSH connection from {writer.get_extra_info('peername')}")
            
            # Close connection
            writer.close()
            await writer.wait_closed()
        except Exception as e:
            logger.error(f"SSH connection error: {e}")


class HTTPDecoyService(BaseDecoyService):
    """HTTP honeypot service"""
    
    async def start(self, host: str, port: int):
        """Start HTTP decoy service"""
        self.server = await asyncio.start_server(
            self.handle_connection, host, port
        )
        self.is_running = True
        logger.info(f"HTTP decoy started on {host}:{port}")
    
    async def stop(self):
        """Stop HTTP decoy service"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        self.is_running = False
        logger.info("HTTP decoy stopped")
    
    async def handle_connection(self, reader, writer):
        """Handle HTTP connection"""
        try:
            # Read HTTP request
            data = await reader.read(1024)
            request = data.decode('utf-8', errors='ignore')
            logger.info(f"HTTP request from {writer.get_extra_info('peername')}: {request[:100]}")
            
            # Send HTTP response
            response = (
                "HTTP/1.1 200 OK\r\n"
                "Server: Apache/2.4.41\r\n"
                "Content-Type: text/html\r\n"
                "Content-Length: 13\r\n"
                "\r\n"
                "Hello, World!"
            )
            writer.write(response.encode())
            await writer.drain()
            
            # Close connection
            writer.close()
            await writer.wait_closed()
        except Exception as e:
            logger.error(f"HTTP connection error: {e}")


class FTPDecoyService(BaseDecoyService):
    """FTP honeypot service"""
    
    async def start(self, host: str, port: int):
        """Start FTP decoy service"""
        self.server = await asyncio.start_server(
            self.handle_connection, host, port
        )
        self.is_running = True
        logger.info(f"FTP decoy started on {host}:{port}")
    
    async def stop(self):
        """Stop FTP decoy service"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        self.is_running = False
        logger.info("FTP decoy stopped")
    
    async def handle_connection(self, reader, writer):
        """Handle FTP connection"""
        try:
            # Send FTP welcome banner
            writer.write(b"220 ProFTPD 1.3.6 Server ready\r\n")
            await writer.drain()
            
            # Read client commands
            data = await reader.read(1024)
            logger.info(f"FTP connection from {writer.get_extra_info('peername')}")
            
            # Close connection
            writer.close()
            await writer.wait_closed()
        except Exception as e:
            logger.error(f"FTP connection error: {e}")


class TelnetDecoyService(BaseDecoyService):
    """Telnet honeypot service"""
    
    async def start(self, host: str, port: int):
        """Start Telnet decoy service"""
        self.server = await asyncio.start_server(
            self.handle_connection, host, port
        )
        self.is_running = True
        logger.info(f"Telnet decoy started on {host}:{port}")
    
    async def stop(self):
        """Stop Telnet decoy service"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        self.is_running = False
        logger.info("Telnet decoy stopped")
    
    async def handle_connection(self, reader, writer):
        """Handle Telnet connection"""
        try:
            # Send login prompt
            writer.write(b"Ubuntu 20.04.3 LTS\r\nlogin: ")
            await writer.drain()
            
            # Read client data
            data = await reader.read(1024)
            logger.info(f"Telnet connection from {writer.get_extra_info('peername')}")
            
            # Close connection
            writer.close()
            await writer.wait_closed()
        except Exception as e:
            logger.error(f"Telnet connection error: {e}")


class DecoyServiceFactory:
    """Factory for creating decoy services"""
    
    def create_service(self, decoy_type: DecoyType, config: DecoyConfiguration) -> BaseDecoyService:
        """Create a decoy service based on type"""
        if decoy_type == DecoyType.SSH:
            return SSHDecoyService(config)
        elif decoy_type == DecoyType.HTTP:
            return HTTPDecoyService(config)
        elif decoy_type == DecoyType.FTP:
            return FTPDecoyService(config)
        elif decoy_type == DecoyType.TELNET:
            return TelnetDecoyService(config)
        else:
            raise ValueError(f"Unsupported decoy type: {decoy_type}") 