#!/usr/bin/env python3
"""
AADN Decoy Testing Script
Tests deployed decoys by making connections and verifying responses
"""

import asyncio
import socket
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.core.logging_config import setup_logging
import logging

logger = logging.getLogger(__name__)


async def test_ssh_decoy(host: str, port: int) -> bool:
    """Test SSH decoy connection"""
    try:
        logger.info(f"Testing SSH decoy at {host}:{port}")
        
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=10
        )
        
        # Read SSH banner
        banner = await asyncio.wait_for(reader.readline(), timeout=5)
        banner_str = banner.decode().strip()
        
        logger.info(f"SSH banner received: {banner_str}")
        
        # Send client identification
        writer.write(b"SSH-2.0-TestClient\r\n")
        await writer.drain()
        
        # Wait a bit for any response
        await asyncio.sleep(1)
        
        writer.close()
        await writer.wait_closed()
        
        return True
        
    except Exception as e:
        logger.error(f"SSH test failed: {e}")
        return False


async def test_http_decoy(host: str, port: int) -> bool:
    """Test HTTP decoy connection"""
    try:
        logger.info(f"Testing HTTP decoy at {host}:{port}")
        
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=10
        )
        
        # Send HTTP request
        request = f"GET / HTTP/1.1\r\nHost: {host}\r\nUser-Agent: TestClient/1.0\r\n\r\n"
        writer.write(request.encode())
        await writer.drain()
        
        # Read response
        response = await asyncio.wait_for(reader.read(1024), timeout=5)
        response_str = response.decode()
        
        logger.info(f"HTTP response received: {len(response_str)} bytes")
        logger.debug(f"Response content: {response_str[:200]}...")
        
        writer.close()
        await writer.wait_closed()
        
        return "HTTP/1.1" in response_str
        
    except Exception as e:
        logger.error(f"HTTP test failed: {e}")
        return False


async def test_ftp_decoy(host: str, port: int) -> bool:
    """Test FTP decoy connection"""
    try:
        logger.info(f"Testing FTP decoy at {host}:{port}")
        
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=10
        )
        
        # Read FTP welcome banner
        banner = await asyncio.wait_for(reader.readline(), timeout=5)
        banner_str = banner.decode().strip()
        
        logger.info(f"FTP banner received: {banner_str}")
        
        # Send USER command
        writer.write(b"USER testuser\r\n")
        await writer.drain()
        
        # Read response
        response = await asyncio.wait_for(reader.readline(), timeout=5)
        response_str = response.decode().strip()
        
        logger.info(f"FTP USER response: {response_str}")
        
        # Send QUIT command
        writer.write(b"QUIT\r\n")
        await writer.drain()
        
        writer.close()
        await writer.wait_closed()
        
        return "220" in banner_str
        
    except Exception as e:
        logger.error(f"FTP test failed: {e}")
        return False


async def test_telnet_decoy(host: str, port: int) -> bool:
    """Test Telnet decoy connection"""
    try:
        logger.info(f"Testing Telnet decoy at {host}:{port}")
        
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=10
        )
        
        # Read login prompt
        prompt = await asyncio.wait_for(reader.read(1024), timeout=5)
        prompt_str = prompt.decode()
        
        logger.info(f"Telnet prompt received: {prompt_str.strip()}")
        
        # Send username
        writer.write(b"testuser\r\n")
        await writer.drain()
        
        # Read password prompt
        password_prompt = await asyncio.wait_for(reader.read(1024), timeout=5)
        password_prompt_str = password_prompt.decode()
        
        logger.info(f"Password prompt: {password_prompt_str.strip()}")
        
        # Send password
        writer.write(b"testpass\r\n")
        await writer.drain()
        
        # Wait for response
        await asyncio.sleep(1)
        
        writer.close()
        await writer.wait_closed()
        
        return "login:" in prompt_str.lower()
        
    except Exception as e:
        logger.error(f"Telnet test failed: {e}")
        return False


def test_port_connectivity(host: str, port: int) -> bool:
    """Test basic port connectivity"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


async def run_decoy_tests():
    """Run tests against common decoy ports"""
    
    # Common test configurations
    test_configs = [
        {"name": "SSH Honeypot", "host": "127.0.0.1", "port": 2222, "test_func": test_ssh_decoy},
        {"name": "HTTP Web Server", "host": "127.0.0.1", "port": 8080, "test_func": test_http_decoy},
        {"name": "FTP Server", "host": "127.0.0.1", "port": 2121, "test_func": test_ftp_decoy},
        {"name": "Telnet Service", "host": "127.0.0.1", "port": 2323, "test_func": test_telnet_decoy},
    ]
    
    print("\n" + "="*60)
    print("AADN Decoy Testing")
    print("="*60)
    
    results = []
    
    for config in test_configs:
        print(f"\nTesting {config['name']} on {config['host']}:{config['port']}...")
        
        # First check basic connectivity
        if not test_port_connectivity(config['host'], config['port']):
            print(f"❌ Port {config['port']} is not accessible")
            results.append({"name": config['name'], "success": False, "reason": "Port not accessible"})
            continue
        
        # Run specific test
        try:
            success = await config['test_func'](config['host'], config['port'])
            if success:
                print(f"✅ {config['name']} test passed")
                results.append({"name": config['name'], "success": True})
            else:
                print(f"❌ {config['name']} test failed")
                results.append({"name": config['name'], "success": False, "reason": "Test failed"})
        except Exception as e:
            print(f"❌ {config['name']} test error: {e}")
            results.append({"name": config['name'], "success": False, "reason": str(e)})
        
        # Small delay between tests
        await asyncio.sleep(1)
    
    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for r in results if r['success'])
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Your decoys are working correctly.")
    else:
        print("\n❌ Some tests failed:")
        for result in results:
            if not result['success']:
                reason = result.get('reason', 'Unknown error')
                print(f"  - {result['name']}: {reason}")
        
        print("\nTroubleshooting tips:")
        print("1. Make sure the AADN application is running")
        print("2. Check that decoys are deployed: python scripts/deploy_sample_decoys.py")
        print("3. Verify decoy status via API: curl http://localhost:8000/api/v1/decoys")
        print("4. Check application logs for errors")


async def main():
    """Main test function"""
    setup_logging()
    
    print("Starting AADN decoy tests...")
    print("This will test connectivity to common decoy ports.")
    print("Make sure AADN is running and decoys are deployed before running this test.")
    
    input("\nPress Enter to continue or Ctrl+C to cancel...")
    
    try:
        await run_decoy_tests()
    except KeyboardInterrupt:
        print("\nTest cancelled by user.")
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        print(f"❌ Test execution failed: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 