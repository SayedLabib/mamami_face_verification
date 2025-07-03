#!/usr/bin/env python
"""
DNS resolution test script
This script tests DNS resolution for the API host and other important services
"""

import socket
import sys
import os
import time
import argparse
import subprocess

def test_dns_resolution():
    """Test DNS resolution for various hosts"""
    hosts_to_test = [
        "api-us.faceplusplus.com",
        "google.com",
        "qdrant",
        "db"
    ]
    
    results = {}
    
    print("Testing DNS resolution...")
    for host in hosts_to_test:
        try:
            ip = socket.gethostbyname(host)
            print(f"✓ {host} resolves to {ip}")
            results[host] = {
                "success": True,
                "ip": ip
            }
        except socket.gaierror as e:
            print(f"✗ {host} failed to resolve: {e}")
            results[host] = {
                "success": False,
                "error": str(e)
            }
    
    return results

def test_host_connection(host, port=80):
    """Test if a host is reachable on a specific port"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2)
    try:
        s.connect((host, port))
        s.close()
        print(f"✓ {host}:{port} is reachable")
        return True
    except Exception as e:
        print(f"✗ {host}:{port} is NOT reachable: {e}")
        return False
    finally:
        s.close()

def ping_host(host):
    """Ping a host and return True if reachable"""
    try:
        if os.name == 'nt':  # Windows
            ping_cmd = ['ping', '-n', '1', '-w', '1000', host]
        else:  # Unix/Linux
            ping_cmd = ['ping', '-c', '1', '-W', '1', host]
            
        result = subprocess.run(ping_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode == 0
    except Exception as e:
        print(f"Error pinging {host}: {e}")
        return False

def test_api_host():
    """Test API host specifically"""
    host = "api-us.faceplusplus.com"
    direct_ip = "104.21.23.75"
    
    print(f"\nTesting API host ({host})...")
    
    # Try DNS resolution
    try:
        ip = socket.gethostbyname(host)
        print(f"✓ DNS resolution: {host} -> {ip}")
    except socket.gaierror:
        print(f"✗ DNS resolution failed for {host}")
    
    # Try connecting to API host
    api_connection = test_host_connection(host, 80)
    
    # Try connecting to direct IP
    direct_ip_connection = test_host_connection(direct_ip, 80)
    
    # Try pinging
    print(f"\nPinging {host}...")
    if ping_host(host):
        print(f"✓ {host} responds to ping")
    else:
        print(f"✗ {host} does not respond to ping (this may be normal)")
    
    print(f"\nPinging {direct_ip}...")
    if ping_host(direct_ip):
        print(f"✓ {direct_ip} responds to ping")
    else:
        print(f"✗ {direct_ip} does not respond to ping (this may be normal)")
    
    return {
        "api_connection": api_connection,
        "direct_ip_connection": direct_ip_connection
    }

def main():
    parser = argparse.ArgumentParser(description="Test DNS resolution and API connectivity")
    parser.add_argument("--verbose", action="store_true", help="Show more detailed output")
    args = parser.parse_args()
    
    print("=== DNS and API Connectivity Test ===\n")
    
    dns_results = test_dns_resolution()
    api_results = test_api_host()
    
    print("\n=== Summary ===")
    if api_results["api_connection"]:
        print("✅ API host is reachable via hostname")
    elif api_results["direct_ip_connection"]:
        print("⚠️ API host is reachable via direct IP but not hostname")
        print("   System will use direct IP or fallback to local processing")
    else:
        print("❌ API host is not reachable")
        print("   System will use local processing fallback")
    
    print("\n=== Recommendations ===")
    if not api_results["api_connection"] and api_results["direct_ip_connection"]:
        print("1. Check DNS configuration in Docker")
        print("2. Ensure 'extra_hosts' is correctly set in docker-compose.yml")
    elif not api_results["api_connection"] and not api_results["direct_ip_connection"]:
        print("1. Check internet connectivity")
        print("2. Ensure Docker has network access")
        print("3. Check if API service is currently available")
    
if __name__ == "__main__":
    main()
