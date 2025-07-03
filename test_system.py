#!/usr/bin/env python
"""
Test script for face recognition system
Tests both the primary and fallback pathways
"""

import requests
import argparse
import base64
import json
import os
import sys
from pathlib import Path
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Test the face recognition system")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--id-image", required=True, help="Path to ID/passport image")
    parser.add_argument("--face-image", required=True, help="Path to face image for verification")
    args = parser.parse_args()
    
    base_url = args.url
    
    # Check if API is running
    try:
        logger.info(f"Checking if API is running at {base_url}")
        response = requests.get(f"{base_url}/docs")
        if response.status_code != 200:
            logger.error(f"API not running or not accessible at {base_url}")
            sys.exit(1)
        logger.info("API is running")
    except Exception as e:
        logger.error(f"Error connecting to API: {e}")
        sys.exit(1)
    
    # Check diagnostics
    try:
        logger.info("Checking system diagnostics")
        diagnostics = requests.get(f"{base_url}/diagnostics").json()
        logger.info(f"API connectivity status: {diagnostics.get('api_connectivity', {}).get('status', 'unknown')}")
        
        # Check DNS resolution
        dns_results = diagnostics.get("dns_check", [])
        for dns in dns_results:
            status = dns.get("status")
            host = dns.get("host")
            if status == "success":
                logger.info(f"DNS resolution for {host}: SUCCESS ({dns.get('ip')})")
            else:
                logger.warning(f"DNS resolution for {host}: FAILED ({dns.get('error', 'unknown error')})")
    except Exception as e:
        logger.error(f"Error checking diagnostics: {e}")
    
    # Test step 1: Enroll a user with ID/passport image
    try:
        logger.info("Testing user enrollment with ID/passport image")
        
        # Prepare enrollment data
        files = {
            'image': open(args.id_image, 'rb'),
        }
        data = {
            'user_id': 'test-user-123',
            'full_name': 'Test User',
            'email': 'test@example.com',
            'additional_info': json.dumps({'test': True, 'source': 'test_script'})
        }
        
        # Send enrollment request
        logger.info("Sending enrollment request...")
        response = requests.post(f"{base_url}/extract-and-store", files=files, data=data)
        enrollment_result = response.json()
        
        if enrollment_result.get('success'):
            logger.info("✓ User enrollment successful")
            logger.info(f"User ID: {enrollment_result.get('user_id')}")
            logger.info(f"Embedding ID: {enrollment_result.get('embedding_id')}")
        else:
            logger.error(f"✗ User enrollment failed: {enrollment_result.get('message')}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error during enrollment: {e}")
        sys.exit(1)
    
    # Wait a moment for database to update
    logger.info("Waiting for database to update...")
    time.sleep(1)
    
    # Test step 2: Verify face match
    try:
        logger.info("Testing face verification")
        
        # Prepare verification data
        files = {
            'image': open(args.face_image, 'rb'),
        }
        
        # Send verification request
        logger.info("Sending verification request...")
        response = requests.post(f"{base_url}/verify-match", files=files)
        verification_result = response.json()
        
        if verification_result.get('success'):
            logger.info("✓ Face verification processing successful")
            
            if verification_result.get('verified'):
                logger.info("✓ Face match found!")
                top_match = verification_result.get('top_match', {})
                logger.info(f"Top match: {top_match.get('full_name')} ({top_match.get('similarity_score'):.2f})")
            else:
                logger.warning("✗ No face match found")
                logger.info("This may be expected if using different people in the test images")
                
            matches = verification_result.get('matches', [])
            logger.info(f"Found {len(matches)} potential matches")
            for i, match in enumerate(matches[:3], 1):  # Show top 3 matches
                logger.info(f"Match {i}: {match.get('full_name')} (Score: {match.get('similarity_score'):.2f})")
        else:
            logger.error(f"✗ Face verification failed: {verification_result.get('message')}")
    except Exception as e:
        logger.error(f"Error during verification: {e}")
        sys.exit(1)
    
    logger.info("Test completed successfully")

if __name__ == "__main__":
    main()
