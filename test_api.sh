#!/bin/bash
# Test script for the dating chat API

echo "Testing the Dating Chat System..."

# Test the main health endpoint
echo "1. Testing health check..."
curl -X GET "http://localhost:8080/" || echo "Health check failed"

echo -e "\n2. Testing dating chat endpoint..."
curl -X POST "http://localhost:8080/dating-chat/chat" \
  -H "Content-Type: application/json" \
  -d '{"user_message": "Hello beautiful, how are you today?"}'

echo -e "\n3. Testing chat history..."
curl -X GET "http://localhost:8080/dating-chat/history"

echo -e "\n4. Testing another chat message..."
curl -X POST "http://localhost:8080/dating-chat/chat" \
  -H "Content-Type: application/json" \
  -d '{"user_message": "I missed you today, what did you do?"}'

echo -e "\n5. Testing history clear..."
curl -X DELETE "http://localhost:8080/dating-chat/history"

echo -e "\nTesting complete!"
