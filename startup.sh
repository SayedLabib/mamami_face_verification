#!/bin/bash
# startup.sh - Helper script to ensure proper system configuration

# Check if we can resolve the API host
if getent hosts api-us.faceplusplus.com > /dev/null; then
    echo "API host resolution successful"
else
    echo "WARNING: Cannot resolve api-us.faceplusplus.com"
fi

echo "Running pre-startup configuration checks..."

# Check DNS configuration
echo "Using custom DNS configuration..."
cat /app/dns_config/resolv.conf

# Check DNS resolution
echo "Testing DNS resolution..."
echo "Resolving api-us.faceplusplus.com..."
getent hosts api-us.faceplusplus.com || echo "Could not resolve with getent, trying nslookup..."
nslookup api-us.faceplusplus.com

echo "Resolving google.com (connectivity test)..."
getent hosts google.com
nslookup google.com

echo "Resolving qdrant (internal service)..."
getent hosts qdrant
nslookup qdrant

# Check network connectivity
echo "Testing network connectivity..."
curl --connect-timeout 5 -I -s https://api-us.faceplusplus.com > /dev/null && echo "API host reachable via HTTPS" || echo "Could not connect to API host via HTTPS"

# Check ports and services
echo "Testing required services..."
curl -s -o /dev/null -w "Qdrant status: %{http_code}\n" http://qdrant:6333/collections || echo "Qdrant service not available yet"

# Set up and check Qdrant collection
echo "Setting up Qdrant collection if needed..."
python -c "
import asyncio
from app.services.service import QdrantService

async def init_qdrant():
    service = QdrantService()
    await service.initialize_collection(recreate=False)

if __name__ == '__main__':
    asyncio.run(init_qdrant())
"

# Check if API host resolves to the expected IP
if getent hosts api-us.faceplusplus.com > /dev/null; then
    echo "API host resolves successfully"
else
    echo "WARNING: Cannot resolve api-us.faceplusplus.com"
    echo "The application will use direct IP access or local fallback"
fi

# Start the application
echo "Starting FastAPI application..."
exec "$@"
