#!/bin/bash
# startup.sh - Simple helper script to initialize Qdrant collection and start application

echo "Setting up Qdrant collection..."
python -c "
import asyncio
from app.services.service import QdrantService

async def init_qdrant():
    service = QdrantService()
    await service.initialize_collection(recreate=False)

if __name__ == '__main__':
    asyncio.run(init_qdrant())
"

# Start the application
echo "Starting FastAPI application..."
exec "$@"
