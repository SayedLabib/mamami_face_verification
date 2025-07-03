import os
import json
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    qdrant_vector_size: int = 512
    similarity_threshold: float = 0.7

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "env_file_encoding": "utf-8",
    }

# Print raw environment variables
print("RAW ENVIRONMENT VARIABLES:")
env_vars = {k: v for k, v in os.environ.items() 
            if k.upper() in ['QDRANT_VECTOR_SIZE', 'SIMILARITY_THRESHOLD']}
print(json.dumps(env_vars, indent=2))

# Try to create settings
print("\nTRYING TO CREATE SETTINGS:")
try:
    settings = Settings()
    print("Settings created successfully!")
    print(f"qdrant_vector_size = {settings.qdrant_vector_size}")
    print(f"similarity_threshold = {settings.similarity_threshold}")
except Exception as e:
    print(f"Error creating settings: {e}")

# Try direct parsing
print("\nTRYING DIRECT PARSING:")
try:
    vector_size = int(os.environ.get('QDRANT_VECTOR_SIZE', '512'))
    threshold = float(os.environ.get('SIMILARITY_THRESHOLD', '0.7'))
    print(f"Direct parsing: vector_size = {vector_size}, threshold = {threshold}")
except Exception as e:
    print(f"Error in direct parsing: {e}")
