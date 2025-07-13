import os
from dotenv import load_dotenv
            
load_dotenv()

class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            # Face++ API settings
            cls._instance.fpp_api_key = os.getenv("FPP_API_KEY")
            cls._instance.fpp_api_secret = os.getenv("FPP_API_SECRET")
            cls._instance.fpp_create = os.getenv("FPP_CREATE")
            cls._instance.fpp_detect = os.getenv("FPP_DETECT")
            cls._instance.fpp_search = os.getenv("FPP_SEARCH")
            cls._instance.fpp_add = os.getenv("FPP_ADD")
            cls._instance.fpp_get_detail = os.getenv("FPP_GET_DETAIL")

            # OpenAI API settings
            cls._instance.openai_api_key = os.getenv("OPENAI_API_KEY")
            cls._instance.openai_model = os.getenv("OPENAI_MODEL")

        return cls._instance

class Settings:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_API_URL: str = os.getenv("GROQ_API_URL", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "")

settings = Settings()        