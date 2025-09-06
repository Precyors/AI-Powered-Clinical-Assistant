import os 
from dotenv import load_dotenv

load_dotenv()
class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

settings = Config()
if not settings.GROQ_API_KEY:
    raise ValueError("Groq_api_key is required in .env file")