from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Base directories
    BASE_DIR: Path = Path(__file__).resolve().parent
    RAW_DIR: Path = BASE_DIR / "raw"
    CLEAN_DIR: Path = BASE_DIR / "clean"
    EMBEDDINGS_DIR: Path = BASE_DIR / "embeddings_ready"

    class Config:
        env_file = ".env"

settings = Settings()

for s in settings.RAW_DIR.glob("*.csv"):
    print(s)