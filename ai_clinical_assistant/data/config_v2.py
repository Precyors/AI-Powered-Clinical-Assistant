from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR /"raw"
CLEAN_DIR = BASE_DIR / "clean"
EMBEDDINGS_DIR: Path = BASE_DIR / "embeddings_ready"

# Ensure dirs exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)