import os
from pathlib import Path

from dotenv import load_dotenv

# Path to the directory where config.py is located (src/)
CURRENT_DIR = Path(__file__).resolve().parent

# Path to the project root (one level up from src/)
BASE_DIR = CURRENT_DIR.parent

# Load .env from the project root
load_dotenv(BASE_DIR / ".env")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DESTINATION = os.getenv("LOG_DESTINATION", "logs/app.log")

LOG_SERIALIZE = os.getenv("LOG_SERIALIZE", "False").lower() == "true"

Path(BASE_DIR / "logs").mkdir(parents=True, exist_ok=True)
