import os
from pathlib import Path

from dotenv import load_dotenv

# --- Project Root Configuration ---
# __file__ is the path to this file (src/config.py)
# .resolve() gets the absolute path, .parent.parent moves two levels up to project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from the .env file located in the project root
load_dotenv(BASE_DIR / ".env")


def get_absolute_path(env_key: str, default_val: str = None) -> str:
    """
    Utility function to convert any path from .env into an absolute path.

    If the path provided in .env is already absolute (e.g., C:/... or /usr/...),
    it remains unchanged. If it's relative (e.g., data/raw/...), it is
    joined with the project's BASE_DIR.
    """
    path_str = os.getenv(env_key, default_val)
    if not path_str:
        return None

    path_obj = Path(path_str)

    # Check if the path is relative and prepend BASE_DIR if necessary
    if not path_obj.is_absolute():
        path_obj = BASE_DIR / path_obj

    return str(path_obj)


# --- Data Path Settings ---
# Path to the main dataset (e.g., vehicles.csv)
VEHICLES_PATH = get_absolute_path("VEHICLES_PATH")

# Path for the first version of cleaned data
CLEANED_V1_PATH = get_absolute_path(
    "CLEANED_V1_PATH", "data/processed/vehicles-cleaned-v1.csv"
)

# --- Logging Configuration ---
# Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Path to the application log file
LOG_DESTINATION = get_absolute_path("LOG_DESTINATION", "logs/app.log")

# Whether to serialize logs into JSON format (useful for production)
LOG_SERIALIZE = os.getenv("LOG_SERIALIZE", "False").lower() == "true"

# --- Infrastructure Setup ---
# Automatically create the log directory if it doesn't exist
if LOG_DESTINATION:
    Path(LOG_DESTINATION).parent.mkdir(parents=True, exist_ok=True)

if CLEANED_V1_PATH:
    Path(CLEANED_V1_PATH).parent.mkdir(parents=True, exist_ok=True)
