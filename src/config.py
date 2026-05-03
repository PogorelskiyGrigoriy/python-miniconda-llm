import os
from pathlib import Path
from dotenv import load_dotenv

# Путь к директории, где находится config.py (src/)
CURRENT_DIR = Path(__file__).resolve().parent

# Путь к корню проекта (на уровень выше от src/)
BASE_DIR = CURRENT_DIR.parent

# Загружаем .env из корня проекта
load_dotenv(BASE_DIR / ".env")

# Настройки логирования
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DESTINATION = os.getenv("LOG_DESTINATION", "logs/app.log")
LOG_SERIALIZE = os.getenv("LOG_SERIALIZE", "False").lower() == "true"

# Путь к данным о машинах
VEHICLES_PATH = os.getenv("VEHICLES_PATH")

# Создание папки для логов в корне проекта
Path(BASE_DIR / "logs").mkdir(parents=True, exist_ok=True)