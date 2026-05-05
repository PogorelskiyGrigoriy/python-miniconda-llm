import os
from abc import ABC, abstractmethod
import joblib

from src.logger_config import setup_logging

logger = setup_logging()


class TextProcessor(ABC):
    def __init__(self, source: str):
        if not source:
            raise ValueError("Source cannot be empty")

        self.source = source
        self.model = None

        if self._is_path(source):
            logger.info(f"Model loaded from: {source}")
            self.load_model(source)
        else:
            logger.info("Raw text provided. Training model...")
            self.train_model(source)

    def _is_path(self, source):
        # Simple method to check path
        return isinstance(source, str) and os.path.isfile(source)

    @abstractmethod
    def train_model(self, text):
        pass

    def load_model(self, path):
        try:
            self.model = joblib.load(path)
        except Exception as e:
            logger.error(f"Error loading model from {path}: {e}")
            raise
        
    def save_model(self, path):
        try:
            joblib.dump(self.model, path)
        except Exception as e:
            logger.error(f"Error saving model to {path}: {e}")
            raise
        
    @abstractmethod
    def predict(self, query):
        pass
