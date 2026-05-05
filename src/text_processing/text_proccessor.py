import os
from abc import ABC, abstractmethod
import joblib

from src.logger_config import setup_logging
logger = setup_logging()

class TextProcessor(ABC):
    def __init__(self):
        """
        Initialize the processor with no model loaded.
        """
        self.model = None

    @classmethod
    def from_file(cls, path: str):
        """
        Factory method to create an instance and load a model from a file.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file not found: {path}")
            
        instance = cls()
        instance.load_model(path)
        return instance

    @classmethod
    def from_text(cls, text: str):
        """
        Factory method to create an instance and train a new model on the provided text.
        """
        if not text or not text.strip():
            raise ValueError("Training text cannot be empty")
            
        instance = cls()
        instance.train_model(text)
        return instance

    @abstractmethod
    def train_model(self, text: str):
        pass

    def load_model(self, path: str):
        try:
            self.model = joblib.load(path)
            logger.info(f"Model successfully loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model from {path}: {e}")
            raise

    def save_model(self, path: str):
        if self.model is None:
            logger.warning("Attempted to save an uninitialized model.")
            raise RuntimeError("Cannot save model: it has not been trained or loaded yet.")           
        try:
            # Ensure the target directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump(self.model, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model to {path}: {e}")
            raise

    @abstractmethod
    def predict(self, query: str):
        pass
