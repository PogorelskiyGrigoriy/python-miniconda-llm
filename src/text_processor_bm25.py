import re
import numpy as np
from rank_bm25 import BM25Okapi

from src.logger_config import setup_logging
from src.text_processor import TextProcessor

logger = setup_logging()

class TextProcessorBM25(TextProcessor):
    def train_model(self, text: str):
        """
        Splits text into sentences using regex, tokenizes them, and trains the BM25 model.
        """
        # Split text into sentences handling '.', '!', and '?'
        sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]

        if not sentences:
            raise ValueError("No valid sentences found in the input text.")

        # BM25 requires a tokenized corpus (list of word lists)
        # Simple tokenization: lowercase and split by whitespace
        tokenized_corpus = [sentence.lower().split() for sentence in sentences]

        # Initialize and train the BM25 model
        bm25 = BM25Okapi(tokenized_corpus)

        # Store sentences and the model in a standardized dictionary for persistence
        self.model = {
            "sentences": sentences,
            "bm25": bm25,
        }
        logger.info(f"BM25 model trained on {len(sentences)} sentences.")

    def predict(self, query: str):
        """
        Finds and returns the most relevant sentence using BM25 ranking.
        """
        if not self.model:
            logger.error("Predict failed: Model is not initialized.")
            return None

        # Unpack model components
        bm25 = self.model["bm25"]
        sentences = self.model["sentences"]

        # Tokenize the query identically to the training corpus
        tokenized_query = query.lower().split()

        # Retrieve relevance scores for each sentence
        scores = bm25.get_scores(tokenized_query)

        # Identify the index of the highest relevance score
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]

        # BM25 scores can exceed 1.0 (unlike cosine similarity)
        # A score > 0 indicates at least some level of matching
        if best_score > 0:
            result_text = sentences[best_idx]
            logger.info(f"Best match found (score: {best_score:.4f}): {result_text}")
            return {"score": float(best_score), "text": result_text}

        logger.warning("No relevant matches found for the query.")
        return None