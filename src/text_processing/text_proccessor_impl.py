import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.text_processing.text_proccessor import TextProcessor
from src.logger_config import setup_logging

logger = setup_logging()

class TextProcessorTfidf(TextProcessor):
    def train_model(self, text: str):
        sentences = [s.strip() for s in text.split('.') if s.strip()]        
        vectorizer = TfidfVectorizer(stop_words='english')
        embeddings = vectorizer.fit_transform(sentences)
        
        # 3. Packaging: store everything in a dictionary for easy persistence
        # This allows save_model() from the base class to save all components at once
        self.model = {
            "sentences": sentences,
            "vectorizer": vectorizer,
            "embeddings": embeddings
        }
        logger.info(f"Model trained on {len(sentences)} sentences.")

    def predict(self, query: str):
        """
        Finds and returns the most semantically similar sentence for a given query.
        """
        if not self.model:
            logger.error("Predict failed: Model is not initialized.")
            return None

        # Extract components from the stored model package
        vectorizer = self.model["vectorizer"]
        embeddings = self.model["embeddings"]
        sentences = self.model["sentences"]

        # 1. Transform the incoming query into a vector
        query_vec = vectorizer.transform([query])

        # 2. Calculate cosine similarity between query and all stored sentences
        similarities = cosine_similarity(query_vec, embeddings)
        
        # 3. Find the index of the highest similarity score
        best_idx = np.argmax(similarities)
        best_score = similarities[0][best_idx]

        # Check if the match is relevant (score > 0)
        if best_score > 0:
            result_text = sentences[best_idx]
            logger.info(f"Best match found ({best_score:.4f}): {result_text}")
            return {"score": float(best_score), "text": result_text}
        
        logger.warning("No relevant matches found for the query.")
        return None