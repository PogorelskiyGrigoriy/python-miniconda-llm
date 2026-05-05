import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.text_processing.text_proccessor import TextProcessor
from src.logger_config import setup_logging

logger = setup_logging()

class TextProccessorTfidf(TextProcessor):
    def train_model(self, text: str):
        """Разбивает текст на предложения и обучает TF-IDF"""
        # 1. Подготовка данных
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # 2. Векторизация
        vectorizer = TfidfVectorizer(stop_words='english')
        embeddings = vectorizer.fit_transform(sentences)
        
        # 3. Сохраняем все необходимое в self.model как один пакет (dict)
        # Это позволит методу save_model из базы сохранить всё одним файлом
        self.model = {
            "sentences": sentences,
            "vectorizer": vectorizer,
            "embeddings": embeddings
        }
        logger.info(f"Model trained on {len(sentences)} sentences.")

    def predict(self, query: str):
        """Находит самое похожее предложение в тексте"""
        if not self.model:
            logger.error("Model is not initialized.")
            return None

        # Извлекаем компоненты из нашего пакета
        vectorizer = self.model["vectorizer"]
        embeddings = self.model["embeddings"]
        sentences = self.model["sentences"]

        # 1. Превращаем запрос в вектор
        query_vec = vectorizer.transform([query])

        # 2. Считаем косинусное сходство
        similarities = cosine_similarity(query_vec, embeddings)
        
        # 3. Поиск лучшего результата
        best_idx = np.argmax(similarities)
        best_score = similarities[0][best_idx]

        if best_score > 0:
            result = sentences[best_idx]
            logger.info(f"Best match ({best_score:.4f}): {result}")
            return {"score": best_score, "text": result}
        
        logger.warning("No relevant matches found.")
        return None
