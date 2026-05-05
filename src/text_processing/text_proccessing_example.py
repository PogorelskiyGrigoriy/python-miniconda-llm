import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.logger_config import setup_logging

logger = setup_logging()

TEXT = """I want to learn machine learning and natural language processing.
Machine learning is fascinating! Natural language processing allows computers to understand human language.
I am excited to explore more about these topics.I learned about supervised and unsupervised learning. I also want to understand how transformers work in NLP.
Also, I am interested in deep learning techniques and how they can be applied to various problems.
I hope to build some cool projects using machine learning and NLP soon!"""

text_arr= TEXT.split('.')
cleaned_text_arr = [sentence.strip() for sentence in text_arr if sentence.strip()]
vectorizer= TfidfVectorizer(stop_words='english')
embeddings= vectorizer.fit_transform(cleaned_text_arr)

query = "What is machine learning?"
embedding_query = vectorizer.transform([query])

similarities = cosine_similarity(embedding_query, embeddings)

best_index = np.argmax(similarities)
best_score = similarities[0][best_index]

if best_score > 0:
    logger.info(f"The best result ({best_score:.4f}): {cleaned_text_arr[best_index]}")
else:
    logger.warning("No relevant sentences found.")