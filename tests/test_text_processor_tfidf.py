import sys
import os
import shutil
import tempfile
import unittest

from src.text_processor_tfidf import TextProcessorTfidf
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestTextProcessorTfidf(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up sample data for all tests."""
        cls.test_text = (
            "Python is great. Machine learning is fun. Beer Sheva is a nice city."
        )
        cls.test_query = "Tell me about machine learning"

    def setUp(self):
        """Create a temporary directory for file-based tests."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up the temporary directory after each test."""
        shutil.rmtree(self.test_dir)

    def test_from_text_initialization(self):
        """Test if factory method from_text creates a valid instance with model data."""
        processor = TextProcessorTfidf.from_text(self.test_text)
        self.assertIsNotNone(processor.model)
        self.assertIn("vectorizer", processor.model)
        self.assertEqual(len(processor.model["sentences"]), 3)

    def test_predict_functionality(self):
        """Test if predict returns the correct sentence and a valid score."""
        processor = TextProcessorTfidf.from_text(self.test_text)
        result = processor.predict(self.test_query)

        self.assertIsNotNone(result)
        self.assertIn("Machine learning is fun", result["text"])
        self.assertGreater(result["score"], 0)

    def test_predict_no_match(self):
        """Test that predict returns None when no relevant matches are found."""
        processor = TextProcessorTfidf.from_text(self.test_text)
        # Using tokens that are not present in the original sentences
        result = processor.predict("xyz abc unknown")
        self.assertIsNone(result)

    def test_save_and_load_lifecycle(self):
        """Test the complete train-save-load-predict lifecycle."""
        model_path = os.path.join(self.test_dir, "model.joblib")

        # 1. Train and save
        original_processor = TextProcessorTfidf.from_text(self.test_text)
        original_processor.save_model(model_path)

        # 2. Load from the created file using the factory method
        loaded_processor = TextProcessorTfidf.from_file(model_path)

        # 3. Verify the loaded model works correctly
        self.assertIsNotNone(loaded_processor.model)
        result = loaded_processor.predict(self.test_query)
        self.assertIn("Machine learning is fun", result["text"])

    def test_empty_text_error(self):
        """Test that an empty string raises ValueError in the factory method."""
        with self.assertRaises(ValueError):
            TextProcessorTfidf.from_text("   ")

    def test_file_not_found_error(self):
        """Test that from_file raises FileNotFoundError for missing paths."""
        with self.assertRaises(FileNotFoundError):
            TextProcessorTfidf.from_file("non_existent_file.joblib")

    def test_save_uninitialized_model_error(self):
        """Test that saving a model without training/loading raises RuntimeError."""
        processor = TextProcessorTfidf()  # Direct instantiation without factory
        with self.assertRaises(RuntimeError):
            processor.save_model(os.path.join(self.test_dir, "fail.joblib"))


if __name__ == "__main__":
    unittest.main()
