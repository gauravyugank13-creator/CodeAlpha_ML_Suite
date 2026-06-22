"""
test_flask_app.py
Task 3 — Handwritten Character Recognition

Purpose:
    Unit testing suite for the Flask web application.
    Verifies that:
        1. The homepage loads successfully (GET /)
        2. The trained Keras model loads correctly at startup
        3. The predict API (POST /predict) successfully classifies a digit
        4. File validations function correctly (invalid types, empty file, size limit)
"""

import io
import os
import sys
import unittest

# Ensure the task directory is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from app.app import app, model
from src.config import Config


class TestFlaskApp(unittest.TestCase):
    """
    Test suite for Flask handwritten digit classifier.
    """

    def setUp(self):
        """Configure test client."""
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_homepage_loads(self):
        """Verify the index page renders with 200 OK and contains key brand elements."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"DigitCognition", response.data)
        self.assertIn(b"Choose Input Method", response.data)
        self.assertIn(b"Inference Output", response.data)

    def test_model_is_loaded(self):
        """Verify the global CNN model has successfully loaded at application startup."""
        self.assertIsNotNone(model, "Global model variable in app.py should not be None")

    def test_prediction_endpoint(self):
        """Verify prediction endpoint with a valid custom image."""
        # Use one of the generated test PNGs
        test_img_path = os.path.join(Config.PREDICTIONS_DIR, "sample_digit_1.png")
        self.assertTrue(os.path.exists(test_img_path), f"Test image not found at: {test_img_path}")
        
        with open(test_img_path, "rb") as img_file:
            img_data = img_file.read()
            
        data = {
            "file": (io.BytesIO(img_data), "sample_digit_1.png")
        }
        
        response = self.client.post("/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        
        json_data = response.get_json()
        self.assertIsNotNone(json_data, "Response should be valid JSON")
        self.assertIn("digit", json_data)
        self.assertIn("confidence", json_data)
        self.assertIn("probabilities", json_data)
        
        # Verify result values and bounds
        self.assertEqual(json_data["digit"], 1)
        self.assertGreater(json_data["confidence"], 0.0)
        self.assertLessEqual(json_data["confidence"], 100.0)
        self.assertEqual(len(json_data["probabilities"]), 10)
        # Sum of probabilities should be close to 100%
        self.assertAlmostEqual(sum(json_data["probabilities"]), 100.0, delta=2.0)

    def test_invalid_upload_extension(self):
        """Verify uploading an unsupported format returns 400 and a friendly message."""
        data = {
            "file": (io.BytesIO(b"dummy text content"), "test_script.txt")
        }
        response = self.client.post("/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        json_data = response.get_json()
        self.assertIn("error", json_data)
        self.assertIn("Unsupported file format", json_data["error"])

    def test_empty_upload(self):
        """Verify uploading an empty file returns 400 and a friendly message."""
        data = {
            "file": (io.BytesIO(b""), "empty.png")
        }
        response = self.client.post("/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        json_data = response.get_json()
        self.assertIn("error", json_data)
        self.assertIn("empty", json_data["error"])

    def test_no_file_part(self):
        """Verify request with no file parameter returns 400 and a friendly message."""
        response = self.client.post("/predict", data={}, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        json_data = response.get_json()
        self.assertIn("error", json_data)
        self.assertIn("No file part", json_data["error"])

    def test_oversized_file(self):
        """Verify file exceeding the 2MB limit returns 413 and a custom handled message."""
        # Create a mock file larger than 2MB
        oversized_data = b"0" * (2 * 1024 * 1024 + 100) # > 2MB
        data = {
            "file": (io.BytesIO(oversized_data), "large_image.png")
        }
        response = self.client.post("/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 413)
        json_data = response.get_json()
        self.assertIn("error", json_data)
        self.assertIn("exceeds the 2MB limit", json_data["error"])


if __name__ == "__main__":
    unittest.main()
