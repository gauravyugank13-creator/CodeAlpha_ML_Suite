"""
Unit testing suite for Task 2: Speech Emotion Recognition Flask Web Application.
Tests routing, form rendering, extension checking, missing payload rejection, and transaction logs.
"""
import os
import sys
import unittest
from io import BytesIO
import pandas as pd

# Resolve paths to allow package imports
task_dir = os.path.dirname(os.path.abspath(__file__))
if task_dir not in sys.path:
    sys.path.insert(0, task_dir)

from app.app import app
from src.config import Config
from src.data_loader import AudioDataLoader

class FlaskAppTestCase(unittest.TestCase):
    """
    Test suite for Flask serving layer.
    """
    def setUp(self):
        # Configure app for testing
        app.config["TESTING"] = True
        self.client = app.test_client()
        
        # Load a valid audio path for testing
        loader = AudioDataLoader()
        catalog = loader.scan_audio_files()
        self.sample_wav_path = None
        if len(catalog) > 0:
            self.sample_wav_path = catalog[0]["file_path"]

    def test_homepage_loads(self):
        """
        Verify that GET / returns status 200.
        """
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_upload_form_renders(self):
        """
        Verify that homepage contains upload form elements.
        """
        response = self.client.get("/")
        content = response.data.decode("utf-8")
        self.assertIn("Analyze Vocal Emotion", content)
        self.assertIn('action="/predict"', content)
        self.assertIn('type="file"', content)

    def test_invalid_extension_rejected(self):
        """
        Verify uploading a non-WAV file (e.g. .txt) redirects to home with error.
        """
        data = {
            "file": (BytesIO(b"dummy text content"), "invalid_format.txt")
        }
        response = self.client.post("/predict", data=data, content_type="multipart/form-data")
        # Flask redirect is expected to index
        self.assertEqual(response.status_code, 302)
        self.assertIn("/", response.location)
        
        # Follow redirect and check warning message
        follow_response = self.client.get(response.location)
        follow_content = follow_response.data.decode("utf-8")
        self.assertIn("Unsupported file format", follow_content)

    def test_missing_upload_file_rejected(self):
        """
        Verify posting empty fields redirects to home with error.
        """
        # Empty filename
        data = {
            "file": (BytesIO(b""), "")
        }
        response = self.client.post("/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 302)
        
        follow_response = self.client.get(response.location)
        follow_content = follow_response.data.decode("utf-8")
        self.assertIn("No file selected", follow_content)

    def test_missing_upload_parameter_rejected(self):
        """
        Verify posting request without 'file' key redirects with error.
        """
        response = self.client.post("/predict", data={}, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 302)
        
        follow_response = self.client.get(response.location)
        follow_content = follow_response.data.decode("utf-8")
        self.assertIn("No file upload parameter found", follow_content)

    def test_valid_wav_prediction_and_logging(self):
        """
        Verify a valid WAV file processes correctly, renders results page, and appends to CSV log.
        """
        if self.sample_wav_path is None or not os.path.exists(self.sample_wav_path):
            self.skipTest("No RAVDESS audio sample files available in raw/ to test valid WAV prediction.")

        # Read actual WAV file bytes
        with open(self.sample_wav_path, "rb") as f:
            wav_bytes = f.read()

        filename = os.path.basename(self.sample_wav_path)
        data = {
            "file": (BytesIO(wav_bytes), filename)
        }
        
        # Track baseline log rows
        log_file = os.path.join(Config.RESULTS_DIR, "predictions", "prediction_history.csv")
        initial_log_rows = 0
        if os.path.exists(log_file):
            initial_log_rows = len(pd.read_csv(log_file))

        # Run POST predict request
        response = self.client.post("/predict", data=data, content_type="multipart/form-data")
        
        # Assert render success
        self.assertEqual(response.status_code, 200)
        content = response.data.decode("utf-8")
        self.assertIn("Predicted Emotion", content)
        self.assertIn("Confidence Rating", content)
        self.assertIn("Class Probabilities Breakdown", content)
        self.assertIn(filename, content)

        # Assert log file gets updated
        self.assertTrue(os.path.exists(log_file))
        new_log_rows = len(pd.read_csv(log_file))
        self.assertEqual(new_log_rows, initial_log_rows + 1)
        
        # Read last log row
        logs_df = pd.read_csv(log_file)
        last_row = logs_df.iloc[-1]
        self.assertEqual(last_row["filename"], filename)
        self.assertIsNotNone(last_row["predicted_emotion"])
        self.assertGreaterEqual(last_row["confidence_score"], 0.0)
        self.assertLessEqual(last_row["confidence_score"], 1.0)

if __name__ == "__main__":
    unittest.main()
