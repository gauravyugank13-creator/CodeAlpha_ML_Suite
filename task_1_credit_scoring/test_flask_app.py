import os
import sys
import unittest
import pandas as pd
import json

# Add current and parent directories to PYTHONPATH to resolve dependencies
task_dir = os.path.dirname(os.path.abspath(__file__))
if task_dir not in sys.path:
    sys.path.insert(0, task_dir)

from app.app import app, predictor, Config

class TestCreditScoringApp(unittest.TestCase):
    
    def setUp(self):
        # Configure app for testing
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        self.client = app.test_client()
        
        # Test input records
        self.valid_payload = {
            "Age": "35",
            "Sex": "male",
            "Job": "2",
            "Housing": "own",
            "Saving accounts": "little",
            "Checking account": "moderate",
            "Credit amount": "2500",
            "Duration": "18",
            "Purpose": "car"
        }
        
        self.invalid_payload_age = {
            "Age": "10",  # Out of range (min 18)
            "Sex": "male",
            "Job": "2",
            "Housing": "own",
            "Saving accounts": "little",
            "Checking account": "moderate",
            "Credit amount": "2500",
            "Duration": "18",
            "Purpose": "car"
        }
        
        self.invalid_payload_missing = {
            "Age": "35",
            "Sex": "male",
            "Job": "",  # Missing field
            "Housing": "own",
            "Saving accounts": "little",
            "Checking account": "moderate",
            "Credit amount": "2500",
            "Duration": "18",
            "Purpose": "car"
        }

    def test_home_page_loads(self):
        """
        Verify GET / loads the homepage successfully and includes the form wrapper.
        """
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html_content = response.data.decode("utf-8")
        self.assertIn("Credit Scoring Assessment Portal", html_content)
        self.assertIn("id=\"credit-form\"", html_content)

    def test_model_loads_correctly(self):
        """
        Verify that the machine learning predictor and preprocessing pipelines load.
        """
        self.assertTrue(predictor.is_loaded)
        self.assertIsNotNone(predictor.model)
        self.assertIsNotNone(predictor.preprocessor)

    def test_valid_prediction(self):
        """
        Verify that a valid form POST performs inference and displays the result.
        """
        response = self.client.post("/predict", data=self.valid_payload)
        self.assertEqual(response.status_code, 200)
        html_content = response.data.decode("utf-8")
        self.assertIn("Assessment Results", html_content)
        self.assertIn("Model Confidence", html_content)
        # Should contain one of the recommendations
        self.assertTrue(
            "LOW CREDIT RISK" in html_content or "HIGH CREDIT RISK" in html_content,
            "Recommendation not found in result layout."
        )

    def test_invalid_input_validation(self):
        """
        Verify that out-of-bounds continuous parameters are rejected by server-side checks.
        """
        response = self.client.post("/predict", data=self.invalid_payload_age)
        self.assertEqual(response.status_code, 200)
        html_content = response.data.decode("utf-8")
        self.assertIn("Validation Error: Age must be between 18 and 100", html_content)

    def test_missing_input_validation(self):
        """
        Verify that empty string payloads are caught and output friendly warnings.
        """
        response = self.client.post("/predict", data=self.invalid_payload_missing)
        self.assertEqual(response.status_code, 200)
        html_content = response.data.decode("utf-8")
        self.assertIn("Validation Error: Missing values in required fields", html_content)

    def test_history_logging_works(self):
        """
        Verify that a new prediction appends a record in prediction_history.csv
        and creates a unique prediction JSON file.
        """
        history_csv_path = os.path.join(Config.PREDICTIONS_DIR, "prediction_history.csv")
        
        # Count existing CSV rows if file exists
        initial_count = 0
        if os.path.exists(history_csv_path):
            initial_count = len(pd.read_csv(history_csv_path))
            
        # Count JSON files
        initial_jsons = [f for f in os.listdir(Config.PREDICTIONS_DIR) if f.endswith(".json")]
        
        # Run prediction
        response = self.client.post("/predict", data=self.valid_payload)
        self.assertEqual(response.status_code, 200)
        
        # Verify CSV row appended
        self.assertTrue(os.path.exists(history_csv_path))
        final_count = len(pd.read_csv(history_csv_path))
        self.assertEqual(final_count, initial_count + 1)
        
        # Verify a new JSON prediction file is created
        final_jsons = [f for f in os.listdir(Config.PREDICTIONS_DIR) if f.endswith(".json")]
        self.assertEqual(len(final_jsons), len(initial_jsons) + 1)
        
        # Verify the content of the new JSON file
        new_files = list(set(final_jsons) - set(initial_jsons))
        self.assertEqual(len(new_files), 1)
        
        new_json_path = os.path.join(Config.PREDICTIONS_DIR, new_files[0])
        with open(new_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        self.assertIn("timestamp", data)
        self.assertEqual(data["inputs"]["Age"], 35)
        self.assertEqual(data["inputs"]["Sex"], "male")
        self.assertTrue(data["prediction"] in ["GOOD", "BAD"])
        self.assertIn("probabilities", data)

if __name__ == "__main__":
    unittest.main()
