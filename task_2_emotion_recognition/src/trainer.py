"""
Trainer module for Task 2: Emotion Recognition from Speech.
Coordinates model training fitting cycles and handles serialized model persistence.
"""
import os
import joblib
from src.config import Config

class Trainer:
    """
    Manages the lifecycle of model training and serialization.
    """
    def __init__(self, config=Config):
        self.config = config

    def train_model(self, pipeline, X_train, y_train) -> None:
        """
        Fits the classification pipeline on raw feature matrices.
        """
        pipeline.fit(X_train, y_train)
        print(f"[Trainer] Successfully fitted pipeline: {pipeline}")

    def save_model(self, pipeline, filepath: str = Config.BEST_MODEL_PATH) -> None:
        """
        Serializes the trained pipeline checkpoint to disk.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(pipeline, filepath)
        print(f"[Trainer] Saved model pipeline checkpoint to: {filepath}")

    def save_training_metadata(self, metadata: dict, filepath: str) -> None:
        """
        Saves training run metadata to disk in JSON format.
        """
        import json
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        print(f"[Trainer] Saved training metadata to: {filepath}")

    def load_model(self, filepath: str = Config.BEST_MODEL_PATH):
        """
        Loads and returns a serialized pipeline checkpoint.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at: {filepath}")
        pipeline = joblib.load(filepath)
        print(f"[Trainer] Loaded model pipeline from: {filepath}")
        return pipeline
