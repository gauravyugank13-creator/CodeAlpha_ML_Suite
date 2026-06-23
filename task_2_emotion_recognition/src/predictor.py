"""
Inference and serving predictor module for Task 2: Speech Emotion Recognition.
Loads serialized winning model and processes audio files to generate predictions,
handles errors gracefully, and registers prediction history logs.
"""
import os
import sys
import numpy as np
import joblib
import pandas as pd
from datetime import datetime
import soundfile as sf

# Resolve paths to allow package imports
task_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if task_dir not in sys.path:
    sys.path.insert(0, task_dir)

from src.config import Config
from src.data_loader import AudioDataLoader
from src.audio_preprocessing import trim_silence, normalize_amplitude, pad_or_truncate
from src.feature_extraction import extract_combined_features

class EmotionPredictor:
    """
    Serves as the prediction service for Speech Emotion Recognition.
    """
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = os.path.join(Config.MODEL_DIR, "final_emotion_model.joblib")
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        self.loader = AudioDataLoader()

    def load_pipeline(self) -> None:
        """
        Loads and validates the serialized classification pipeline model checkpoint.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint file not found at: {self.model_path}")
        try:
            self.model = joblib.load(self.model_path)
            self.is_loaded = True
            print(f"[Predictor] Loaded model pipeline: {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to deserialize model checkpoint. Error: {e}")

    def predict_emotion(self, file_path: str) -> dict:
        """
        Ingests a WAV file path, runs preprocessing and feature extraction, 
        and returns the classification target, confidence score, and probability vector.
        Logs details to results/predictions/prediction_history.csv.
        """
        # 1. Validate model loading
        if not self.is_loaded:
            self.load_pipeline()

        # 2. Robust validation of the input file
        # Check existence
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        # Check format/extension (case-insensitive check for .wav)
        if not file_path.lower().endswith(".wav"):
            raise ValueError(f"Unsupported file format: {os.path.basename(file_path)}. Only WAV format is supported.")

        # Check size & corruption via opening file
        if os.path.getsize(file_path) == 0:
            raise ValueError("Empty audio file (contains zero bytes).")

        try:
            info = sf.info(file_path)
            if info.frames == 0 or info.duration == 0:
                raise ValueError("Empty audio file (contains zero audio frames).")
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            raise RuntimeError(f"Corrupted audio file structure. Failed to read file metadata. Error: {e}")

        # 3. Load waveform and sample rate
        try:
            waveform, sr = self.loader.load_audio_waveform(file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to read audio waveform stream. Error: {e}")

        if waveform is None or len(waveform) == 0:
            raise ValueError("Audio clip waveform contains no samples.")

        # 4. Preprocessing pipeline
        try:
            trimmed = trim_silence(waveform)
            # Check if trimmed waveform is empty (e.g. contained only absolute silence)
            if len(trimmed) == 0:
                trimmed = waveform  # Fall back to untrimmed instead of failing
            
            normalized = normalize_amplitude(trimmed)
            processed = pad_or_truncate(normalized, target_sr=sr)
        except Exception as e:
            raise RuntimeError(f"Inference preprocessing pipeline failed. Error: {e}")

        # 5. Extract features
        try:
            features = extract_combined_features(processed, sr)
            if features is None or len(features) != 374:
                raise ValueError(f"Feature vector dimensionality mismatch: expected 374, got {len(features) if features is not None else 0}")
            features = features.reshape(1, -1)
        except Exception as e:
            raise RuntimeError(f"Feature extraction failed. Error: {e}")

        # 6. Perform inference
        try:
            predicted_emotion = str(self.model.predict(features)[0])
            probs = self.model.predict_proba(features)[0]
            classes = self.model.classes_
            
            probabilities_dict = {str(cls): float(prob) for cls, prob in zip(classes, probs)}
            confidence_score = float(np.max(probs))
        except Exception as e:
            raise RuntimeError(f"Model classification inference failed. Error: {e}")

        # 7. Write to transaction logs
        self._log_prediction(file_path, predicted_emotion, confidence_score)

        return {
            "predicted_emotion": predicted_emotion,
            "confidence_score": confidence_score,
            "probabilities": probabilities_dict
        }

    def _log_prediction(self, file_path: str, emotion: str, confidence: float) -> None:
        """
        Appends prediction transaction log to CSV registry in results/predictions/.
        """
        predictions_dir = os.path.join(Config.RESULTS_DIR, "predictions")
        os.makedirs(predictions_dir, exist_ok=True)
        log_file = os.path.join(predictions_dir, "prediction_history.csv")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename = os.path.basename(file_path)
        
        log_entry = pd.DataFrame([{
            "timestamp": timestamp,
            "filename": filename,
            "predicted_emotion": emotion,
            "confidence_score": confidence
        }])

        try:
            if not os.path.exists(log_file):
                log_entry.to_csv(log_file, index=False)
            else:
                log_entry.to_csv(log_file, mode="a", header=False, index=False)
        except Exception as e:
            print(f"[Warning] Failed to write prediction log row. Error: {e}", file=sys.stderr)

    # Legacy support methods
    def predict_waveform(self, waveform: np.ndarray, sr: int = Config.SAMPLE_RATE) -> str:
        if not self.is_loaded:
            self.load_pipeline()
        trimmed = trim_silence(waveform)
        normalized = normalize_amplitude(trimmed)
        fixed_length = pad_or_truncate(normalized, target_sr=sr)
        features = extract_combined_features(fixed_length, sr).reshape(1, -1)
        return str(self.model.predict(features)[0])

    def predict_file(self, file_path: str) -> str:
        res = self.predict_emotion(file_path)
        return res["predicted_emotion"]

    def predict_proba_file(self, file_path: str) -> dict:
        res = self.predict_emotion(file_path)
        return res["probabilities"]
