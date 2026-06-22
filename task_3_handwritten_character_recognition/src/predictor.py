"""
Task 3 — Handwritten Character Recognition
src/predictor.py

Purpose:
    Provides inference capabilities for the trained CNN model.
    Loads the trained model, preprocesses inputs, and outputs predicted digits
    with confidence values.
"""

import os
import numpy as np


class Predictor:
    """
    Runs inference on a trained MNIST CNN model.
    """

    @staticmethod
    def load_model(model_path: str):
        """
        Load a saved Keras model from disk.

        Args:
            model_path: Path to the .keras model file.

        Returns:
            tf.keras.Model: The loaded Keras model.

        Raises:
            FileNotFoundError: If the model file is not found.
            Exception: If model loading fails.
        """
        try:
            import tensorflow as tf
        except ImportError as e:
            raise ImportError(
                "TensorFlow is not installed. Run: pip install tensorflow"
            ) from e

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at path: {model_path}")

        try:
            model = tf.keras.models.load_model(model_path)
            print(f"[Predictor] Model loaded successfully from: {model_path}")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}. Error: {e}") from e

    @staticmethod
    def predict_image(model, image_path: str):
        """
        Predict the digit class for a single image file path.

        Args:
            model:      Loaded tf.keras.Model instance.
            image_path: Path to the input image file.

        Returns:
            tuple: (predicted_digit, confidence_score, probabilities_list)
        """
        from src.image_utils import preprocess_image
        try:
            img_array = preprocess_image(image_path)
            return Predictor.predict_array(model, img_array)
        except Exception as e:
            print(f"[Predictor] Error processing image {image_path}: {e}")
            raise

    @staticmethod
    def predict_array(model, img_array: np.ndarray):
        """
        Predict the digit class for a preprocessed image array.

        Args:
            model:     Loaded tf.keras.Model instance.
            img_array: Preprocessed image array of shape (1, 28, 28, 1).

        Returns:
            tuple: (predicted_digit, confidence_score, probabilities_list)
        """
        try:
            if img_array.shape != (1, 28, 28, 1):
                raise ValueError(
                    f"Invalid image array shape {img_array.shape}. Expected (1, 28, 28, 1)"
                )

            probabilities = model.predict(img_array, verbose=0)[0]
            predicted_digit = int(np.argmax(probabilities))
            confidence_score = float(probabilities[predicted_digit])
            probabilities_list = [round(float(p), 4) for p in probabilities]

            return predicted_digit, confidence_score, probabilities_list
        except Exception as e:
            print(f"[Predictor] Prediction failure: {e}")
            raise

    @staticmethod
    def predict_with_confidence(model, image_path_or_array):
        """
        Predict the digit class and confidence for either an image path or array.

        Args:
            model:                 Loaded tf.keras.Model instance.
            image_path_or_array:   String path to an image file or a preprocessed numpy array.

        Returns:
            tuple: (predicted_digit, confidence_score, probabilities_list)
        """
        if isinstance(image_path_or_array, (str, os.PathLike)):
            return Predictor.predict_image(model, image_path_or_array)
        elif isinstance(image_path_or_array, np.ndarray):
            return Predictor.predict_array(model, image_path_or_array)
        else:
            raise TypeError(
                "Input must be either a string path to an image or a preprocessed numpy array."
            )


# ---------------------------------------------------------------------------
# TODO: Phase 5 — Deployment
#   - Add save_predictions_to_csv(results, path) to export batch results.
#   - Wrap predict_single() in a Flask route in app/app.py.
# ---------------------------------------------------------------------------
