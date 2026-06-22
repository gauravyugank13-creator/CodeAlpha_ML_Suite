"""
predict_digit.py
Task 3 — Handwritten Character Recognition

Usage:
    python predict_digit.py path/to/image.png
"""

import os
import sys

# Ensure task directory is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from src.config import Config
from src.predictor import Predictor


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_digit.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = os.path.join(Config.MODELS_DIR, "best_mnist_model.keras")

    if not os.path.exists(model_path):
        print(f"Error: Trained model not found at '{model_path}'. Please run training first.")
        sys.exit(1)

    if not os.path.exists(image_path):
        print(f"Error: Input image not found at '{image_path}'.")
        sys.exit(1)

    try:
        # Load the model
        model = Predictor.load_model(model_path)

        # Run prediction
        digit, confidence, _ = Predictor.predict_image(model, image_path)

        # Format output as requested
        print(f"Predicted Digit: {digit}")
        print(f"Confidence: {confidence * 100:.2f}%")

    except Exception as e:
        print(f"Error executing prediction: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
