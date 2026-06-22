"""
app.py
Task 3 — Handwritten Character Recognition

Flask application providing a web interface and a prediction API for the MNIST digit classifier.
"""

import os
import sys
from flask import Flask, render_template, request, jsonify

# Ensure the root of task_3 is in sys.path so we can import src
APP_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(APP_DIR)
if TASK_DIR not in sys.path:
    sys.path.insert(0, TASK_DIR)

from src.config import Config
from src.predictor import Predictor
from src.image_utils import preprocess_image_bytes

app = Flask(__name__)

# Max upload size: 2MB
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Global model variable to load once on startup
MODEL_PATH = os.path.join(Config.MODELS_DIR, "best_mnist_model.keras")
model = None

try:
    print(f"[Flask] Loading model from: {MODEL_PATH}")
    model = Predictor.load_model(MODEL_PATH)
except Exception as e:
    print(f"[Flask Critical Error] Model loading failed: {e}")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def home():
    """Render the homepage UI."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST endpoint to predict the digit from an uploaded image.
    Expects a multipart/form-data upload with a 'file' parameter.
    """
    if model is None:
        return jsonify({"error": "Machine learning model is not loaded on server."}), 500

    # 1. Validation
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request payload."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file was selected for upload."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file format. Supported extensions: PNG, JPG, JPEG."}), 400

    try:
        # Read the file stream directly in memory to avoid writing to disk
        file_bytes = file.read()
        if len(file_bytes) == 0:
            return jsonify({"error": "Uploaded file is empty."}), 400

        # Preprocess and execute prediction
        img_array = preprocess_image_bytes(file_bytes)
        predicted_digit, confidence, probabilities = Predictor.predict_array(model, img_array)

        # Return JSON structure as requested:
        # {
        #     "digit": X,
        #     "confidence": XX.XX,
        #     "probabilities": [...]
        # }
        return jsonify({
            "digit": predicted_digit,
            "confidence": round(confidence * 100, 2),
            "probabilities": [round(float(p) * 100, 2) for p in probabilities]
        })

    except ValueError as val_err:
        return jsonify({"error": str(val_err)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal prediction failure: {str(e)}"}), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle files exceeding MAX_CONTENT_LENGTH."""
    return jsonify({"error": "File size exceeds the 2MB limit."}), 413


if __name__ == "__main__":
    # Local development server
    app.run(host="127.0.0.1", port=5000, debug=True)
