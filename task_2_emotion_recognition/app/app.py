"""
Flask Web Server Application for Task 2: Speech Emotion Recognition.
Provides a browser interface to upload WAV files and visualize predicted emotions and probabilities.
"""
import os
import sys
from flask import Flask, request, render_template, redirect, url_for

# Resolve paths to allow package imports
app_dir = os.path.dirname(os.path.abspath(__file__))
task_dir = os.path.dirname(app_dir)
if task_dir not in sys.path:
    sys.path.insert(0, task_dir)

from src.config import Config
from src.predictor import EmotionPredictor

app = Flask(__name__)

# Enforce file upload size limits (e.g. 5 MB max)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

# Instantiate prediction service once at startup
predictor = EmotionPredictor()
predictor.load_pipeline()

@app.route("/", methods=["GET"])
def index():
    """
    Serves the main dashboard uploader form.
    """
    error = request.args.get("error")
    return render_template("index.html", error=error)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Processes the uploaded audio clip, runs feature extraction & inference,
    registers log transactions, and serves the results dashboard.
    """
    if "file" not in request.files:
        return redirect(url_for("index", error="No file upload parameter found in the request."))

    file = request.files["file"]
    
    # Check if empty file submission
    if file.filename == "":
        return redirect(url_for("index", error="No file selected. Please select a valid WAV audio file."))

    # Validate file extension (.wav only)
    if not file.filename.lower().endswith(".wav"):
        return redirect(url_for("index", error="Unsupported file format. Only WAV format (.wav) is supported."))

    # Create temporary directory inside processed/ to store active upload
    temp_dir = os.path.join(Config.PROCESSED_DATA_DIR, "temp_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    temp_filepath = os.path.join(temp_dir, file.filename)

    try:
        # Save file to temp path
        file.save(temp_filepath)
        
        # Run prediction pipeline (handles size checks and corruption validations internally)
        result = predictor.predict_emotion(temp_filepath)
        
        # Cleanup temp file immediately
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
            
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Map to display percentages
        prob_percentages = {emotion: prob * 100 for emotion, prob in result["probabilities"].items()}

        return render_template(
            "result.html",
            filename=file.filename,
            predicted_emotion=result["predicted_emotion"],
            confidence_score=result["confidence_score"] * 100,
            probabilities=prob_percentages,
            timestamp=timestamp
        )

    except Exception as e:
        # Cleanup temp file on failure
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        
        # Log to stderr and return to homepage with warning message
        print(f"[Error] Prediction serving error: {e}", file=sys.stderr)
        return redirect(url_for("index", error=str(e)))

@app.errorhandler(413)
def request_entity_too_large(error):
    """
    Catches 413 error (uploads > 5MB limit) and redirects to homepage with warning.
    """
    return redirect(url_for("index", error="Upload file size exceeds the maximum limit of 5 MB."))

if __name__ == "__main__":
    # Standard local debug port 5002 to avoid conflicts with other apps (e.g. digit server on 5000)
    app.run(host="127.0.0.1", port=5002, debug=True)
