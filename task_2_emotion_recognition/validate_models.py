"""
Sanity checking and validation script for Task 2 serialized emotion recognition model.
Asserts model files exist, reload cleanly, execute inference, and predict valid classes.
"""
import os
import sys
import numpy as np
import joblib

# Resolve paths to allow package imports
task_dir = os.path.dirname(os.path.abspath(__file__))
if task_dir not in sys.path:
    sys.path.insert(0, task_dir)

from src.config import Config
from src.trainer import Trainer

def run_model_verification():
    print("==========================================================")
    print("  validate_models.py - Serialized Pipeline Audits         ")
    print("==========================================================")

    final_model_path = os.path.join(Config.MODEL_DIR, "final_emotion_model.joblib")
    
    # 1. Check if model file exists on disk
    print("[Audit] Verifying serialized winning model existence...")
    if not os.path.exists(final_model_path):
        print(f"[FAIL] Final winning model checkpoint does not exist at: {final_model_path}")
        sys.exit(1)
    print(f"[PASS] Serialized model found at: {final_model_path}")

    # 2. Reload the model pipeline checkpoint
    print("[Audit] Reloading model pipeline checkpoint from disk...")
    try:
        trainer = Trainer()
        model_pipeline = trainer.load_model(final_model_path)
    except Exception as e:
        print(f"[FAIL] Failed to load final model pipeline from joblib serialization. Error: {e}")
        sys.exit(1)
    print("[PASS] Model pipeline loaded cleanly without namespace conflicts.")

    # 3. Assert correct classifier step properties
    print("[Audit] Validating model pipeline steps...")
    steps = dict(model_pipeline.steps)
    if "scaler" not in steps:
        print("[FAIL] Scaling step 'scaler' missing from pipeline.")
        sys.exit(1)
    print("[PASS] StandardScaler step found in pipeline.")
    
    classifier_step_name = list(steps.keys())[-1]
    classifier_obj = steps[classifier_step_name]
    print(f"[PASS] Model classifier step detected: '{classifier_step_name}' ({classifier_obj.__class__.__name__})")

    # 4. Verify inference execution on a mock sample of shape (1, 374)
    print("[Audit] Running inference test on mock data...")
    try:
        mock_sample = np.random.normal(size=(1, 374))
        prediction = model_pipeline.predict(mock_sample)
        probabilities = model_pipeline.predict_proba(mock_sample)
    except Exception as e:
        print(f"[FAIL] Inference execution failed on dummy 374-element sample. Error: {e}")
        sys.exit(1)
        
    print(f"[PASS] Inference executed successfully.")
    print(f"  - Predicted Emotion: {prediction[0]}")
    print(f"  - Confidence Probabilities Vector Shape: {probabilities.shape}")

    # 5. Check prediction classes belong to valid emotion labels
    print("[Audit] Verifying predicted class belongs to valid emotions list...")
    predicted_emotion = prediction[0]
    if predicted_emotion not in Config.EMOTIONS:
        print(f"[FAIL] Predicted label '{predicted_emotion}' does not belong to valid RAVDESS emotions: {Config.EMOTIONS}")
        sys.exit(1)
    print(f"[PASS] Predicted label is valid.")

    print("==========================================================")
    print("  [PASS] Serialized model validation checks passed!      ")
    print("==========================================================")

if __name__ == "__main__":
    run_model_verification()
