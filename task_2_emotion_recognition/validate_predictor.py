"""
Verification and testing script for Task 2 Emotion serving predictor.
Asserts schema correctness, confidence limits, logging transaction, and exception boundaries.
"""
import os
import sys
import numpy as np
import pandas as pd

# Resolve paths to allow package imports
task_dir = os.path.dirname(os.path.abspath(__file__))
if task_dir not in sys.path:
    sys.path.insert(0, task_dir)

from src.config import Config
from src.predictor import EmotionPredictor
from src.data_loader import AudioDataLoader

def main():
    print("==========================================================")
    print("  validate_predictor.py - serving Inference verification  ")
    print("==========================================================")

    # 1. Instantiating predictor
    print("[Testing] Initializing EmotionPredictor service...")
    predictor = EmotionPredictor()
    
    # Assert model loading does not crash
    try:
        predictor.load_pipeline()
    except Exception as e:
        print(f"[FAIL] Predictor failed to load model checkpoint. Error: {e}")
        sys.exit(1)
    
    # Assert load status
    assert predictor.is_loaded, "Assertion Error: predictor status is_loaded should be True."
    print("[PASS] Predictor loaded serialized pipeline cleanly.")

    # 2. Scanning for a real audio sample to verify inference
    print("[Testing] Scanning for a valid audio sample...")
    loader = AudioDataLoader()
    catalog = loader.scan_audio_files()
    if len(catalog) == 0:
        print("[FAIL] No RAVDESS WAV files cataloged in data/raw/. Please check dataset ingestion.")
        sys.exit(1)
        
    sample_item = catalog[0]
    sample_file_path = sample_item["file_path"]
    print(f"  - Selected sample file: {os.path.basename(sample_file_path)}")

    # 3. Verify prediction output schema and values
    print("[Testing] Running inference and checking schema bounds...")
    try:
        result = predictor.predict_emotion(sample_file_path)
    except Exception as e:
        print(f"[FAIL] Inference execution failed. Error: {e}")
        sys.exit(1)
        
    print(f"  - Prediction Results: {result}")

    # Schema keys check
    for key in ["predicted_emotion", "confidence_score", "probabilities"]:
        assert key in result, f"Assertion Error: Output dictionary missing key '{key}'"
    print("[PASS] Output dictionary schema is correct.")

    # Confidence range validation
    conf = result["confidence_score"]
    assert 0.0 <= conf <= 1.0, f"Assertion Error: Confidence score {conf} must be in [0.0, 1.0]"
    print(f"[PASS] Confidence score is within valid limits: {conf:.4f}")

    # Probability vector properties
    probs = result["probabilities"]
    prob_sum = sum(probs.values())
    assert np.isclose(prob_sum, 1.0, atol=1e-2), f"Assertion Error: Probabilities sum is {prob_sum}, expected approx 1.0"
    print(f"[PASS] Probabilities sum to approximately 1.0: {prob_sum:.4f}")

    # Predicted class validity
    pred_emo = result["predicted_emotion"]
    assert pred_emo in Config.EMOTIONS, f"Assertion Error: Predicted class '{pred_emo}' not in target emotions: {Config.EMOTIONS}"
    print(f"[PASS] Predicted emotion class is valid: '{pred_emo}'")

    # 4. Check Logging transaction
    print("[Testing] Verifying prediction transaction logging...")
    log_file = os.path.join(Config.RESULTS_DIR, "predictions", "prediction_history.csv")
    assert os.path.exists(log_file), f"Assertion Error: Log registry does not exist at: {log_file}"
    
    # Load logs and assert last row matches
    logs_df = pd.read_csv(log_file)
    last_row = logs_df.iloc[-1]
    
    assert last_row["filename"] == os.path.basename(sample_file_path), "Assertion Error: Logged filename mismatch."
    assert last_row["predicted_emotion"] == pred_emo, "Assertion Error: Logged emotion mismatch."
    assert np.isclose(last_row["confidence_score"], conf), "Assertion Error: Logged confidence mismatch."
    print("[PASS] Prediction transaction logged successfully to prediction_history.csv.")

    # 5. Check Exception boundaries
    print("[Testing] Testing robust boundary check exceptions...")
    
    # Case A: Missing file
    missing_path = os.path.join(Config.DATA_DIR, "missing_sample.wav")
    try:
        predictor.predict_emotion(missing_path)
        print("[FAIL] Missing file check: did not raise FileNotFoundError.")
        sys.exit(1)
    except FileNotFoundError:
        print("  - Case A (Missing file): Raised FileNotFoundError successfully.")
    except Exception as e:
        print(f"[FAIL] Case A raised incorrect exception: {e.__class__.__name__}")
        sys.exit(1)

    # Case B: Unsupported format extension
    bad_extension_path = os.path.join(Config.DATA_DIR, "invalid_format.txt")
    with open(bad_extension_path, "w", encoding="utf-8") as f:
        f.write("modality-vocal-emotion-intensity-statement-repetition-actor")
        
    try:
        predictor.predict_emotion(bad_extension_path)
        print("[FAIL] Bad extension check: did not raise ValueError.")
        os.remove(bad_extension_path)
        sys.exit(1)
    except ValueError as e:
        print(f"  - Case B (Unsupported format): Raised ValueError successfully. Message: {e}")
    except Exception as e:
        print(f"[FAIL] Case B raised incorrect exception: {e.__class__.__name__}")
        os.remove(bad_extension_path)
        sys.exit(1)
    os.remove(bad_extension_path)

    # Case C: Empty WAV file (0-byte file)
    empty_wav_path = os.path.join(Config.DATA_DIR, "empty_clip.wav")
    with open(empty_wav_path, "wb") as f:
        pass
        
    try:
        predictor.predict_emotion(empty_wav_path)
        print("[FAIL] Empty WAV check: did not raise ValueError.")
        os.remove(empty_wav_path)
        sys.exit(1)
    except ValueError as e:
        print(f"  - Case C (Empty WAV): Raised ValueError successfully. Message: {e}")
    except Exception as e:
        print(f"[FAIL] Case C raised incorrect exception: {e.__class__.__name__} ({e})")
        os.remove(empty_wav_path)
        sys.exit(1)
    os.remove(empty_wav_path)

    # Case D: Corrupted WAV file
    corrupted_wav_path = os.path.join(Config.DATA_DIR, "corrupted.wav")
    with open(corrupted_wav_path, "w", encoding="utf-8") as f:
        f.write("RIFFxxxxWAVEfmt xxxx data xxxx text stuff not audio bytes format")
        
    try:
        predictor.predict_emotion(corrupted_wav_path)
        print("[FAIL] Corrupted WAV check: did not raise exception.")
        os.remove(corrupted_wav_path)
        sys.exit(1)
    except (RuntimeError, ValueError) as e:
        print(f"  - Case D (Corrupted WAV): Raised expected exception successfully. Message: {e}")
    except Exception as e:
        print(f"[FAIL] Case D raised incorrect exception: {e.__class__.__name__} ({e})")
        os.remove(corrupted_wav_path)
        sys.exit(1)
    os.remove(corrupted_wav_path)

    print("==========================================================")
    print("  [PASS] serving Inference validator audits complete!     ")
    print("==========================================================")

if __name__ == "__main__":
    main()
