"""
Validation and verification runner for Task 2 Flask Serving application.
Checks imports, route mappings, size configurations, and model predictor load compatibility.
"""
import os
import sys

# Resolve paths to allow package imports
task_dir = os.path.dirname(os.path.abspath(__file__))
if task_dir not in sys.path:
    sys.path.insert(0, task_dir)

def run_flask_validation():
    print("==========================================================")
    print("  validate_flask_app.py - Serving Setup Verification     ")
    print("==========================================================")

    # 1. Test Imports
    print("[Audit] Verifying serving application imports...")
    try:
        from app.app import app, predictor
        from src.config import Config
    except Exception as e:
        print(f"[FAIL] Failed to import Flask app or models modules. Error: {e}")
        sys.exit(1)
    print("[PASS] Flask application and predictors imported cleanly.")

    # 2. Check maximum size content configuration
    print("[Audit] Verifying upload limit configurations...")
    max_size = app.config.get("MAX_CONTENT_LENGTH")
    if max_size is None:
        print("[FAIL] MAX_CONTENT_LENGTH size limit is not configured in Flask app.")
        sys.exit(1)
    
    max_mb = max_size / (1024 * 1024)
    print(f"[PASS] Upload limit configured: {max_mb:.1f} MB ({max_size} bytes).")
    
    # 3. Check endpoint mapping configurations
    print("[Audit] Verifying endpoint route mappings...")
    routes = [str(rule) for rule in app.url_map.iter_rules()]
    expected_routes = ["/", "/predict"]
    
    for route in expected_routes:
        mapped = any(route == r for r in routes)
        if not mapped:
            print(f"[FAIL] Endpoint route '{route}' is not registered on Flask URL map.")
            sys.exit(1)
        print(f"[PASS] Endpoint route '{route}' found on server URL map.")

    # 4. Check Predictor is integrated and initialized
    print("[Audit] Verifying global Predictor integration...")
    if predictor is None:
        print("[FAIL] Predictor instance not loaded or initialized on server.")
        sys.exit(1)
        
    if not predictor.is_loaded:
        print("[FAIL] Predictor loaded status should be True at server startup.")
        sys.exit(1)
        
    final_model_path = os.path.join(Config.MODEL_DIR, "final_emotion_model.joblib")
    if not os.path.samefile(predictor.model_path, final_model_path):
        print(f"[FAIL] Predictor targets wrong model path. Expected {final_model_path}, got {predictor.model_path}")
        sys.exit(1)
        
    print(f"[PASS] Integrated predictor loads from target checkpoint: {os.path.basename(predictor.model_path)}")

    print("==========================================================")
    print("  [PASS] Flask serving application validation successful! ")
    print("==========================================================")

if __name__ == "__main__":
    run_flask_validation();
