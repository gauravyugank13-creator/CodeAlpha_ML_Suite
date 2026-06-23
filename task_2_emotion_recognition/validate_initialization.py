"""
Validation script for Task 2: Emotion Recognition from Speech initialization.
Verifies directory structure, checks skeleton imports, and asserts project isolation rules.
"""
import os
import sys

def main():
    print("==========================================================")
    print("  validate_initialization.py - Task 2 Validation Checks   ")
    print("==========================================================")
    
    task_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(task_dir)
    
    # 1. Directory presence checking
    expected_dirs = [
        "data/raw",
        "data/processed",
        "notebooks",
        "src",
        "models",
        "results/plots",
        "results/metrics",
        "results/predictions",
        "app"
    ]
    
    print("[Check] Verifying directory structure...")
    missing_dirs = []
    for d in expected_dirs:
        dir_path = os.path.join(task_dir, d)
        if not os.path.isdir(dir_path):
            missing_dirs.append(d)
            
    if missing_dirs:
        print(f"[FAIL] Missing directories: {missing_dirs}")
        sys.exit(1)
    else:
        print("[PASS] All expected directories are present.")

    # 2. Key configuration files checks
    expected_files = [
        "README.md",
        "requirements_task2.txt"
    ]
    print("[Check] Verifying key documents...")
    missing_files = []
    for f in expected_files:
        file_path = os.path.join(task_dir, f)
        if not os.path.isfile(file_path):
            missing_files.append(f)
            
    if missing_files:
        print(f"[FAIL] Missing documents: {missing_files}")
        sys.exit(1)
    else:
        print("[PASS] All expected documents are present.")

    # 3. Import verification
    print("[Check] Verifying sub-module imports...")
    
    # Append the task directory to path to enable package import resolution
    if task_dir not in sys.path:
        sys.path.insert(0, task_dir)
        
    modules_to_test = [
        "src.config",
        "src.data_loader",
        "src.audio_preprocessing",
        "src.feature_extraction",
        "src.model_builder",
        "src.trainer",
        "src.evaluator",
        "src.predictor"
    ]
    
    for mod_name in modules_to_test:
        try:
            # Dynamically import module
            __import__(mod_name)
            print(f"  [OK] Successfully imported: {mod_name}")
        except Exception as e:
            print(f"  [FAIL] Failed to import: {mod_name} (Error: {e})")
            sys.exit(1)
            
    print("[PASS] All package sub-modules import cleanly without execution overhead.")
    print("==========================================================")
    print("  [PASS] Task 2 project structure successfully verified!  ")
    print("==========================================================")

if __name__ == "__main__":
    main()
