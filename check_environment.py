#!/usr/bin/env python
"""
Environment Verification Script
Attempts to import key machine learning and data science libraries and reports status.
"""

import sys

def check_libraries():
    # Mapping of (User-friendly Name, Import Name)
    libraries = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("scikit-learn", "sklearn"),
        ("tensorflow", "tensorflow"),
        ("flask", "flask"),
        ("opencv-python", "cv2"),
        ("librosa", "librosa"),
        ("joblib", "joblib")
    ]

    print("=" * 60)
    print("         MACHINE LEARNING SUITE ENVIRONMENT CHECK          ")
    print("=" * 60)
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"Executable:     {sys.executable}")
    print("-" * 60)

    all_passed = True
    failed_libs = []

    for name, import_name in libraries:
        try:
            # Dynamically import the library
            module = __import__(import_name)
            
            # Get version if available
            version = "Unknown"
            if hasattr(module, "__version__"):
                version = module.__version__
            elif hasattr(module, "__git_version__"):
                version = module.__git_version__
            
            print(f"[ PASS ] {name:<18} | Version: {version}")
        except ImportError as e:
            print(f"[ FAIL ] {name:<18} | Error: {str(e)}")
            all_passed = False
            failed_libs.append(name)

    print("-" * 60)
    if all_passed:
        print(" SUCCESS: All libraries are successfully installed and ready!")
        print(" You can safely proceed to building the models.")
    else:
        print(" WARNING: Some libraries failed to import.")
        print(f" Missing or broken packages: {', '.join(failed_libs)}")
        print(" Make sure you have activated your virtual environment and run:")
        print("   pip install -r requirements.txt")
    print("=" * 60)

    # Return exit code: 0 if all pass, 1 if any fails
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(check_libraries())
