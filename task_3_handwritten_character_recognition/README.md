# Task 3: Handwritten Character Recognition

> **CodeAlpha ML Internship — Phase 1A**  
> Status: Project Initialized — Ready for MNIST Integration

---

## 🎯 Project Objective

Build a deep learning model that can automatically recognize and classify
handwritten digits (0–9) from grayscale images using:

- **Dataset:** MNIST (60,000 training + 10,000 test images)
- **Architecture:** Convolutional Neural Network (CNN)
- **Framework:** TensorFlow / Keras
- **Target Accuracy:** ≥ 98% on the MNIST test set

---

## 📁 Folder Structure

```
task_3_handwritten_character_recognition/
│
├── README.md                    ← You are here
├── DEPLOYMENT.md                ← Deployment guide (Local, Render, Railway)
├── requirements_task3.txt       ← Task-specific dependencies
├── test_flask_app.py            ← Unit test suite for the Flask web application
├── predict_digit.py             ← Command-line inference script
├── train_model.py               ← Model training script
├── validate_dataset.py          ← Standalone dataset validation script
├── validate_prediction_pipeline.py ← Standalone prediction pipeline verification script
│
├── data/
│   ├── raw/                     ← Downloaded raw MNIST data
│   ├── processed/               ← Preprocessed .npy arrays
│   └── .gitkeep
│
├── notebooks/
│   └── exploration.ipynb        ← EDA, sample visualisation, preprocessing check
│
├── src/                         ← Core Python modules
│   ├── __init__.py              ← Makes src/ a package
│   ├── config.py                ← All settings and paths (start here!)
│   ├── data_loader.py           ← Load MNIST dataset
│   ├── preprocessing.py         ← Normalise, reshape, one-hot encode
│   ├── model_builder.py         ← CNN architecture definition
│   ├── trainer.py               ← Training loop and callbacks
│   ├── evaluator.py             ← Metrics, confusion matrix, loss plots
│   ├── image_utils.py           ← Image decoding and auto-inversion utilities
│   └── predictor.py             ← Inference runner wrappers
│
├── models/                      ← Saved trained model weights (including best_mnist_model.keras)
│   └── .gitkeep
│
├── results/
│   ├── plots/                   ← Training curves, confusion matrix images
│   ├── metrics/                 ← JSON evaluation metrics and classification report
│   ├── predictions/             ← Batch prediction outputs and generated samples
│   └── .gitkeep
│
└── app/                         ← Flask Web Application
    ├── app.py                   ← Flask server with /predict and index routes
    ├── templates/
    │   └── index.html           ← Responsive dark dashboard UI
    └── static/
        ├── styles.css           ← Glassmorphic styling and animations
        └── script.js            ← Canvas drawing & file upload interaction
```

---

## 🔧 Setup Instructions

### Prerequisites

- Python 3.11 (target system environment)
- pip (package manager)
- Virtual environment activated (see root `README.md`)

### Step 1: Navigate to the project root

```powershell
cd "c:\Users\HP\OneDrive\Desktop\CodeAlpha_ML_Suite"
```

### Step 2: Activate virtual environment

```powershell
.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```powershell
pip install -r requirements.txt
```

Or install only task-specific dependencies:

```powershell
pip install -r task_3_handwritten_character_recognition/requirements_task3.txt
```

### Step 4: Verify environment

```powershell
python check_environment.py
```

---

## 🗺️ Development Workflow (All Phases)

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 0** | Workspace setup, requirements, gitignore | ✅ Done |
| **Phase 1A** | Task 3 project structure initialization | ✅ Done |
| **Phase 2** | MNIST dataset integration & validation | ✅ Done |
| **Phase 3** | CNN model training & hyperparameter tuning | ✅ Done |
| **Phase 4** | Model evaluation, confusion matrix, metrics | ✅ Done |
| **Phase 5** | Flask deployment, web UI & validation tests | ✅ Done |

---

## 🛠️ How to Run Each Module

### Run Dataset Validation
```powershell
python task_3_handwritten_character_recognition/validate_dataset.py
```

### Run Model Training Pipeline
```powershell
python task_3_handwritten_character_recognition/train_model.py
```

### Run Prediction Pipeline Verification
```powershell
python task_3_handwritten_character_recognition/validate_prediction_pipeline.py
```

### Predict on a Custom Image via CLI
```powershell
python task_3_handwritten_character_recognition/predict_digit.py task_3_handwritten_character_recognition/results/predictions/sample_digit_1.png
```

---

## 🚀 Web Application Startup

To start the Flask web application locally:
1. Ensure the virtual environment is active.
2. Run the start command:
   ```powershell
   python task_3_handwritten_character_recognition/app/app.py
   ```
3. Open your browser and go to:
   ```text
   http://127.0.0.1:5000/
   ```

---

## 🧪 Running Unit Tests

To run the Flask application test suite (verifying page load, model load, prediction logic, and upload validation):
```powershell
python task_3_handwritten_character_recognition/test_flask_app.py
```

---

## 📸 Screenshots

*(Place screenshots here demonstrating drawing canvas and file drag-and-drop operations)*

---

## 📝 Notes for Beginners

- **Start with `src/config.py`** — all file paths and settings live there.
- **Never hardcode paths** — always use `Config.SOME_PATH`.
- **Never run training inside a notebook** — use `train_model.py` and keep notebooks for exploration only.
- **If something breaks**, log it in `docs/ERROR_NOTES.md`.

