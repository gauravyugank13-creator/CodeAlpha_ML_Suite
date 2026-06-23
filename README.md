# Machine Learning Portfolio Suite

Welcome to the finalized **Machine Learning Portfolio Suite** repository. This workspace contains three production-ready, fully validated machine learning projects, each complete with optimized estimators, pipeline architectures, unit test coverage, and responsive Flask web applications.

## 🚀 Projects Status

| Project | Folder | Status | Primary Model | Key Features |
| :--- | :--- | :---: | :--- | :--- |
| **Task 1: Credit Scoring Model** | [`task_1_credit_scoring/`](file:///c:/Users/HP/OneDrive/Desktop/CodeAlpha_ML_Suite/task_1_credit_scoring/) | **✅ COMPLETE** | Optimized Random Forest | Feature engineering, target encoding, demographic and credit risk profiling dashboard |
| **Task 2: Speech Emotion Recognition** | [`task_2_emotion_recognition/`](file:///c:/Users/HP/OneDrive/Desktop/CodeAlpha_ML_Suite/task_2_emotion_recognition/) | **✅ COMPLETE** | Hyperparameter-tuned SVM | 374-dimensional acoustic vectors (MFCC, Chroma, Mel-Spec), drag-and-drop glassmorphic interface |
| **Task 3: Handwritten Digit Recognition** | [`task_3_handwritten_character_recognition/`](file:///c:/Users/HP/OneDrive/Desktop/CodeAlpha_ML_Suite/task_3_handwritten_character_recognition/) | **✅ COMPLETE** | CNN (Keras/TensorFlow) | MNIST CNN (>98% accuracy), real-time drawing canvas interface, image upload processing |

For a detailed breakdown of models, evaluations, and validation outputs, see the [Submission Summary](file:///c:/Users/HP/OneDrive/Desktop/CodeAlpha_ML_Suite/submission_summary.md) file.

---

## 🛠️ Step-by-Step Environment Setup

Follow these instructions to configure a local Python development environment on Windows:

### 1. Create a Python Virtual Environment
Open your terminal (PowerShell or Command Prompt) in the project root directory and run:
```powershell
python -m venv .venv
```

### 2. Activate the Virtual Environment
Activate the environment to isolate the dependencies:
*   **On PowerShell (Recommended):**
    ```powershell
    .venv\Scripts\Activate.ps1
    ```
    *(Note: If you receive an execution policy error, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first, then activate).*
*   **On Command Prompt (CMD):**
    ```cmd
    .venv\Scripts\activate.bat
    ```
*   **On Git Bash:**
    ```bash
    source .venv/bin/activate
    ```

### 3. Upgrade Pip & Install Dependencies
Upgrade pip to ensure smooth installations, then run the installer:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Local Imports Setup
Run the verify environment helper script:
```powershell
python check_environment.py
```
This confirms that TensorFlow, Flask, Librosa, Scikit-learn, and other packages import successfully.

---

## 💻 Running the Web Applications

Each project has its own web application that can be run concurrently by mapping distinct ports:

### Task 1: Credit Scoring App
*   **Start Command**:
    ```powershell
    python task_1_credit_scoring/app/app.py
    ```
*   **Default Port**: Runs on [http://127.0.0.1:5000](http://127.0.0.1:5000)
*   **Log Location**: Transaction csv log saves to `task_1_credit_scoring/results/predictions/credit_scoring_logs.csv`

### Task 2: Speech Emotion Recognition App
*   **Start Command**:
    ```powershell
    python task_2_emotion_recognition/app/app.py
    ```
*   **Default Port**: Runs on [http://127.0.0.1:5002](http://127.0.0.1:5002) (Port 5002 is selected to avoid port conflicts with the digit canvas app)
*   **Log Location**: Inference csv log saves to `task_2_emotion_recognition/results/predictions/prediction_history.csv`

### Task 3: Handwritten Character Recognition App
*   **Start Command**:
    ```powershell
    python task_3_handwritten_character_recognition/app/app.py
    ```
*   **Default Port**: Runs on [http://127.0.0.1:5000](http://127.0.0.1:5000) (Please run separately or reconfigure port parameters in `app.py` to run concurrently with Task 1)
*   **Log Location**: Canvas predictions and uploads log inside the console stdout.

---

## 🧪 Verification & Test Suites

Each subproject has custom verification runners and unit test files to ensure code correctness and reproducibility:

### Task 1 Tests
*   Run unit test suite (endpoints, forms, validation limits):
    ```powershell
    python -m unittest task_1_credit_scoring/test_flask_app.py
    ```
*   Run data pipeline verification tests:
    ```powershell
    python task_1_credit_scoring/validate_preprocessing.py
    ```

### Task 2 Tests
*   Run unit test suite (file extension checks, size validation, endpoint logs):
    ```powershell
    python -m unittest task_2_emotion_recognition/test_flask_app.py
    ```
*   Run server routing audit and predictor sanity validation:
    ```powershell
    python task_2_emotion_recognition/validate_flask_app.py
    python task_2_emotion_recognition/validate_predictor.py
    ```

### Task 3 Tests
*   Run unit test suite (canvas drawing uploads, model prediction logic):
    ```powershell
    python -m unittest task_3_handwritten_character_recognition/test_flask_app.py
    ```
*   Run dataset matrix structure audits:
    ```powershell
    python task_3_handwritten_character_recognition/validate_dataset.py
    ```

---

## 📁 Workspace Directory Structure

*   `requirements.txt`: Root dependencies file.
*   `check_environment.py`: Environment validation helper.
*   `submission_summary.md`: Final completion summary details.
*   `docs/`: Workspace logging files:
    *   `PROJECT_LOG.md`: Chronological log of phases, files modified, and model leaderboards.
    *   `ERROR_NOTES.md`: Tracking of debugging issues, mitigations, and solutions.
    *   `NEXT_STEPS.md`: Completed phases roadmap.
*   `task_1_credit_scoring/`: Credit Scoring subproject directory.
    *   `models/final_credit_scoring_model.joblib`: Serialized Random Forest estimator.
*   `task_2_emotion_recognition/`: Speech Emotion Recognition subproject.
    *   `models/final_emotion_model.joblib`: Serialized SVM estimator.
*   `task_3_handwritten_character_recognition/`: Handwritten character recognition.
    *   `models/best_mnist_model.keras`: Serialized Keras CNN estimator.

---

## 🛡️ Git Isolation Status
All subproject directories are locked. Generated datasets, raw audio, and compiled `.npy` matrices are properly excluded by `.gitignore` rules to keep the git index small and prevent file boundary limits issues.

The workspace is finalized and ready for evaluation!
