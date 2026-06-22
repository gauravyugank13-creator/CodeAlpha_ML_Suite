# Machine Learning Internship Suite

Welcome to your Machine Learning Internship Suite! This repository contains a clean workspace designed to house three projects:
1. **Credit Scoring Model** (`task_1_credit_scoring/`)
2. **Emotion Recognition from Speech** (`task_2_emotion_recognition/`)
3. **Handwritten Character Recognition** (`task_3_handwritten_character_recognition/`)

This guide will walk you through setting up your local Python development environment from scratch on Windows.

---

## 🛠️ Step-by-Step Environment Setup

Follow these steps to create a local virtual environment, install dependencies, and verify your installation.

### Step 1: Open Terminal in Project Directory
Open your terminal (PowerShell or Command Prompt) and ensure you are in the project root directory:
```powershell
cd "c:\Users\HP\OneDrive\Desktop\CodeAlpha_ML_Suite"
```

### Step 2: Create a Python Virtual Environment
A virtual environment keeps your project dependencies isolated from the rest of your system. Run:
```powershell
python -m venv .venv
```
*Note: This creates a folder named `.venv` containing a clean Python environment.*

### Step 3: Activate the Virtual Environment
Activate the environment so that any packages you install will be contained inside `.venv`.

*   **On PowerShell (Recommended):**
    ```powershell
    .venv\Scripts\Activate.ps1
    ```
    *If you get an Execution Policy error on PowerShell, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` first, then activate.*
    
*   **On Windows Command Prompt (CMD):**
    ```cmd
    .venv\Scripts\activate.bat
    ```

*   **On Git Bash / WSL / macOS / Linux:**
    ```bash
    source .venv/bin/activate
    ```

Once activated, you will see `(.venv)` displayed at the beginning of your terminal prompt line.

### Step 4: Upgrade Package Manager (`pip`)
To ensure smooth installations, run:
```powershell
python -m pip install --upgrade pip
```

### Step 5: Install Required Dependencies
Install all packages defined in `requirements.txt`:
```powershell
pip install -r requirements.txt
```
*Note: Downloading TensorFlow and other libraries might take a few minutes. Please wait until the process completes.*

### Step 6: Verify Setup
Run the check script to make sure everything was installed successfully:
```powershell
python check_environment.py
```

---

## 📁 Workspace Structure

- `requirements.txt`: List of dependencies needed for all tasks.
- `.gitignore`: Instructions to Git on what files and folders to ignore (e.g., virtual environment and caches).
- `check_environment.py`: A Python script to check if the required packages import correctly.
- `docs/`: Holds project tracking, error documentation, and next steps:
  - `PROJECT_LOG.md`: Detailed logs of milestones, modifications, and dependencies.
  - `ERROR_NOTES.md`: Tracking sheet for troubleshooting issues and their solutions.
  - `NEXT_STEPS.md`: High-level plan for what to work on next.
- `task_1_credit_scoring/`: Directory for the credit scoring model.
- `task_2_emotion_recognition/`: Directory for the speech emotion recognition model.
- `task_3_handwritten_character_recognition/`: Directory for the handwritten character recognition model.
