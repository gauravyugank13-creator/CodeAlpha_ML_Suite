# Deployment Guide: Handwritten Digit Recognition

This document provides complete instructions for deploying the Handwritten Digit Recognition Flask application locally and to cloud platforms (Render and Railway).

---

## 💻 1. Local Deployment

### Prerequisites
- Python 3.11.x installed (stable target version)
- Active virtual environment (`.venv`) with all packages installed.

### Steps to Run Locally

1. **Activate the virtual environment**:
   - On Windows (PowerShell):
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   - On Linux/macOS:
     ```bash
     source .venv/bin/activate
     ```

2. **Verify environment and model**:
   Before running the app, ensure the trained model exists at `models/best_mnist_model.keras`. If needed, you can run the prediction validation script:
   ```bash
   python task_3_handwritten_character_recognition/validate_prediction_pipeline.py
   ```

3. **Start the Flask server**:
   Run the application from the project root using:
   ```bash
   python task_3_handwritten_character_recognition/app/app.py
   ```
   Alternatively, you can navigate into the `app/` folder and run `python app.py`.

4. **Access the application**:
   Open your browser and navigate to:
   ```text
   http://127.0.0.1:5000/
   ```

---

## ☁️ 2. Render Deployment

Render provides a straightforward way to deploy Python web applications directly from a Git repository.

### Prerequisites for Render
Because TensorFlow is large (~500MB+), running on Render's free tier can occasionally trigger Memory Limit Exceeded errors (which have a limit of 512MB RAM). To mitigate this:
- We use a lightweight model loader.
- We run inference strictly on CPU.

### Deployment Steps on Render

1. **Create a Render Account**:
   Sign up at [render.com](https://render.com) and link your GitHub repository.

2. **Create a Web Service**:
   Click **New +** and select **Web Service**. Select your repository.

3. **Configure Settings**:
   - **Name**: `digit-recognition-suite` (or custom name)
   - **Language**: `Python`
   - **Region**: Select region closest to you
   - **Branch**: `main`
   - **Root Directory**: `task_3_handwritten_character_recognition`
   - **Build Command**:
     ```bash
     pip install -r requirements_task3.txt
     ```
   - **Start Command**:
     We recommend using `gunicorn` for production deployment:
     ```bash
     gunicorn app.app:app
     ```
     *(Note: If you run gunicorn, ensure `gunicorn` is listed in your `requirements_task3.txt` or main `requirements.txt` file)*.

4. **Environment Variables**:
   Under the **Environment** tab, add the following key-value pair if needed:
   - `PYTHON_VERSION`: `3.11.9`
   - `TF_CPP_MIN_LOG_LEVEL`: `2` (silences TensorFlow info logs to save memory)
   - `TF_ENABLE_ONEDNN_OPTS`: `0`

5. **Deploy**:
   Click **Create Web Service**. Render will build the environment, load the packages, and launch the server.

---

## ☁️ 3. Railway Deployment

Railway is a developer-centric cloud platform that supports fast deployments.

### Deployment Steps on Railway

1. **Create a Railway Account**:
   Sign up at [railway.app](https://railway.app) and connect your GitHub account.

2. **Create a New Project**:
   Click **New Project** -> **Deploy from GitHub repo** and select your repository.

3. **Set Root Directory / Monorepo Config**:
   Since the suite is structured as a monorepo containing multiple tasks:
   - In the service settings, set the **Root Directory** to `/task_3_handwritten_character_recognition`.
   - Set the build builder to **Nixpacks** (Railway's default, which automatically detects `requirements_task3.txt` and sets up the Python environment).

4. **Configure Start Command**:
   If Nixpacks does not automatically detect the entrypoint, configure the **Start Command** in the service settings variables/configuration:
   ```bash
   python -m gunicorn app.app:app
   ```
   Or simply:
   ```bash
   python app/app.py
   ```
   *(Since `app.py` has a standard `app.run(...)` blocker, it can run directly, but `gunicorn` is preferred).*

5. **Set Ports**:
   Railway will automatically inject the `PORT` environment variable. In `app/app.py`, the default port is `5000`. Set the Port variable in Railway's service settings:
   - **PORT**: `5000`

6. **Generate Domain**:
   Under the service **Settings** tab, click **Generate Domain** under the networking section to get a public URL for your web app.
