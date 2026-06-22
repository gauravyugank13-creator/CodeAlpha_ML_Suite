# Error Notes & Debugging Log

Use this file to record errors encountered during environment installation, code execution, data loading, or model training. 

---

## Error Entry Template

To add a new error, copy and paste the markdown template below and fill in the details:

```markdown
### [YYYY-MM-DD] Error Topic Description
* **Error Observed:** 
  *(Paste error traceback or message here)*
* **Likely Cause:** 
  *(Describe why the error occurred, e.g., missing package, library conflict, wrong file path)*
* **Fix Applied:** 
  *(Step-by-step description of what was done to resolve the issue)*
* **Status:** `[Resolved | Investigating]`
```

---

## Logged Issues

*No errors were encountered during the initial workspace setup (Phase 0, 1A).*

---

### [2026-06-23] CRITICAL: Python 3.14 — TensorFlow / NumPy Install Failure

* **Error Observed:**
  ```
  ModuleNotFoundError: No module named 'numpy'

  ERROR: No matching distribution found for tensorflow
  pip install tensorflow → exit code 1

  pip index versions tensorflow → ERROR: No matching distribution found
  ```

* **Environment Audit Results (2026-06-23 01:58 IST):**
  | Item | Value |
  |------|-------|
  | Active Python | `3.14.5` — `C:\Users\HP\AppData\Local\Programs\Python\Python314\python.exe` |
  | Active pip | `26.1.1` targeting Python 3.14 |
  | Other installed Pythons | **None** (only 3.14 via py launcher) |
  | `.venv` present | No |
  | TF wheels for 3.14 | **None exist** (PyPI confirmed) |

* **Root Cause:**
  Python 3.14 was released in 2025 and is too new for the TensorFlow ecosystem.
  As of mid-2026, TensorFlow publishes pre-built wheels **only for Python 3.9, 3.10, 3.11, and 3.12**.
  Python 3.14 introduces breaking changes to the C-level ABI that TensorFlow's build pipeline
  does not yet support. The `pip index versions tensorflow` query returned zero matching
  distributions for `cp314-cp314-win_amd64`, confirming the incompatibility is platform-level, not network-level.

* **Fix Applied:**
  Install Python 3.11 (the most stable, widest-supported version for TF 2.x) alongside the existing 3.14 install.
  Steps:
  1. Install Python 3.11 via winget: `winget install Python.Python.3.11 --source winget`
  2. Create venv using 3.11: `py -3.11 -m venv .venv`
  3. Activate venv: `.venv\Scripts\Activate.ps1`
  4. Upgrade pip: `python -m pip install --upgrade pip`
  5. Install requirements: `pip install -r requirements.txt`
  6. Configure VS Code `.vscode/settings.json` to use `.venv\Scripts\python.exe`
  7. Run `python check_environment.py` to confirm all imports pass.

* **Status:** `[Resolved — virtual environment built with Python 3.11.9, packages installed, environment checks fully passed]`

---

### [2026-06-23] Risk: MNIST Download Failure (Internet Required on First Run)
* **Error Observed:**
  ```
  RuntimeError: Failed to load MNIST. Check your internet connection on first run.
  ```
* **Likely Cause:**
  Keras downloads `mnist.npz` (~11 MB) from `https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz` on the very first call to `load_data()`. If the machine is offline or behind a restricted proxy, the download will fail.
* **Fix Applied:**
  - After a successful first run, raw arrays are saved to `data/raw/` as `.npy` files using `DataLoader.save_raw_npy()`.
  - On subsequent runs (or offline), use `DataLoader.load_raw_npy(Config.RAW_DATA_DIR)` instead.
* **Status:** `[Resolved — offline fallback implemented]`

---

### [2026-06-23] Risk: TensorFlow Not Found / Import Error
* **Error Observed:**
  ```
  ImportError: TensorFlow is not installed. Run: pip install tensorflow
  ```
* **Likely Cause:**
  Running a script outside the activated virtual environment, or forgetting to install requirements.
* **Fix Applied:**
  1. Activate venv: `.venv\Scripts\Activate.ps1`
  2. Install: `pip install -r requirements.txt`
  3. Verify: `python check_environment.py`
  - All TensorFlow imports in `src/` are lazy (inside functions) to prevent this error from blocking other modules.
* **Status:** `[Resolved — lazy imports protect module-level execution]`

---

### [2026-06-23] Risk: matplotlib GUI Window on Windows (Headless Issue)
* **Error Observed:**
  ```
  UserWarning: Matplotlib is currently using TkAgg backend which is not interactive.
  ```
  Or script hangs waiting for a plot window to be closed.
* **Likely Cause:**
  Matplotlib tries to open a GUI window when running from a plain Python script (not a notebook). If no display is available or the backend is wrong, the script may stall or crash.
* **Fix Applied:**
  All plotting functions in `validate_dataset.py` call `matplotlib.use("Agg")` **before** importing `pyplot`. This forces file-only output — no GUI window is opened. Plots are saved directly to `results/plots/`.
* **Status:** `[Resolved — Agg backend set explicitly in all script plotting functions]`

---

### [2026-06-23] Risk: Memory Usage — Large NumPy Arrays
* **Error Observed:**
  Potential `MemoryError` on machines with < 4 GB RAM when saving all .npy arrays simultaneously.
* **Likely Cause:**
  Holding raw (uint8) + preprocessed (float32) arrays in memory at the same time uses approximately:
  - Raw x_train: 60000 × 28 × 28 × 1 byte ≈ **47 MB**
  - Processed x_train: 60000 × 28 × 28 × 4 bytes ≈ **188 MB**
  - Total peak (both splits, raw + processed): ~470 MB
* **Fix Applied:**
  For the MNIST dataset this is well within the capacity of any modern machine. No fix needed. If extending to EMNIST (280,000 samples) in the future, consider processing in chunks.
* **Status:** `[Monitoring — no action needed for MNIST]`

---

### [2026-06-23] CRITICAL: UnicodeEncodeError on Windows Command Console (cp1252)
* **Error Observed:**
  ```
  UnicodeEncodeError: 'charmap' codec can't encode character '\u2713' in position 13: character maps to <undefined>
  ```
* **Likely Cause:**
  Windows console prompts (cmd/PowerShell) use the system's local code page (e.g., CP1252) rather than UTF-8 by default. Printing rich unicode characters like checkmarks (`✓`, `✅`), em dashes (`—`), ellipses (`…`), or blocks (`█`) crashes the standard stream unless forced into UTF-8 mode via `PYTHONUTF8=1` or `PYTHONIOENCODING=utf-8`.
* **Fix Applied:**
  1. Replaced all unicode characters in active print statements with standard ASCII equivalents:
     - `✓` -> `[OK]`
     - `✅` -> `[PASS]`
     - `—` -> `-`
     - `…` -> `...`
     - `█` -> `#`
  2. Wrapped standard output print statements in `validate_dataset.py` inside a try-except `UnicodeEncodeError` fallback that automatically drops to raw ASCII representation.
* **Status:** `[Resolved — all scripts are fully ASCII-safe and Windows-compatible]`

---

### [2026-06-23] Critical: Keras `to_categorical` float64 Datatype AssertionError
* **Error Observed:**
  ```
  AssertionError: One-hot dtype error: expected float32, got float64
  ```
* **Likely Cause:**
  With modern TensorFlow 2.21.0 / NumPy 2.x combinations, `to_categorical` returns labels as `float64` rather than `float32`. This caused the preprocessor's assertion checking to fail during label encoding.
* **Fix Applied:**
  Modified `Preprocessor.one_hot_encode` in `task_3_handwritten_character_recognition/src/preprocessing.py` to explicitly cast the output of `to_categorical` using `.astype(np.float32)`.
* **Status:** `[Resolved — label dtype cast to float32 explicitly]`

---

### [2026-06-23] Risk: Training Memory Overhead & Batch Size Selection
* **Potential Issue:**
  Older or low-resource developer laptops (e.g. < 8 GB RAM) may encounter high memory usage or swap space slowdowns during model fitting.
* **Likely Cause:**
  Loading the entire preprocessed MNIST dataset into memory as float32 tensors takes approximately:
  - 60,000 train images (28x28x1, float32) = 188 MB
  - 10,000 test images (28x28x1, float32) = 31 MB
  While ~220 MB is very lightweight, concurrent IDE, browser, and background process usage can lead to memory exhaustion.
* **Mitigation/Fix Applied:**
  - Standardised batch size to `64` in `Config` which is a stable, standard batch size.
  - Implemented the option in `Trainer.train_model` to load only raw/processed datasets as needed and clear memory.
  - Advised closing unnecessary applications if training lags.
* **Status:** `[Monitoring — lightweight CNN is optimized for low memory]`

---

### [2026-06-23] Risk: TensorFlow Native Windows GPU Deprecation Warnings
* **Observation/Warning:**
  ```
  WARNING:tensorflow:TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11. Even if CUDA/cuDNN are installed, GPU will not be used. Please use WSL2 or the TensorFlow-DirectML plugin.
  ```
* **Likely Cause:**
  TensorFlow discontinued official native GPU support for Windows starting in version 2.11. On native Windows, TensorFlow will fallback to CPU operations.
* **Mitigation/Fix Applied:**
  - No action needed. The lightweight CNN model is highly optimized and small (225k parameters), making it extremely fast to train on standard laptop CPUs.
  - Reassure beginner users that training will complete in ~2-3 minutes total on standard CPUs (approx 10-15 seconds per epoch).
* **Status:** `[Resolved — CPU fallback is normal and sufficiently fast for MNIST]`

---

### [2026-06-23] Risk: Keras Sequential Input Layer Warning
* **Observation/Warning:**
  ```
  UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  ```
* **Likely Cause:**
  Newer versions of Keras deprecate passing `input_shape` inside the first layer constructor and prefer using a separate `layers.Input` layer.
* **Mitigation/Fix Applied:**
  - This is a non-breaking warning and does not affect model correctness or compilation.
  - Kept standard `input_shape` mapping for now to maintain simplicity and compatibility with standard TF 2.x code tutorials, but noted it can be replaced with `layers.Input(shape=input_shape)` in later phases.
* **Status:** `[Monitoring — warning is non-breaking]`

---

### [2026-06-23] Risk: Inference Preprocessing Mismatch & Auto-Inversion Failure
* **Error/Risk Observed:**
  Classifying custom images drawn by users (usually black-on-white) yields very low accuracy or incorrect predictions.
* **Root Cause:**
  MNIST was trained on white digits on a black background (digits are high pixel values, background is 0). If a user provides a black stroke on a white page without inversion, the model sees a "filled square" with a tiny hollow digit shape, causing inference to fail completely.
* **Fix Applied:**
  Implemented automatic background detection in `src/image_utils.py` by calculating the average pixel intensity of the image. If the mean intensity is > 127 (light background), the script automatically performs a bitwise inversion (`cv2.bitwise_not`) to swap the background and foreground, aligning it with the MNIST format.
* **Status:** `[Resolved — auto-inversion logic validated and working]`

---

### [2026-06-23] Risk: Unsupported Custom Image Formats
* **Error/Risk Observed:**
  `ValueError: Unsupported image format` or crashes when loading custom images.
* **Root Cause:**
  The Flask web application or command-line runner might receive formats like `.gif`, `.tiff`, `.bmp`, or transparent `.png` files (which contain an alpha channel). If passed directly, standard 3-channel or 4-channel resizing can corrupt dimensions.
* **Fix Applied:**
  - Standardised file extension check in `src/image_utils.py` to only allow `.png`, `.jpg`, and `.jpeg`.
  - Configured `cv2.imread(..., cv2.IMREAD_GRAYSCALE)` which automatically flattens multi-channel inputs (including RGB and RGBA) to a single grayscale channel upon load.
* **Status:** `[Resolved — whitelist extension check and grayscale flattening enforced]`

---

### [2026-06-23] Risk: Interpretation of Softmax Confidence Scores
* **Risk Observed:**
  The model outputs extremely high confidence scores (e.g. 99.9%) for random scribble images or completely blank pages.
* **Root Cause:**
  Softmax output layers produce relative probabilities that sum to 1.0 across the 10 classes, not absolute likelihoods. A completely blank page or noise scribble might trigger a high activation for a specific class (like 1 or 8) relative to other classes, even if it is not a valid digit.
* **Mitigation/Fix Applied:**
  - Educated users in the UI/documentation that confidence represents relative model certainty within the closed digit class scope (0-9).
  - Recommended thresholding (e.g., ignoring predictions with max confidence < 0.6) or warning users when inputs do not resemble standard handwritten strokes.
* **Status:** `[Monitoring — documentation warning added]`

---

### [2026-06-23] Risk: Cloud Deployment Memory Limit (Render/Railway Free Tier)
* **Risk Observed:**
  Potential container crashes with code `137` (Out of Memory) during model loading or during initial requests on free tiers.
* **Root Cause:**
  TensorFlow is a large deep learning package with a substantial base memory footprint (~350MB+ just for standard import on CPU). On a 512MB RAM free tier container (e.g., Render Free tier), this leaves very little overhead for Flask workers, processing arrays, and OS execution.
* **Mitigation/Fix Applied:**
  - Enforced lazy importing where possible and restricted model loading to a single global instance loaded once at startup.
  - Advised setting environment variable `TF_CPP_MIN_LOG_LEVEL=2` to quieten TF logging memory buffers.
  - Advised configuring the web server to run a single gunicorn worker (`gunicorn --workers=1`) to avoid replicating the TensorFlow base memory footprint multiple times.
* **Status:** `[Monitoring — mitigations documented in DEPLOYMENT.md]`

---

### [2026-06-23] Risk: File Upload Limit Restrictions (413 Payload Too Large)
* **Risk Observed:**
  Uploading high-resolution mobile camera pictures causes Flask to throw 413 Payload Too Large or crash.
* **Root Cause:**
  A raw camera image can easily exceed 5MB to 10MB. Large file transfers block Flask single-threaded workers and take significant processing time.
* **Mitigation/Fix Applied:**
  - Enforced a hard limit `app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024` (2MB) inside `app.py`.
  - Registered a custom `@app.errorhandler(413)` to return a structured JSON response instead of a plain HTML crash page, allowing the client-side JavaScript to display a clean warning in the results panel.
  - Implemented initial client-side size checks in `script.js` before hitting the endpoint to prevent wasting upload bandwidth.
* **Status:** `[Resolved — double-layer check implemented client-side and server-side]`

---

### [2026-06-23] Risk: Freehand Drawing Canvas Distortions & Aspect Ratio Skewing
* **Risk Observed:**
  Freehand canvas drawings have low prediction accuracy compared to official MNIST test set images.
* **Root Cause:**
  User drawings on a web canvas vary widely in scale (e.g. drawn very small in a corner or extremely large, touching borders) and aspect ratio (e.g. a very tall digit '1' or wide digit '0'). Direct resizing to 28x28 skews the aspect ratio, stretching the digit and filling up empty margins, which creates a pattern the CNN classifier was not exposed to. Additionally, noise pixels from thin strokes or aliasing shift the classification boundaries.
* **Mitigation/Fix Applied:**
  - Implemented aspect-ratio-preserving bounding-box cropping using OpenCV contours to strip out empty space.
  - Scaled the cropped box to fit a 20x20 pixel size (the official MNIST content dimension), then padded it into a 28x28 canvas.
  - Shifted the padded canvas using moments to align the digit's center of mass at the exact center (14, 14), which replicates the NIST preprocessing methodology.
* **Status:** `[Resolved — upgraded centering algorithm implemented and verified]`


