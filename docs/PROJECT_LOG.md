# Project Log

This file tracks the historical development progress, environment updates, and validation logs for the ML Suite.

---

## [2026-06-23 00:56:39 +05:30] Phase 0: Workspace Setup and Environment Configuration

### What Was Created
- Initialized clean folder structure:
  - `task_1_credit_scoring/` (placeholder)
  - `task_2_emotion_recognition/` (placeholder)
  - `task_3_handwritten_character_recognition/` (placeholder)
  - `docs/` folder containing tracking and reference documents.
- Created root-level environment files:
  - `requirements.txt`: Curated list of library dependencies.
  - `.gitignore`: Configured exclusions for Python, Jupyter notebooks, Windows metadata, and IDEs.
  - `README.md`: Step-by-step setup guides for beginners.
  - `check_environment.py`: Automated environment import validation script.

### What Dependencies Were Added
The following dependencies were included in `requirements.txt`:
1. Core Utilities: `numpy`, `pandas`, `matplotlib`, `seaborn`, `joblib`
2. Modeling & Deep Learning: `scikit-learn`, `tensorflow`
3. Domain Specific: `opencv-python` (computer vision/images), `librosa` (audio processing)
4. Integration / Serving: `flask`
5. Development / Interaction: `jupyter`, `ipykernel`

> [!NOTE]
> Package versions are left unpinned in `requirements.txt` to maximize compatibility with the user's specific local Python version. If a mismatch or build issue occurs, packages will be pinned accordingly.

### What Was Verified
- Structure and syntax of all configuration files, markdown guides, and scripts.
- Tested `check_environment.py` imports and execution logic structurally.

### Warnings & Limitations
- **TensorFlow Size:** Downloading and installing TensorFlow might take longer or require more disk space depending on network speed.
- **Audio Processing on Windows:** `librosa` can sometimes raise warnings if `ffmpeg` or `ffprobe` is missing from the system path when importing or loading compressed formats (e.g., MP3). Standard WAV files are usually supported out-of-the-box.
- **PowerShell Execution Policy:** On Windows, the default execution policy might prevent activating the virtual environment via `.venv\Scripts\Activate.ps1`. The solution is covered in the README.

---

## [2026-06-23 01:03:06 +05:30] Phase 1A: Task 3 — Handwritten Character Recognition Project Initialization

### Files Created

| File | Purpose |
|------|---------|
| `task_3_handwritten_character_recognition/README.md` | Project objective, folder structure, phase roadmap, and usage instructions |
| `task_3_handwritten_character_recognition/requirements_task3.txt` | Task-specific Python dependencies |
| `task_3_handwritten_character_recognition/src/__init__.py` | Marks `src/` as a Python package |
| `task_3_handwritten_character_recognition/src/config.py` | Central configuration for image size, batch size, epochs, all paths |
| `task_3_handwritten_character_recognition/src/data_loader.py` | MNIST dataset loading via `tf.keras.datasets` |
| `task_3_handwritten_character_recognition/src/preprocessing.py` | Normalisation, reshape for CNN, one-hot label encoding |
| `task_3_handwritten_character_recognition/src/model_builder.py` | CNN architecture definition and model compilation |
| `task_3_handwritten_character_recognition/src/trainer.py` | Training loop with callbacks (checkpoint, early stopping, reduce LR) |
| `task_3_handwritten_character_recognition/src/evaluator.py` | Test evaluation, training history plot, confusion matrix, JSON metrics |
| `task_3_handwritten_character_recognition/src/predictor.py` | Single-image and batch inference using OpenCV + trained model |
| `task_3_handwritten_character_recognition/notebooks/exploration.ipynb` | Markdown-only starter notebook with step-by-step EDA plan |
| `task_3_handwritten_character_recognition/app/placeholder.md` | Flask deployment plan document for Phase 5 |

### Folder Structure Created

```
task_3_handwritten_character_recognition/
├── README.md
├── requirements_task3.txt
├── data/
│   ├── raw/          ← MNIST raw files will be placed here
│   ├── processed/    ← Preprocessed .npy arrays will be saved here
│   └── .gitkeep
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model_builder.py
│   ├── trainer.py
│   ├── evaluator.py
│   └── predictor.py
├── models/
│   └── .gitkeep
├── results/
│   ├── plots/
│   ├── metrics/
│   ├── predictions/
│   └── .gitkeep
└── app/
    └── placeholder.md
```

### Validation Checks Performed

- ✅ All `src/` imports are consistent (lazy TensorFlow imports to avoid GPU init on load)
- ✅ All file paths in `config.py` use `os.path.join` with `BASE_DIR` — cross-platform safe
- ✅ No training code executes automatically on import in any module
- ✅ `config.py` paths are consistent with actual directory structure
- ✅ `exploration.ipynb` is markdown-only — no code cells that could run automatically
- ✅ `predictor.py` uses `opencv-python` (`cv2`) which is in `requirements_task3.txt`
- ✅ `evaluator.py` uses `scikit-learn` for confusion matrix — present in requirements

### Warnings & Limitations

- **MNIST first download:** The first call to `DataLoader.load_mnist()` will download ~11MB from the internet to `~/.keras/datasets/`. Subsequent runs use local cache.
- **TensorFlow GPU:** TF will run on CPU by default on most laptops. This is fine for MNIST — training should complete in ~2 minutes per epoch on CPU.
- **Notebook kernel:** The `exploration.ipynb` notebook requires a Jupyter kernel named "ML Env" to be registered. Instructions are in the notebook itself.

---

## [2026-06-23 01:32:40 +05:30] Phase 2: Task 3 — MNIST Dataset Integration and Validation

### Files Modified / Created

| File | Action | What Changed |
|------|--------|--------------|
| `task_3_handwritten_character_recognition/src/data_loader.py` | Modified | Added `validate()`, `get_data_stats()`, `save_raw_npy()`, `load_raw_npy()` |
| `task_3_handwritten_character_recognition/src/preprocessing.py` | Modified | Added inline asserts, `validate()`, `save_processed_npy()`, `load_processed_npy()` |
| `task_3_handwritten_character_recognition/src/config.py` | Modified | Added `DATASET_REPORT_PATH`, `SAMPLE_GRID_PLOT`, `CLASS_DIST_PLOT`, `RAW_NPY` |
| `task_3_handwritten_character_recognition/notebooks/exploration.ipynb` | Modified | Replaced markdown-only cells with 10 working code+markdown cells |
| `task_3_handwritten_character_recognition/validate_dataset.py` | Created | Full standalone validation/plotting/reporting script |

### Validation Completed

- ✅ `DataLoader.validate()` — checks: sample counts, spatial shape, dtype, label range, class count (5 assertions)
- ✅ `Preprocessor.validate()` — checks: dtypes, pixel range, spatial shape, label shape (10 assertions)
- ✅ All functions use lazy TensorFlow imports — module is importable without TF installed
- ✅ `matplotlib.use("Agg")` set in all plot functions — plots save to disk, no GUI window
- ✅ No training code present in any modified or created file
- ✅ `validate_dataset.py` is independently runnable from the task_3 directory

### Dataset Statistics (Expected on Run)

| Metric | Value |
|--------|-------|
| Training samples | 60,000 |
| Test samples | 10,000 |
| Total samples | 70,000 |
| Raw image shape | (28, 28) |
| CNN input shape | (28, 28, 1) |
| Raw pixel range | [0, 255] uint8 |
| Preprocessed range | [0.0, 1.0] float32 |
| Number of classes | 10 (digits 0–9) |
| Class balance | ~10% per class (balanced) |

### Output Files (Generated at Runtime)

| File | Contents |
|------|----------|
| `results/metrics/dataset_report.txt` | Full text summary of dataset stats |
| `results/plots/sample_digit_grid.png` | 2×5 grid of one sample per digit |
| `results/plots/class_distribution.png` | Grouped bar chart: train vs test |
| `data/raw/x_train.npy` | Raw training images |
| `data/raw/y_train.npy` | Raw training labels |
| `data/raw/x_test.npy` | Raw test images |
| `data/raw/y_test.npy` | Raw test labels |
| `data/processed/x_train.npy` | Preprocessed training images (float32) |
| `data/processed/x_test.npy` | Preprocessed test images (float32) |
| `data/processed/y_train.npy` | One-hot training labels |
| `data/processed/y_test.npy` | One-hot test labels |

---

## [2026-06-23 02:24:00 +05:30] Phase 2 Execution: Virtual Environment & Dataset Validation Complete

### Environment Setup & Installation
- **Python Version:** Verified Python 3.11.9 (located at `.venv\Scripts\python.exe`).
- **Dependency Installation:** Installed all requirements from `requirements.txt` successfully (TensorFlow 2.21.0, NumPy 2.4.6, Matplotlib 3.11.0, Seaborn 0.13.2, Scikit-Learn 1.9.0, OpenCV, Librosa, Flask, Jupyter, etc.).
- **Import Verification:** Executed a clean environment check (`check_environment.py`) verifying that all imports of key libraries load properly without issue.

### Fixes Applied to task_3_handwritten_character_recognition/
- **Label Datatype Fix:** Cast one-hot encoded label arrays to `np.float32` explicitly in `src/preprocessing.py` to prevent float64 assertion errors originating from `to_categorical` in newer Keras/TensorFlow versions.
- **Encoding Robustness:** Standardised print statements in `src/data_loader.py`, `src/preprocessing.py`, and `validate_dataset.py` to use pure ASCII-safe characters (e.g., changing unicode symbols like `✓` to `[OK]`, `✅` to `[PASS]`, and blocks `█` to `#`), making execution 100% robust against `UnicodeEncodeError` in Windows consoles running non-UTF-8 character maps.
- **Unicode Safety Catch:** Wrapped terminal report display prints in `validate_dataset.py` in a try-except block to gracefully handle encoding fallbacks if needed.

### Validation execution results
- **Execution Command:** `.venv\Scripts\python.exe task_3_handwritten_character_recognition/validate_dataset.py`
- **Status:** **PASS** (completed successfully with exit code 0).
- **Generated Outputs:**
  - Raw and preprocessed `.npy` files saved under `data/raw/` and `data/processed/`.
  - Visualisations saved under `results/plots/class_distribution.png` and `results/plots/sample_digit_grid.png`.
  - Report saved under `results/metrics/dataset_report.txt`.

---

## [2026-06-23 02:30:00 +05:30] Phase 3: CNN Model Development & Pipeline Verification

### Model Architecture
- **Input:** 28x28x1 grayscale images.
- **Convolutional Block 1:** Conv2D (32 filters, 3x3 kernel, ReLU activation) -> MaxPooling2D (2x2 pool size).
- **Convolutional Block 2:** Conv2D (64 filters, 3x3 kernel, ReLU activation) -> MaxPooling2D (2x2 pool size).
- **Classifier Head:** Flatten -> Dense (128 units, ReLU activation) -> Dropout (rate 0.3) -> Dense (10 units, Softmax activation).
- **Compilation Config:**
  - **Optimizer:** Adam (learning rate = 0.001)
  - **Loss Function:** `categorical_crossentropy`
  - **Metrics:** `['accuracy']`

### Parameter Counts
- **Total Parameters:** 225,034
- **Trainable Parameters:** 225,034
- **Non-Trainable Parameters:** 0

### Training Pipeline Settings
- **Batch Size:** 64
- **Epochs:** 10
- **Validation Split:** 0.1
- **Callbacks:**
  - `EarlyStopping` (monitor=`val_loss`, patience=3, restore_best_weights=True)
  - `ModelCheckpoint` (saves best weights to `models/best_mnist_model.keras`, monitor=`val_loss`, save_best_only=True)

### Verification Execution
- **Command:** `.venv\Scripts\python.exe task_3_handwritten_character_recognition/train_model.py --verify`
- **Verification Status:** **PASS**
- **Verification Execution Output:**
  - Model loaded preprocessed arrays, compiled correctly, ran 1 epoch dry-run on 100 samples, and successfully computed all evaluation metrics and generated all plots.
- **Generated Output Files:**
  - Best Model: `models/best_mnist_model.keras`
  - Loss Plot: `results/plots/training_loss.png`
  - Accuracy Plot: `results/plots/training_accuracy.png`
  - Confusion Matrix: `results/plots/confusion_matrix.png`
  - Classification Report: `results/metrics/classification_report.txt`
  - Final Metrics: `results/metrics/final_metrics.txt`
  - History log: `results/metrics/training_history.json`

---

## [2026-06-23 02:40:00 +05:30] Phase 4: Prediction Pipeline Integration & Validation Complete

### Modules & Files Created
- **Image Preprocessing Utilities (`src/image_utils.py`):**
  - Standardised scaling, grayscale conversion, and resizing to 28x28.
  - Implemented automatic colour inversion if a light background is detected (ensuring black-on-white custom drawings map perfectly to the white-on-black MNIST format).
- **Predictor Module (`src/predictor.py`):**
  - Fully implemented `load_model`, `predict_image`, `predict_array`, and `predict_with_confidence` interfaces.
  - Returns predicted digit class index, confidence score (softmax probability), and the complete 10-dimensional probabilities vector.
  - Added robust exception handling to catch bad array shapes, corrupted images, and file system errors.
- **Command-Line Runner (`predict_digit.py`):**
  - CLI wrapper that enables predicting custom digits: `python predict_digit.py image.png`
- **Validation Suite (`validate_prediction_pipeline.py`):**
  - Script that loads the model, generates test PNGs (0-4), runs predictions through all three prediction interfaces, and verifies confidence constraints.

### Validation Results
- **Pipeline Status:** **PASS** (all interface consistency assertions and confidence range checks passed).
- **Inference Statistics (Sample Digits):**
  - `sample_digit_0.png` -> Predicted: `0` (Confidence: 99.99%)
  - `sample_digit_1.png` -> Predicted: `1` (Confidence: 99.99%)
  - `sample_digit_2.png` -> Predicted: `2` (Confidence: 100.00%)
  - `sample_digit_3.png` -> Predicted: `3` (Confidence: 48.48%)
  - `sample_digit_4.png` -> Predicted: `4` (Confidence: 99.97%)
- **Generated Report File:**
  - `results/predictions/prediction_examples.txt`

---

## [2026-06-23 02:50:00 +05:30] Phase 5: Flask Web Application Integration & Verification Complete

### Modules & Files Created / Modified

| File | Action | What Changed |
|------|--------|--------------|
| `task_3_handwritten_character_recognition/app/app.py` | Modified | Enabled loading model once at startup, verified `/predict` and `GET /` endpoints, handled file validation errors (size limit, empty, invalid format). |
| `task_3_handwritten_character_recognition/app/templates/index.html` | Created | Modern dark-themed user interface dashboard with Canvas drawing board and file upload drag-and-drop zone. |
| `task_3_handwritten_character_recognition/app/static/styles.css` | Created | Elegant layout styling including radial glowing background, CSS transitions, and responsive grid layouts. |
| `task_3_handwritten_character_recognition/app/static/script.js` | Created | Canvas stroke mechanics (mouse & touch support), uploader drag & drop events, AJAX endpoint submission, and UI output metrics display. |
| `task_3_handwritten_character_recognition/test_flask_app.py` | Created | Standalone unit test suite using Flask's test client covering GET `/`, model loading, valid predictions, and invalid file rejection. |
| `task_3_handwritten_character_recognition/DEPLOYMENT.md` | Created | Guide for local run, Render deployment configurations, and Railway service configuration instructions. |
| `task_3_handwritten_character_recognition/README.md` | Modified | Updated directories structure map, status indicators to Complete, and added startup/testing guides. |

### Validation Checks Completed
- ✅ **Homepage loads**: `GET /` returns status code `200` and serves the static assets.
- ✅ **Model loads at startup**: Asserted that `model` loads successfully once on startup and is available to routes.
- ✅ **Prediction endpoint works**: `POST /predict` returns classification results (digit `1`, confidence `99.99%`, and full softmax probabilities vector) in the specified JSON structure.
- ✅ **Unsupported extension validation**: Returns status code `400` with standard JSON error message.
- ✅ **Empty file validation**: Returns status code `400` with standard JSON error message.
- ✅ **Payload size restriction**: Requests >2MB are rejected with status code `413` Payload Too Large.
- ✅ **Test suite verification**: 7 tests run and successfully pass (`test_flask_app.py`).

---

## [2026-06-23 03:10:00 +05:30] Preprocessing Upgrade: Hand-Drawn Canvas Recognition Improvements

### Modules & Files Modified

| File | Action | What Changed |
|------|--------|--------------|
| `task_3_handwritten_character_recognition/src/image_utils.py` | Modified | Replaced simple 28x28 resize with: (1) finding the digit's bounding box to crop empty background, (2) resizing aspect-ratio preserved content to fit inside a 20x20 box, (3) centering the bounding box inside a 28x28 black canvas, and (4) fine-centering the digit via center of mass (moments shift). |
| `task_3_handwritten_character_recognition/validate_prediction_pipeline.py` | Modified | Updated prediction verification to assert correct ground truth digits under the new centering algorithm. Generated a visual subplot mapping 5 samples comparing original input, old direct scaling, and new centered/aspect-preserved outputs. Saved to `results/plots/preprocessing_comparison.png`. |

### Validation Results
- ✅ **Correct predictions on test digits**: All 5 sample images successfully map to the correct ground-truth digits under the new centering preprocessing.
- ✅ **Confidence score improvement**: Sample digit 3 confidence increased from `48.48%` to `57.62%`, showing a significant boost in classification certainty.
- ✅ **Visual validation plot**: Generated and saved `results/plots/preprocessing_comparison.png` displaying the three stages of processing for each sample.
- ✅ **Web application regression check**: Confirmed that all 7 Flask app unit tests continue to pass successfully.

---

## [2026-06-23 03:45:00 +05:30] GitHub Repository Cleanup

### Root Cause
GitHub rejected pushes to the remote repository because the initial commit included large binary dataset files, specifically:
- `task_3_handwritten_character_recognition/data/processed/x_train.npy` (179.44 MB)
This file exceeded GitHub's hard file size limit of 100 MB. Although the file was unstaged/removed from tracking in a later commit index, the original root commit object containing the file remained in the Git DAG history.

### Actions Taken
1. **Exclusion Rules:** Updated `.gitignore` to ignore:
   - `task_3_handwritten_character_recognition/data/raw/`
   - `task_3_handwritten_character_recognition/data/processed/`
   - `*.npy`
2. **Orphan History Rewrite:** Checked out an `--orphan` branch (`clean-main`), unstaged all `.npy` dataset files and directory paths via `git rm --cached`, and staged the updated `.gitignore`.
3. **Commit Metadata Preservation:** Committed the clean codebase under the original commit message ("Task 3 completed - Handwritten Character Recognition"), authorship, and timestamp.
4. **Tag Realignment:** Deleted the old `task3-complete` tag and recreated it pointing to the clean root commit.
5. **Garbage Collection:** Ran aggressive garbage collection (`git gc --prune=now --aggressive`) to permanently remove dangling packfiles containing the large files from the local git database.
6. **Remote Deployment:** Force-pushed the clean branch and tag to the empty remote repository (`origin main`).

### Verification Performed
- Checked git tracked files index: `git ls-files | findstr ".npy"` returns empty (confirmed no `.npy` files are tracked).
- Checked disk files: verified all 8 `.npy` datasets are still intact in their local paths.
- Ran tests: all 7 Flask app tests (`test_flask_app.py`) and the digit prediction validation suite (`validate_prediction_pipeline.py`) run and pass successfully.
- Verified remote push: origin received the branch `main` and tag `task3-complete` successfully.

---

## [2026-06-23 03:50:00 +05:30] Phase 6: Task 1 — Credit Scoring Model Project Structure Initialization

### Files Created
- `task_1_credit_scoring/README.md` (Objective, folder maps, future plans)
- `task_1_credit_scoring/requirements_task1.txt` (Tailored dependencies list: numpy, pandas, scikit-learn, matplotlib, seaborn, joblib, jupyter, ipykernel)
- `task_1_credit_scoring/notebooks/exploration.ipynb` (Starter markdown notebook for EDA)
- `task_1_credit_scoring/app/placeholder.md` (Flask serving plans)
- `task_1_credit_scoring/src/__init__.py` (Python package indicator)
- `task_1_credit_scoring/src/config.py` (Central config class containing seeds, feature lists, and paths)
- `task_1_credit_scoring/src/data_loader.py` (CSV loader skeleton method)
- `task_1_credit_scoring/src/preprocessing.py` (Tabular preprocessor fit/transform skeleton method)
- `task_1_credit_scoring/src/feature_engineering.py` (Domain financial metric construction skeleton)
- `task_1_credit_scoring/src/model_builder.py` (SKLearn classifier setup factory logic)
- `task_1_credit_scoring/src/trainer.py` (Training pipeline coordinator and joblib saving loops)
- `task_1_credit_scoring/src/evaluator.py` (Evaluation metrics: classification reports and ROC parameters)
- `task_1_credit_scoring/src/predictor.py` (Model serving pipeline predictor helper skeleton)

### Folder Structure Created
```text
task_1_credit_scoring/
├── README.md
├── requirements_task1.txt
├── data/
│   ├── raw/                   ← raw datasets directory (contains .gitkeep)
│   └── processed/             ← processed datasets directory (contains .gitkeep)
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model_builder.py
│   ├── trainer.py
│   ├── evaluator.py
│   └── predictor.py
├── models/                     ← model storage directory (contains .gitkeep)
├── results/
│   ├── plots/                 ← validation plots storage (contains .gitkeep)
│   ├── metrics/               ← evaluation metrics output (contains .gitkeep)
│   └── predictions/           ← batch predictions output (contains .gitkeep)
└── app/
    └── placeholder.md
```

### Validation Checks Completed
- ✅ Verified all file paths are correctly generated cross-platform.
- ✅ Confirmed all created source files are valid Python modules and successfully imported.
- ✅ Asserted that no model training, network dataset downloads, or UI serving executes on import.
- ✅ Verified that no Task 3 files or operations were touched.

---

## [2026-06-23 04:00:00 +05:30] Phase 7: Task 1 — German Credit Dataset Integration and Initial EDA

### Dataset Selected
- **Statlog German Credit Dataset** (1,000 samples, 10 attributes, binary risk target).
- **Reason for Selection:** Mix of numerical and categorical financial features, realistic missing value rates in account columns, binary target, and low computational footprint suitable for standard CPUs.

### Files Created/Modified
- `task_1_credit_scoring/data/raw/german_credit_data.csv` [NEW] (Downloaded dataset cache)
- `task_1_credit_scoring/data/raw/dataset_source.txt` [NEW] (Metadata source URL records)
- `task_1_credit_scoring/dataset_info.md` [NEW] (Attribute map profiles and future preprocessing specs)
- `task_1_credit_scoring/validate_dataset.py` [NEW] (Pipeline runner script)
- `task_1_credit_scoring/src/config.py` [MODIFY] (Feature lists configured, added dataset URL)
- `task_1_credit_scoring/src/data_loader.py` [MODIFY] (Concrete data loader, integrity validator, and stats calculator methods implemented)
- `task_1_credit_scoring/notebooks/exploration.ipynb` [MODIFY] (Added working python cells explaining initial loading, inspections, duplicates, and heatmaps)

### EDA Outputs Generated
- `task_1_credit_scoring/results/plots/class_distribution.png` (Risk distribution bar countplot)
- `task_1_credit_scoring/results/plots/missing_values.png` (Missing counts bar plot)
- `task_1_credit_scoring/results/plots/correlation_heatmap.png` (Pearson correlation values for numeric attributes)
- `task_1_credit_scoring/results/metrics/dataset_report.txt` (Complete counts, distributions, null percentages, and datatypes map)

### Validation Results
- ✅ **Shape verification:** Successfully loaded 1,000 instances and 10 features.
- ✅ **Schema verification:** Verified all expected numerical and categorical feature mappings are correctly parsed.
- ✅ **Missing values analysis:** Found missing values in `Saving accounts` (18.30%) and `Checking account` (39.40%). All other columns are complete.
- ✅ **Class balance verification:** Found 70.00% `good` credit risk and 30.00% `bad` credit risk classes.
- ✅ **Duplicate verification:** Verified 0 duplicate instances exist in the dataset.
- ✅ **Workspace safety verification:** Verified no files in other task suites were altered.

---

## [2026-06-23 04:15:00 +05:30] Phase 8: Task 1 — Preprocessing and Feature Engineering Pipeline

### Preprocessing Pipeline Built
- **Pipeline Architecture:** Custom `FeatureEngineer` scikit-learn transformer followed by a `ColumnTransformer` wrapping:
  - **Numerical scaling pipeline:** `SimpleImputer(strategy='median')` -> `StandardScaler()` applied to numeric and engineered features.
  - **Categorical encoding pipeline:** `SimpleImputer(strategy='constant', fill_value='unknown')` -> `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` applied to categorical and engineered features.
- **Split Strategy:** Stratified train/test split (80/20) with random seed `42` ensuring exact proportion of `bad` credit risk classes (30.00%) in both folds.

### Engineered Features Created
1. `credit_per_month` (Credit amount / Duration) - proxy for applicant monthly debt service.
2. `age_group` (young, adult, middle_age, senior) - categorical age group classification.
3. `credit_to_age_ratio` (Credit amount / Age) - debt burden index relative to age maturity.

### Validation Results
- ✅ **Complete matrix dimensions:** Transformed raw columns into a 35-feature sparse float matrix.
- ✅ **Imputation success:** Verified that zero `NaN` values remain in either training (`(800, 35)`) or test (`(200, 35)`) matrices.
- ✅ **Scaling verify:** Asserted numerical variables scale correctly to mean of ~0 and standard deviation of exactly 1.0.
- ✅ **Split verification:** Confirmed exact stratification with `30.00%` high-risk targets in both splits.

### Plots Generated (Saved to `results/plots/`)
- `age_distribution.png` (Histogram mapping applicant age frequency distribution)
- `credit_amount_distribution.png` (Histogram displaying requested credit size frequency)
- `engineered_feature_correlations.png` (Heatmap illustrating correlation ratios across raw and engineered numeric metrics)

### Report File Generated
- `task_1_credit_scoring/results/metrics/preprocessing_report.txt` (Complete list of 35 generated feature column tags and validation dimensions)

---

## [2026-06-23 04:30:00 +05:30] Phase 9: Task 1 — Baseline Model Development and Comparative Evaluation

### Model Performance Metrics (Test Set Summary)
| Model Name | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|------------|----------|-----------|--------|----------|---------|
| **Random Forest** | 0.7500 | 0.5781 | 0.6167 | 0.5968 | 0.7811 |
| **Logistic Regression** | 0.7150 | 0.5176 | 0.7333 | 0.6069 | 0.7481 |
| **Decision Tree** | 0.6750 | 0.4576 | 0.4500 | 0.4538 | 0.6107 |

### Model Leaderboard Ranking
1. **Random Forest** (Rank 1 - ROC-AUC: 0.7811, F1: 0.5968, Recall: 0.6167, Accuracy: 0.7500)
2. **Logistic Regression** (Rank 2 - ROC-AUC: 0.7481, F1: 0.6069, Recall: 0.7333, Accuracy: 0.7150)
3. **Decision Tree** (Rank 3 - ROC-AUC: 0.6107, F1: 0.4538, Recall: 0.4500, Accuracy: 0.6750)

- **Best Baseline Model Selected:** `random_forest` (based on highest ROC-AUC of 0.7811 and balanced precision/recall scores).

### Files Generated
- **Model Checkpoints (`models/`):**
  - `models/logistic_regression.joblib` (dictionary containing preprocessor and model)
  - `models/decision_tree.joblib`
  - `models/random_forest.joblib`
- **Reports (`results/metrics/`):**
  - `results/metrics/logistic_regression_report.txt`
  - `results/metrics/decision_tree_report.txt`
  - `results/metrics/random_forest_report.txt`
  - `results/metrics/model_comparison.csv`
  - `results/metrics/model_ranking.txt`
- **Plots (`results/plots/`):**
  - `results/plots/logistic_regression_confusion_matrix.png`
  - `results/plots/decision_tree_confusion_matrix.png`
  - `results/plots/random_forest_confusion_matrix.png`
  - `results/plots/roc_curve_comparison.png` (displays ROC curves for all models on one figure)
  - `results/plots/model_metrics_comparison.png` (bar chart comparing all metrics across models)

### Validation Checks
- ✅ **Execution check:** `train_models.py` ran end-to-end without errors.
- ✅ **Checkpoints verification:** Verified all 3 model checkpoints load correctly and evaluate on the test split.
- ✅ **Project isolation:** Confirmed no task 2/3 folders were touched.

---

## [2026-06-23 04:45:00 +05:30] Phase 10: Task 1 — Hyperparameter Optimization and Final Model Selection

### Best Hyperparameters Discovered
- `n_estimators`: 100
- `min_samples_split`: 2
- `min_samples_leaf`: 8
- `max_features`: `sqrt`
- `max_depth`: 5
- `class_weight`: `balanced`

### Baseline vs Optimized Random Forest Metrics Comparison
| Metric Name | Baseline RF | Optimized RF | Absolute Change | % Improvement |
|:---|:---:|:---:|:---:|:---:|
| **Accuracy** | 0.7500 | 0.7150 | -0.0350 | -4.67% |
| **Precision** | 0.5781 | 0.5172 | -0.0609 | -10.53% |
| **Recall** | 0.6167 | 0.7500 | +0.1333 | **+21.62%** |
| **F1 Score** | 0.5968 | 0.6122 | +0.0155 | **+2.59%** |
| **ROC-AUC** | 0.7811 | 0.7880 | +0.0069 | **+0.88%** |

### Selected Final Production Model
- **Model Checkpoint:** `models/final_credit_scoring_model.joblib` (preprocessor + model bundle)
- **Preprocessor Checkpoint:** `models/final_preprocessor.joblib` (preprocessor alone)
- **Decision Justification:** The optimized Random Forest classifier was selected because it maximizes general discriminative capacity (**ROC-AUC = 0.7880**) and dramatically improves default protection (**Recall = 75.00%**, catching 45 out of 60 default instances, representing a **+21.62% improvement** over the baseline). The shallow tree constraints (`max_depth=5`, `min_samples_leaf=8`) mitigate overfitting.

### Files & Plots Generated
- **Reports:** `results/metrics/tuning_results.csv`, `results/metrics/best_parameters.txt`, and `results/metrics/final_model_report.txt`.
- **Plots:**
  - `results/plots/tuning_performance.png` (displays mean CV test score progress across iterations)
  - `results/plots/hyperparameter_importance.png` (plots the top 15 Gini feature importances)

### Validation Checks
- ✅ **Execution check:** `optimize_model.py` executed successfully in 20 iterations under stratified 5-fold CV.
- ✅ **Re-loading verify:** Confirmed both the final model checkpoint and the preprocessor checkpoint load correctly.
- ✅ **Workspace safety verification:** Verified no task 2/3 folders were modified.

---

## [2026-06-23 15:45:00 +05:30] Phase 11: Task 1 — Flask Serving Deployment and Unit Validation

### Files Created/Modified

| File | Action | What Changed |
|------|--------|--------------|
| `task_1_credit_scoring/src/predictor.py` | Modified | Replaced placeholder prediction logic with real pipeline inference loading `final_credit_scoring_model.joblib`. |
| `task_1_credit_scoring/app/app.py` | Created | Formulated backend server; loaded global predictor once at startup; added `/` and `/predict` routes; implemented server-side validations; logged predictions. |
| `task_1_credit_scoring/app/templates/index.html` | Created | Built form dashboard with fields for Age, Sex, Job, Housing, Savings, Checking, Credit, Duration, Purpose; added client-side validator script. |
| `task_1_credit_scoring/app/templates/result.html` | Created | Rendered result view displaying risk outcome badges, probability metrics, confidence value, and applicant inputs summary. |
| `task_1_credit_scoring/app/static/style.css` | Created | Implemented glassmorphic dark-theme design style, responsive grid, dynamic hover effects, and color-coded risk alerts. |
| `task_1_credit_scoring/test_flask_app.py` | Created | Assembled 6 unit validation tests (loads, submissions, prediction values, range boundaries, and file output logs). |

### Prediction Logging Registry
- **CSV Registry:** All successful assessments automatically append a row containing timestamps, 9 input variables, model risk predictions (`GOOD`/`BAD`), probability scores, and confidence rating to `results/predictions/prediction_history.csv`.
- **JSON Transaction Logs:** Saves detailed parameters as individual `.json` prediction files (e.g. `pred_<timestamp>_<uuid>.json`) inside `results/predictions/`.

### Validation Checks Completed
- ✅ **Startup initialization check:** Confirmed the production Random Forest model checkpoint loads cleanly.
- ✅ **Test suite validation:** 6 tests run and successfully pass (`test_flask_app.py`).
- ✅ **Browser client-side validation:** Verified HTML5 and JavaScript bounds validation (blocks invalid age, duration, credit size).
- ✅ **Server-side sanitization:** Confirmed incorrect categorical values or empty elements return friendly user-facing alert notifications.
- ✅ **Workspace isolation check:** Verified no files or configurations in `task_2_emotion_recognition` or `task_3_handwritten_character_recognition` were touched.

---

## [2026-06-23 15:35:00 +05:30] Phase 12 (Task 2 — Phase 1): Speech Emotion Recognition Initialization

### Files Created
- `task_2_emotion_recognition/README.md`
- `task_2_emotion_recognition/requirements_task2.txt`
- `task_2_emotion_recognition/src/__init__.py`
- `task_2_emotion_recognition/src/config.py`
- `task_2_emotion_recognition/src/data_loader.py`
- `task_2_emotion_recognition/src/audio_preprocessing.py`
- `task_2_emotion_recognition/src/feature_extraction.py`
- `task_2_emotion_recognition/src/model_builder.py`
- `task_2_emotion_recognition/src/trainer.py`
- `task_2_emotion_recognition/src/evaluator.py`
- `task_2_emotion_recognition/src/predictor.py`
- `task_2_emotion_recognition/validate_initialization.py`

### Validation Checks Completed
- ✅ **Folder setup verify:** Confirmed expected folder structure created.
- ✅ **Import verification check:** Asserted that all python skeleton files load without error or training overhead.
- ✅ **Isolation rule validation:** Verified zero files outside `task_2_emotion_recognition/` were modified.

---

## [2026-06-23 16:15:00 +05:30] Phase 13 (Task 2 — Phase 2): Speech Emotion Dataset Ingestion and Validation

### Files Created/Modified
- `task_2_emotion_recognition/src/data_loader.py` [MODIFY] (Implemented Zenodo down/extract and RAVDESS name parser)
- `task_2_emotion_recognition/data/raw/dataset_source.txt` [NEW] (Reference URLs, DOI, citation rules)
- `task_2_emotion_recognition/dataset_info.md` [NEW] (Detailed name parsing guidelines and statistics mappings)
- `task_2_emotion_recognition/validate_dataset.py` [NEW] (Run download, MD5 checks, plot generation, and reporting)

### Verified Speech Statistics
- **Total Cataloged Files:** 1,440 wav audio files.
- **Valid Waveforms:** 1,440 (all files successfully opened by soundfile).
- **Corrupted Count:** 0.
- **Duplicates Found:** 1 (`03-01-03-01-02-02-07.wav` checksum matched duplicate of `03-01-03-01-02-01-07.wav`).
- **Audio Duration Profile:** min = 2.936s, max = 5.272s, mean = 3.701s (± 0.337s).
- **Class Distributions:** Perfectly balanced gender splits (720 male, 720 female) and emotion categories (192 clips per emotion for 7 categories, and 96 clips for neutral).

### Generated Plots & Reports
- **Visual Plots (results/plots/):**
  - `emotion_distribution.png`: Count per emotion category.
  - `actor_distribution.png`: Count per actor (60 clips each).
  - `duration_distribution.png`: Duration lengths distribution histogram.
- **Quantitative Text Report (results/metrics/):**
  - `dataset_report.txt`: Summary of counts, durations, classes, and identified duplicate paths.

### Validation Checks Completed
- ✅ **Dataset validation execution:** Standalone runner `validate_dataset.py` ran successfully.
- ✅ **Workspace isolation check:** Verified no files or configurations in `task_1_credit_scoring/` or `task_3_handwritten_character_recognition/` were modified.

---

## [2026-06-23 16:30:00 +05:30] Phase 14 (Task 2 — Phase 3): Audio Preprocessing and Feature Extraction Pipeline

### Files Created/Modified
- `task_2_emotion_recognition/src/feature_extraction.py` [MODIFY] (Upgraded to extract both temporal mean and standard deviation vectors)
- `task_2_emotion_recognition/validate_features.py` [NEW] (Pipeline orchestrator script that loads, cleans, extracts, and splits the data)
- `task_2_emotion_recognition/data/processed/features.csv` [NEW] (Tabular features, actor metadata, and target classes)
- `task_2_emotion_recognition/data/processed/features.npy` [NEW] (Unified acoustic features matrix, size 1440x374)
- `task_2_emotion_recognition/data/processed/labels.npy` [NEW] (Acoustic labels vector, size 1440)
- `task_2_emotion_recognition/data/processed/X_train.npy` [NEW] (Training split features, size 1152x374)
- `task_2_emotion_recognition/data/processed/X_test.npy` [NEW] (Testing split features, size 288x374)
- `task_2_emotion_recognition/data/processed/y_train.npy` [NEW] (Training split labels, size 1152)
- `task_2_emotion_recognition/data/processed/y_test.npy` [NEW] (Testing split labels, size 288)

### Audio Preprocessing & Cleaning Statistics
- **Clip Truncating:** Formatted raw audio clips to exact 3.0s duration at 22,050 Hz.
- **Trimming Margin Silence:** Cleaned silences at clip boundaries, reducing average clip length from 3.701s to 1.871s (average reduction of 49.78%).
- **Scaling Normalisation:** Applied peak amplitude scaling to standard max value 1.0 (average scaling multiplier: 16.942).

### Acoustic Descriptor Properties
- **MFCC (40 Coefficients temporal mean + std):** 80 features.
- **Chroma (12 pitch bins temporal mean + std):** 24 features.
- **Mel Spectrogram (128 bands temporal mean + std):** 256 features.
- **Spectral Contrast (7 bands temporal mean + std):** 14 features.
- **Total Combined Features:** 374 variables.

### Generated Plots & Reports
- **Visual Profiling (results/plots/):**
  - `emotion_distribution_split.png`: Train (1,152 clips) vs Test (288 clips) stratified category balances.
  - `feature_variance_distribution.png`: Histogram profiling variance ratios across all 374 feature columns.
  - `feature_correlation_heatmap.png`: Pearson correlation matrix mapping the 15 features with highest variance.
- **Quantitative Metrics Log (results/metrics/):**
  - `feature_report.txt`: Matrix shapes, preprocessing ratios, NaNs counts, and class split balances.

### Validation Checks Completed
- ✅ **Matrix sizes check:** Verified exact split counts (1,152 train, 288 test) and feature sizes (374).
- ✅ **Anomalies audit check:** Confirmed 0 NaN values and 0 infinite values inside all arrays.
- ✅ **Stratification balance check:** Verified target emotion proportions remain identical between splits.
- ✅ **Workspace isolation check:** Confirmed zero files outside `task_2_emotion_recognition/` were modified.

---

## [2026-06-23 16:35:00 +05:30] Phase 15 (Task 2 — Phase 4): Model Training and Selection

### Files Created/Modified
- `task_2_emotion_recognition/src/model_builder.py` [MODIFY] (Added class_weight="balanced" to SVM, configured MLP architecture defaults hidden_layer_sizes=(256, 128) and max_iter=500)
- `task_2_emotion_recognition/src/trainer.py` [MODIFY] (Added save_training_metadata interface method)
- `task_2_emotion_recognition/train_models.py` [NEW] (Coordinator script that fits SVM, RF, and MLP, compares metrics, ranks models, generates plots, and serializes the best model)
- `task_2_emotion_recognition/validate_models.py` [NEW] (Validation script checking model file existence, successful unpickling, and inference prediction correctness)

### Model Evaluation Leaderboard
Scores computed on the 288 test split samples:

| Rank | Model Name | Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **1** | **MLP (Winning Model)** | **0.5729** | **0.5855** | **0.5729** | **0.5744** |
| **2** | Random Forest | 0.5417 | 0.5477 | 0.5417 | 0.5359 |
| **3** | SVM | 0.4757 | 0.5175 | 0.4757 | 0.4842 |

*Selected final model:* `mlp` (ranked highest on primary metric Weighted F1-score).

### Saved Checkpoints & Visuals
- **winning model checkpoint:** `models/final_emotion_model.joblib`
- **individual classification reports:**
  - `results/metrics/svm_report.txt`
  - `results/metrics/random_forest_report.txt`
  - `results/metrics/mlp_report.txt`
- **leaderboard and comparison matrix:**
  - `results/metrics/model_comparison.csv`
  - `results/metrics/model_ranking.txt`
  - `results/metrics/svm_training_metadata.json`
  - `results/metrics/random_forest_training_metadata.json`
  - `results/metrics/mlp_training_metadata.json`
- **charts and heatmaps (results/plots/):**
  - `confusion_matrix_svm.png`
  - `confusion_matrix_random_forest.png`
  - `confusion_matrix_mlp.png`
  - `model_metrics_comparison.png`

### Validation Checks Completed
- ✅ **Serialized model file check:** Confirmed winning MLP pipeline file exists on disk.
- ✅ **Clean unpickling check:** Confirmed final_emotion_model.joblib reloads successfully.
- ✅ **Inference check:** Confirmed reloaded pipeline runs execution successfully on mock sample inputs of size 374.
- ✅ **Output validity check:** Confirmed prediction outputs fall within the 8 valid RAVDESS emotion target categories.
- ✅ **Workspace isolation check:** Confirmed zero files outside `task_2_emotion_recognition/` and shared docs were modified.

---

## [2026-06-23 16:45:00 +05:30] Phase 16 (Task 2 — Phase 5): Hyperparameter Optimization and Model Selection

### Files Created/Modified
- `task_2_emotion_recognition/optimize_models.py` [NEW] (Tuning script using RandomizedSearchCV and StratifiedKFold on SVM, RF, and MLP)
- `task_2_emotion_recognition/validate_optimized_models.py` [NEW] (Sanity check validation script for optimized model checkpoint)

### Optimized Model Leaderboard
Scores computed on the 288 test split samples:

| Rank | Model Name | Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score | F1 Improvement |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **SVM (Winning Model)** | **0.5938** | **0.6052** | **0.5938** | **0.5925** | **+10.83%** |
| **2** | MLP | 0.5799 | 0.5781 | 0.5799 | 0.5718 | -0.26% |
| **3** | Random Forest | 0.5451 | 0.5467 | 0.5451 | 0.5292 | -0.67% |

*Selected final model:* `svm` (ranked highest on primary metric Weighted F1-score). Overwrote previous MLP baseline winner on disk.

### Best Hyperparameters Found
* **SVM**: `{'svm__C': 10, 'svm__gamma': 'scale', 'svm__class_weight': 'balanced'}`
* **Random Forest**: `{'rf__n_estimators': 300, 'rf__min_samples_split': 2, 'rf__min_samples_leaf': 2, 'rf__max_features': 'sqrt', 'rf__max_depth': None, 'rf__class_weight': 'balanced'}`
* **MLP**: `{'mlp__max_iter': 750, 'mlp__learning_rate_init': 0.0005, 'mlp__hidden_layer_sizes': (512, 256), 'mlp__alpha': 0.0001, 'mlp__activation': 'relu'}`

### Saved Checkpoints & Visuals
- **final model checkpoint:** `models/final_emotion_model.joblib`
- **tuning metrics log:**
  - `results/metrics/tuning_results.csv`
  - `results/metrics/best_parameters.txt`
  - `results/metrics/optimized_model_comparison.csv`
  - `results/metrics/optimized_model_ranking.txt`
- **individual classification reports:**
  - `results/metrics/opt_svm_report.txt`
  - `results/metrics/opt_random_forest_report.txt`
  - `results/metrics/opt_mlp_report.txt`
- **charts and heatmaps (results/plots/):**
  - `confusion_matrix_opt_svm.png`
  - `confusion_matrix_opt_random_forest.png`
  - `confusion_matrix_opt_mlp.png`
  - `tuning_performance.png`
  - `optimization_comparison.png` (Baseline vs Optimized F1-Scores)
  - `feature_importance_random_forest.png` (Top 15 features)

### Validation Checks Completed
- ✅ **Serialized model file check:** Confirmed winning optimized SVM pipeline exists on disk.
- ✅ **Clean unpickling check:** Confirmed final_emotion_model.joblib reloads successfully.
- ✅ **Inference check:** Confirmed reloaded pipeline runs execution successfully on mock sample inputs of size 374.
- ✅ **Output validity check:** Confirmed prediction outputs fall within the 8 valid RAVDESS emotion target categories.
- ✅ **Workspace isolation check:** Confirmed zero files outside `task_2_emotion_recognition/` and shared docs were modified.

---

## [2026-06-23 16:55:00 +05:30] Phase 17 (Task 2 — Phase 6): Predictor Integration

### Files Created/Modified
- `task_2_emotion_recognition/src/predictor.py` [MODIFY] (Upgraded EmotionPredictor class with file boundary validation checks, logging, and prediction schema mappings)
- `task_2_emotion_recognition/validate_predictor.py` [NEW] (Inference serving validation runner verifying schema, ranges, logs, and exception states)

### Predictor Interface Design
* **Inference Method**: `EmotionPredictor.predict_emotion(file_path: str) -> dict`
* **Output Format**:
  ```python
  {
      "predicted_emotion": "neutral",
      "confidence_score": 0.8239,
      "probabilities": {
          "angry": 0.0100,
          "calm": 0.0354,
          ...
      }
  }
  ```

### Prediction Logging Registry
* **Registry Path**: `task_2_emotion_recognition/results/predictions/prediction_history.csv`
* **Log Structure**: `timestamp`, `filename`, `predicted_emotion`, `confidence_score` (each transaction dynamically appends to the log file)

### Exception Boundary Safeguards
* **File Non-Existence**: Raises `FileNotFoundError` if the WAV target doesn't exist on disk.
* **Unsupported File Format**: Raises `ValueError` for extensions other than case-insensitive `.wav`.
* **Zero-Byte File**: Raises `ValueError` if size is 0 bytes.
* **Format Corruption**: Raises `RuntimeError` / `ValueError` if headers or chunks are malformed.

### Validation Checks Completed
- ✅ **Clean instantiation check:** Confirmed model pipeline deserializes and reloads cleanly.
- ✅ **Inference schema check:** Verified predicted keys, datatype formats, and emotion string bounds.
- ✅ **Confidence range check:** Confirmed score outputs remain within $[0.0, 1.0]$.
- ✅ **Probabilities sum check:** Confirmed values sum to approximately 1.0.
- ✅ **Logged transaction check:** Verified prediction details are successfully written to prediction_history.csv.
- ✅ **Boundary exception checks:** Verified correct type errors are thrown under missing, wrong format, empty, and corrupted file tests.
- ✅ **Workspace isolation check:** Confirmed zero files outside `task_2_emotion_recognition/` and shared docs were modified.

---

## [2026-06-23 17:15:00 +05:30] Phase 18 (Task 2 — Phase 7): Flask Web Application Serving

### Files Created/Modified
- `task_2_emotion_recognition/app/app.py` [NEW] (Flask server application providing GET / and POST /predict)
- `task_2_emotion_recognition/app/static/style.css` [NEW] (Responsive glassmorphic UI stylesheet with violet/teal glow)
- `task_2_emotion_recognition/app/templates/index.html` [NEW] (Speech analyze home dashboard with drag-and-drop support)
- `task_2_emotion_recognition/app/templates/result.html` [NEW] (Speech classification outputs view highlighting the winning class and probability bars)
- `task_2_emotion_recognition/test_flask_app.py` [NEW] (Flask app test suite containing 6 automated unit tests)
- `task_2_emotion_recognition/validate_flask_app.py` [NEW] (Verification script checking route maps and setup integrity)

### Web App Route Endpoints
* **GET `/`**: Renders `index.html` with file selection prompts. Accepts `error` query parameter to display friendly warning banners.
* **POST `/predict`**:
  - Ingests multipart files, enforcing a **5 MB size limit** (`MAX_CONTENT_LENGTH`).
  - Implements strict validation on extension (`.wav` only) and empty selections.
  - Temporarily saves active uploads, processes inference, deletes temporary uploads, and renders `result.html` on success.
  - Gracefully redirects to `/` on failure.

### Testing Results Summary
All **6 tests** inside the unit test suite [test_flask_app.py](file:///c:/Users/HP/OneDrive/Desktop/CodeAlpha_ML_Suite/task_2_emotion_recognition/test_flask_app.py) passed successfully:
* `test_homepage_loads`: Confirmed home GET `/` status is 200.
* `test_upload_form_renders`: Confirmed upload form markup exists on the page.
* `test_invalid_extension_rejected`: Confirmed non-WAV files redirect with warning message.
* `test_missing_upload_file_rejected`: Confirmed empty file submission redirects with alert.
* `test_missing_upload_parameter_rejected`: Confirmed requests missing payload parameter redirect with warning.
* `test_valid_wav_prediction_and_logging`: Verified valid WAV file processes correctly, yields results markup, and registers transaction log row.

### Validation Checks Completed
- ✅ **Setup verification check:** Standalone runner `validate_flask_app.py` ran successfully.
- ✅ **Route URL mappings check:** Verified `/` and `/predict` endpoints exist in URL maps.
- ✅ **Upload size configuration check:** Confirmed 5 MB upload limit is set.
- ✅ **Predictor integration check:** Confirmed server predictor is initialized and reloads target `final_emotion_model.joblib`.
- ✅ **Workspace isolation check:** Confirmed zero files outside `task_2_emotion_recognition/` and shared docs were modified.






