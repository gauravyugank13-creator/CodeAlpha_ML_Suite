"""
Task 3 — Handwritten Character Recognition
src/config.py

Purpose:
    Central configuration file for all project settings.
    All hyperparameters, file paths, and constants are defined here.
    Other modules should import from this file instead of hardcoding values.

Usage:
    from src.config import Config
    print(Config.IMAGE_SIZE)
    print(Config.MODEL_SAVE_PATH)
"""

import os

# ---------------------------------------------------------------------------
# Base directory: resolves to task_3_handwritten_character_recognition/
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Config:
    """
    Central configuration class.

    All settings are class-level attributes so they can be accessed
    without instantiation:  Config.EPOCHS, Config.BATCH_SIZE, etc.
    """

    # -------------------------------------------------------------------
    # Image settings
    # -------------------------------------------------------------------
    IMAGE_SIZE = (28, 28)        # MNIST standard image dimensions (height, width)
    IMAGE_CHANNELS = 1           # Grayscale images
    NUM_CLASSES = 10             # Digits 0-9

    # -------------------------------------------------------------------
    # Training hyperparameters
    # -------------------------------------------------------------------
    BATCH_SIZE = 64              # Number of samples per gradient update
    EPOCHS = 10                  # Number of full passes through the dataset
    LEARNING_RATE = 0.001        # Optimizer learning rate
    VALIDATION_SPLIT = 0.1      # Fraction of training data used for validation
    RANDOM_SEED = 42             # For reproducibility

    # -------------------------------------------------------------------
    # Dataset paths
    # -------------------------------------------------------------------
    DATA_DIR = os.path.join(BASE_DIR, "data")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

    # -------------------------------------------------------------------
    # Model paths
    # -------------------------------------------------------------------
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "mnist_cnn_model.keras")
    MODEL_CHECKPOINT_DIR = os.path.join(MODELS_DIR, "checkpoints")

    # -------------------------------------------------------------------
    # Results / output paths
    # -------------------------------------------------------------------
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
    METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
    PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")

    # Specific output files — evaluation (Phase 4)
    METRICS_FILE = os.path.join(METRICS_DIR, "evaluation_metrics.json")
    CONFUSION_MATRIX_PLOT = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    TRAINING_HISTORY_PLOT = os.path.join(PLOTS_DIR, "training_history.png")
    TRAINING_ACCURACY_PLOT = os.path.join(PLOTS_DIR, "training_accuracy.png")
    TRAINING_LOSS_PLOT = os.path.join(PLOTS_DIR, "training_loss.png")
    CLASSIFICATION_REPORT_PATH = os.path.join(METRICS_DIR, "classification_report.txt")
    FINAL_METRICS_PATH = os.path.join(METRICS_DIR, "final_metrics.txt")

    # Phase 2 — Dataset validation outputs
    DATASET_REPORT_PATH = os.path.join(METRICS_DIR, "dataset_report.txt")
    SAMPLE_GRID_PLOT    = os.path.join(PLOTS_DIR,   "sample_digit_grid.png")
    CLASS_DIST_PLOT     = os.path.join(PLOTS_DIR,   "class_distribution.png")

    # Raw .npy file paths (written by DataLoader.save_raw_npy)
    RAW_NPY = {
        "x_train": os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw"), "x_train.npy"),
        "y_train": os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw"), "y_train.npy"),
        "x_test" : os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw"), "x_test.npy"),
        "y_test" : os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw"), "y_test.npy"),
    }

    # -------------------------------------------------------------------
    # Class labels
    # -------------------------------------------------------------------
    CLASS_NAMES = [str(i) for i in range(10)]   # ['0','1',...,'9']

    @classmethod
    def display(cls):
        """Print all configuration values to stdout for quick inspection."""
        print("=" * 50)
        print("  Task 3 Configuration Summary")
        print("=" * 50)
        for key, value in vars(cls).items():
            if not key.startswith("_") and not callable(value):
                print(f"  {key:<30} = {value}")
        print("=" * 50)


# ---------------------------------------------------------------------------
# Phase 2 — DONE: DATASET_REPORT_PATH, SAMPLE_GRID_PLOT, CLASS_DIST_PLOT, RAW_NPY added.
#
# TODO: Phase 3 — Model Training
#   - Add early stopping patience as Config.EARLY_STOP_PATIENCE.
#   - Add dropout rate as Config.DROPOUT_RATE.
#
# TODO: Phase 5 — Deployment
#   - Add Config.FLASK_HOST and Config.FLASK_PORT.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    Config.display()
