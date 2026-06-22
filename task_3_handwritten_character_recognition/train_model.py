"""
train_model.py
Task 3 — Handwritten Character Recognition

Purpose:
    Command-line interface to build, train, evaluate, and save the CNN model.
    Runs the complete pipeline end-to-end:
        1. Loads preprocessed datasets (x_train, x_test, y_train, y_test)
        2. Builds and compiles the lightweight CNN model
        3. Configures EarlyStopping and ModelCheckpoint callbacks
        4. Trains the model (with an optional quick verification dry-run)
        5. Evaluates model performance on the test split
        6. Generates loss/accuracy plots and confusion matrix
        7. Saves classification reports and metric text files

Usage:
    # To run a quick end-to-end verification (1 epoch, 100 samples):
    python train_model.py --verify

    # To run the full model training (10 epochs on full dataset):
    python train_model.py --train
"""

import os
import sys
import argparse

# Make sure src/ is importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from src.config import Config
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.model_builder import ModelBuilder
from src.trainer import Trainer
from src.evaluator import Evaluator


def run_pipeline(dry_run: bool = False):
    print("\n" + "=" * 60)
    print("  MNIST CNN Model Training & Evaluation Pipeline")
    print("=" * 60 + "\n")

    # 1. Load preprocessed arrays
    try:
        print("[Pipeline] Loading preprocessed data...")
        x_train, x_test, y_train, y_test = Preprocessor.load_processed_npy(Config.PROCESSED_DATA_DIR)
    except FileNotFoundError:
        print("[Pipeline] Preprocessed data not found. Running loading and preprocessing first...")
        # Fallback to loading and preprocessing from raw if not present
        (x_tr_raw, y_tr_raw), (x_te_raw, y_te_raw) = DataLoader.load_mnist()
        x_train, x_test, y_train, y_test = Preprocessor.run(x_tr_raw, y_tr_raw, x_te_raw, y_te_raw)
        Preprocessor.save_processed_npy(x_train, x_test, y_train, y_test, Config.PROCESSED_DATA_DIR)

    # 2. Build and compile the model
    print("\n[Pipeline] Building lightweight CNN model...")
    model = ModelBuilder.build_cnn_model(
        input_shape=(Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1], Config.IMAGE_CHANNELS),
        num_classes=Config.NUM_CLASSES,
        learning_rate=Config.LEARNING_RATE
    )

    # Parameter count
    param_info = ModelBuilder.get_parameter_count(model)
    print(f"\n[Pipeline] Model parameters count:")
    print(f"  Total Params     : {param_info['total_params']:,}")
    print(f"  Trainable Params : {param_info['trainable_params']:,}")
    print(f"  Non-Trainable    : {param_info['non_trainable_params']:,}")
    print("-" * 60)

    # 3. Train the model
    print("\n[Pipeline] Training model...")
    history = Trainer.train_model(
        model=model,
        x_train=x_train,
        y_train=y_train,
        batch_size=Config.BATCH_SIZE,
        epochs=Config.EPOCHS,
        validation_split=Config.VALIDATION_SPLIT,
        dry_run=dry_run
    )

    # 4. Evaluate model
    print("\n[Pipeline] Evaluating model on test split...")
    test_metrics = Evaluator.evaluate(model, x_test, y_test)

    # 5. Save metrics and reports
    print("\n[Pipeline] Saving metrics and reports...")
    Evaluator.save_final_metrics(test_metrics, Config.FINAL_METRICS_PATH)
    Evaluator.generate_classification_report(model, x_test, y_test, Config.CLASSIFICATION_REPORT_PATH)

    # 6. Generate plots
    print("\n[Pipeline] Generating visual plots...")
    Evaluator.plot_training_accuracy(history, Config.TRAINING_ACCURACY_PLOT)
    Evaluator.plot_training_loss(history, Config.TRAINING_LOSS_PLOT)
    Evaluator.plot_confusion_matrix(model, x_test, y_test, class_names=Config.CLASS_NAMES, save_path=Config.CONFUSION_MATRIX_PLOT)

    print("\n" + "=" * 60)
    print("  [PASS] Pipeline Finished Successfully!")
    print("=" * 60)
    print(f"  Best model saved to : {os.path.join(Config.MODELS_DIR, 'best_mnist_model.keras')}")
    print(f"  Accuracy plot       : {Config.TRAINING_ACCURACY_PLOT}")
    print(f"  Loss plot           : {Config.TRAINING_LOSS_PLOT}")
    print(f"  Confusion matrix    : {Config.CONFUSION_MATRIX_PLOT}")
    print(f"  Classification rep. : {Config.CLASSIFICATION_REPORT_PATH}")
    print(f"  Final metrics       : {Config.FINAL_METRICS_PATH}\n")


def main():
    parser = argparse.ArgumentParser(description="MNIST CNN Training and Evaluation pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Execute full training on dataset")
    group.add_argument("--verify", action="store_true", help="Execute rapid dry-run to verify scripts and compile model")
    
    args = parser.parse_args()
    
    if args.verify:
        run_pipeline(dry_run=True)
    elif args.train:
        run_pipeline(dry_run=False)


if __name__ == "__main__":
    main()
