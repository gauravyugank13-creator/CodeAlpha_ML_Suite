"""
Task 3 — Handwritten Character Recognition
src/trainer.py

Purpose:
    Manages the full training loop for the CNN model.
    Handles callbacks (model checkpointing, early stopping),
    training history logging, and saving the trained model.

    Receives a compiled model from model_builder.py and
    preprocessed data from preprocessing.py.

Usage:
    from src.trainer import Trainer
    history = Trainer.train(model, x_train, y_train, x_val, y_val)
    Trainer.save_model(model, Config.MODEL_SAVE_PATH)
"""

import os


class Trainer:
    """
    Handles model training configuration, execution, and saving.

    Methods:
        get_callbacks(checkpoint_dir) — Build a list of Keras callbacks.
        train(...)                    — Run the training loop.
        save_model(model, path)       — Persist the trained model to disk.
    """

    @staticmethod
    def get_callbacks(checkpoint_dir: str) -> list:
        """
        Build standard Keras callbacks for training.

        Callbacks included:
            - ModelCheckpoint: Saves the best model weights during training.
            - EarlyStopping:   Stops training if val_loss stops improving.
            - ReduceLROnPlateau: Reduces learning rate when stuck.

        Args:
            checkpoint_dir: Path to save model checkpoints.

        Returns:
            List of tf.keras.callbacks objects.

        TODO (Phase 3 — Model Training):
            - Add TensorBoard callback for interactive loss/accuracy visualisation.
            - Add CSVLogger callback to log epoch metrics to a CSV file.
        """
        try:
            from tensorflow.keras.callbacks import (
                ModelCheckpoint,
                EarlyStopping,
                ReduceLROnPlateau,
            )
        except ImportError as e:
            raise ImportError(
                "TensorFlow is not installed. Run: pip install tensorflow"
            ) from e

        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=5,             # Stop if no improvement for 5 epochs
            restore_best_weights=True,
            verbose=1,
        )

        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,             # Halve the learning rate
            patience=3,
            min_lr=1e-6,
            verbose=1,
        )

        return [checkpoint, early_stop, reduce_lr]

    @staticmethod
    def train_model(
        model=None,
        x_train=None,
        y_train=None,
        batch_size: int = 64,
        epochs: int = 10,
        validation_split: float = 0.1,
        dry_run: bool = False,
    ):
        """
        Execute the full model training pipeline.
        Loads preprocessed data from data/processed/, builds/compiles the
        lightweight CNN model, configures EarlyStopping and ModelCheckpoint,
        and saves the best model to models/best_mnist_model.keras.

        Args:
            model:            Optional tf.keras.Model instance. If None, builds build_cnn_model().
            x_train:          Optional preprocessed training images.
            y_train:          Optional preprocessed training labels.
            batch_size:       Batch size per gradient update. Default: 64.
            epochs:           Max epochs. Default: 10.
            validation_split: Fraction of training data for validation. Default: 0.1.
            dry_run:          If True, trains only on 100 samples for 1 epoch for verification.

        Returns:
            tf.keras.callbacks.History: Training history object.
        """
        import numpy as np
        import json
        from src.config import Config
        from src.preprocessing import Preprocessor
        from src.model_builder import ModelBuilder
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

        # 1. Load processed data if not provided
        if x_train is None or y_train is None:
            print(f"[Trainer] Loading processed data from: {Config.PROCESSED_DATA_DIR}")
            x_train, _, y_train, _ = Preprocessor.load_processed_npy(Config.PROCESSED_DATA_DIR)

        if dry_run:
            print("[Trainer] Dry run mode enabled. Subsetting dataset to 100 samples for 1 epoch.")
            x_train = x_train[:100]
            y_train = y_train[:100]
            epochs = 1

        # 2. Build model if not provided
        if model is None:
            model = ModelBuilder.build_cnn_model()

        # 3. Configure Callbacks
        os.makedirs(Config.MODELS_DIR, exist_ok=True)
        checkpoint_path = os.path.join(Config.MODELS_DIR, "best_mnist_model.keras")
        
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        )

        print("[Trainer] Starting model training...")
        print(f"  Batch size       : {batch_size}")
        print(f"  Epochs           : {epochs}")
        print(f"  Validation split : {validation_split}")

        # 4. Train model
        history = model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=[checkpoint, early_stop],
            verbose=1,
        )

        print("[Trainer] Training complete.")

        # 5. Store training history
        os.makedirs(Config.METRICS_DIR, exist_ok=True)
        history_path = os.path.join(Config.METRICS_DIR, "training_history.json")
        history_dict = {k: [float(val) for val in v] for k, v in history.history.items()}
        with open(history_path, "w") as f:
            json.dump(history_dict, f, indent=4)
        print(f"[Trainer] Training history saved to: {history_path}")

        return history

    @staticmethod
    def save_model(model, save_path: str) -> None:
        """
        Save the trained Keras model to disk in .keras format.

        Args:
            model:     Trained tf.keras.Model instance.
            save_path: Destination file path (e.g., models/mnist_cnn_model.keras).

        Returns:
            None

        TODO (Phase 3 — Model Training):
            - Also export as SavedModel format for TensorFlow Serving.
            - Print file size of the saved model.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        print(f"[Trainer] Model saved to: {save_path}")


# ---------------------------------------------------------------------------
# TODO: Phase 3 — Model Training
#   - Add a run_training_pipeline() convenience function that chains
#     DataLoader → Preprocessor → ModelBuilder → Trainer in one call.
# ---------------------------------------------------------------------------
