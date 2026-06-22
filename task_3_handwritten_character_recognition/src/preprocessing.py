"""
Task 3 — Handwritten Character Recognition
src/preprocessing.py

Purpose:
    Transforms raw MNIST arrays into a format suitable for CNN training.
    Handles normalisation, reshaping, and one-hot encoding.
    Also provides assert-based validation after each step and helpers
    to save/load processed arrays for fast reuse.

    Receives raw arrays from data_loader.py and returns processed
    arrays ready to be consumed by model_builder.py / trainer.py.

Usage:
    from src.preprocessing import Preprocessor

    # Full pipeline in one call
    x_tr, x_te, y_tr, y_te = Preprocessor.run(x_train, y_train, x_test, y_test)

    # Validate the result
    Preprocessor.validate(x_tr, y_tr, x_te, y_te)

    # Save for offline / fast reuse
    Preprocessor.save_processed_npy(x_tr, x_te, y_tr, y_te, processed_dir)
"""

import os
import numpy as np


class Preprocessor:
    """
    Applies all preprocessing steps to the raw MNIST dataset.

    Steps performed:
        1. Normalise pixel values from [0, 255] to [0.0, 1.0].
        2. Reshape images to add a channel dimension: (N,28,28) → (N,28,28,1).
        3. One-hot encode integer class labels: (N,) → (N,10).

    Methods:
        normalise(x)                  — Scale pixels to [0, 1].
        reshape_for_cnn(x)            — Add channel dimension.
        one_hot_encode(y, n_cls)      — Convert labels to one-hot vectors.
        validate(x_tr, y_tr, x_te, y_te) — Assert all shapes/ranges are correct.
        run(...)                      — Apply all steps in order.
        save_processed_npy(...)       — Persist processed arrays to data/processed/.
        load_processed_npy(path)      — Load previously saved processed arrays.
    """

    # Expected post-processing shapes
    EXPECTED_IMG_SHAPE   = (28, 28, 1)
    EXPECTED_N_CLASSES   = 10

    # ------------------------------------------------------------------ #
    #  Individual transformation steps
    # ------------------------------------------------------------------ #

    @staticmethod
    def normalise(x: np.ndarray) -> np.ndarray:
        """
        Normalise pixel values to the range [0.0, 1.0].

        Args:
            x: uint8 NumPy array with values in [0, 255].

        Returns:
            float32 NumPy array with values in [0.0, 1.0].

        Example:
            pixel value 128  →  0.502
            pixel value 255  →  1.0
            pixel value 0    →  0.0
        """
        normalised = x.astype("float32") / 255.0
        # Quick sanity guard
        assert normalised.min() >= 0.0, "Normalised min is below 0.0"
        assert normalised.max() <= 1.0, "Normalised max is above 1.0"
        return normalised

    @staticmethod
    def reshape_for_cnn(x: np.ndarray) -> np.ndarray:
        """
        Reshape images to include an explicit channel dimension.

        MNIST images are (28, 28) — grayscale.
        TensorFlow/Keras Conv2D layers expect (H, W, C), so we add C=1.

        Args:
            x: NumPy array with shape (N, H, W).

        Returns:
            NumPy array with shape (N, H, W, 1).
        """
        H, W = x.shape[1], x.shape[2]
        reshaped = x.reshape(x.shape[0], H, W, 1)
        assert reshaped.shape[1:] == (H, W, 1), (
            f"Reshape failed: expected (N, {H}, {W}, 1), got {reshaped.shape}"
        )
        return reshaped

    @staticmethod
    def one_hot_encode(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
        """
        Convert integer class labels to one-hot encoded vectors.

        Example:
            label  3  →  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
            label  0  →  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        Args:
            y:           NumPy array of integer labels, shape (N,).
            num_classes: Total number of classes. Default: 10 (MNIST digits).

        Returns:
            NumPy float32 array of shape (N, num_classes).

        Raises:
            ImportError: If TensorFlow is not installed.
        """
        try:
            from tensorflow.keras.utils import to_categorical
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is not installed. Run: pip install tensorflow"
            ) from exc

        encoded = to_categorical(y, num_classes=num_classes).astype(np.float32)
        assert encoded.shape == (len(y), num_classes), (
            f"One-hot shape error: expected ({len(y)}, {num_classes}), "
            f"got {encoded.shape}"
        )
        assert encoded.dtype == np.float32, (
            f"One-hot dtype error: expected float32, got {encoded.dtype}"
        )
        return encoded

    # ------------------------------------------------------------------ #
    #  Full pipeline
    # ------------------------------------------------------------------ #

    @classmethod
    def run(
        cls,
        x_train : np.ndarray,
        y_train : np.ndarray,
        x_test  : np.ndarray,
        y_test  : np.ndarray,
        num_classes: int = 10,
        verbose : bool = True,
    ):
        """
        Execute the complete preprocessing pipeline.

        Order of operations:
            1. Normalise  (images)
            2. Reshape    (add channel dim)
            3. One-hot    (labels)
            4. Validate   (assert shapes)

        Args:
            x_train, y_train : Raw training arrays.
            x_test,  y_test  : Raw test arrays.
            num_classes      : Number of output classes (default 10).
            verbose          : Whether to print progress messages.

        Returns:
            tuple: (x_train_p, x_test_p, y_train_p, y_test_p)
                x_train_p : float32 (60000, 28, 28, 1)
                x_test_p  : float32 (10000, 28, 28, 1)
                y_train_p : float32 (60000, 10)
                y_test_p  : float32 (10000, 10)

        IMPORTANT:
            This function does NOT train a model — it only transforms data.
        """
        if verbose:
            print("\n[Preprocessor] Starting preprocessing pipeline ...")

        if verbose:
            print("[Preprocessor] Step 1/3 - Normalising pixel values ...")
        x_train_p = cls.normalise(x_train)
        x_test_p  = cls.normalise(x_test)

        if verbose:
            print("[Preprocessor] Step 2/3 - Reshaping for CNN input ...")
        x_train_p = cls.reshape_for_cnn(x_train_p)
        x_test_p  = cls.reshape_for_cnn(x_test_p)

        if verbose:
            print("[Preprocessor] Step 3/3 - One-hot encoding labels ...")
        y_train_p = cls.one_hot_encode(y_train, num_classes)
        y_test_p  = cls.one_hot_encode(y_test,  num_classes)

        # Full validation
        cls.validate(x_train_p, y_train_p, x_test_p, y_test_p, num_classes)

        if verbose:
            print(f"[Preprocessor] [PASS] Pipeline complete.")
            print(f"  x_train : {x_train_p.shape}  dtype={x_train_p.dtype}")
            print(f"  x_test  : {x_test_p.shape}  dtype={x_test_p.dtype}")
            print(f"  y_train : {y_train_p.shape}  dtype={y_train_p.dtype}")
            print(f"  y_test  : {y_test_p.shape}  dtype={y_test_p.dtype}\n")

        return x_train_p, x_test_p, y_train_p, y_test_p

    # ------------------------------------------------------------------ #
    #  Validation
    # ------------------------------------------------------------------ #

    @staticmethod
    def validate(
        x_train_p  : np.ndarray,
        y_train_p  : np.ndarray,
        x_test_p   : np.ndarray,
        y_test_p   : np.ndarray,
        num_classes: int = 10,
    ) -> bool:
        """
        Assert that preprocessed arrays have correct shapes and value ranges.

        Checks:
            1.  x_train_p dtype == float32
            2.  x_test_p  dtype == float32
            3.  x_train_p pixel range in [0.0, 1.0]
            4.  x_test_p  pixel range in [0.0, 1.0]
            5.  x_train_p shape[1:] == (28, 28, 1)
            6.  x_test_p  shape[1:] == (28, 28, 1)
            7.  y_train_p shape[1]  == num_classes
            8.  y_test_p  shape[1]  == num_classes
            9.  y_train_p dtype == float32
            10. y_test_p  dtype == float32

        Returns:
            True if all checks pass.

        Raises:
            AssertionError with a descriptive message on first failure.
        """
        print("[Preprocessor] Running post-processing validation ...")

        # dtypes
        assert x_train_p.dtype == np.float32, f"x_train dtype: {x_train_p.dtype}"
        assert x_test_p.dtype  == np.float32, f"x_test  dtype: {x_test_p.dtype}"
        assert y_train_p.dtype == np.float32, f"y_train dtype: {y_train_p.dtype}"
        assert y_test_p.dtype  == np.float32, f"y_test  dtype: {y_test_p.dtype}"

        # pixel range
        assert x_train_p.min() >= 0.0, f"x_train min {x_train_p.min()} < 0.0"
        assert x_train_p.max() <= 1.0, f"x_train max {x_train_p.max()} > 1.0"
        assert x_test_p.min()  >= 0.0, f"x_test  min {x_test_p.min()} < 0.0"
        assert x_test_p.max()  <= 1.0, f"x_test  max {x_test_p.max()} > 1.0"

        # spatial shape
        assert x_train_p.shape[1:] == (28, 28, 1), (
            f"x_train spatial: {x_train_p.shape[1:]}"
        )
        assert x_test_p.shape[1:] == (28, 28, 1), (
            f"x_test  spatial: {x_test_p.shape[1:]}"
        )

        # label shape
        assert y_train_p.shape[1] == num_classes, (
            f"y_train classes: {y_train_p.shape[1]}"
        )
        assert y_test_p.shape[1] == num_classes, (
            f"y_test  classes: {y_test_p.shape[1]}"
        )

        print("[Preprocessor] [PASS] All post-processing checks passed.\n")
        return True

    # ------------------------------------------------------------------ #
    #  Save / Load processed arrays
    # ------------------------------------------------------------------ #

    @staticmethod
    def save_processed_npy(
        x_train_p  : np.ndarray,
        x_test_p   : np.ndarray,
        y_train_p  : np.ndarray,
        y_test_p   : np.ndarray,
        processed_dir: str,
    ) -> None:
        """
        Save preprocessed arrays as .npy files to avoid reprocessing on every run.

        Files written:
            <processed_dir>/x_train.npy
            <processed_dir>/x_test.npy
            <processed_dir>/y_train.npy
            <processed_dir>/y_test.npy

        Args:
            x_train_p, x_test_p : Preprocessed image arrays (float32).
            y_train_p, y_test_p : One-hot label arrays (float32).
            processed_dir       : Destination directory.
        """
        os.makedirs(processed_dir, exist_ok=True)
        np.save(os.path.join(processed_dir, "x_train.npy"), x_train_p)
        np.save(os.path.join(processed_dir, "x_test.npy"),  x_test_p)
        np.save(os.path.join(processed_dir, "y_train.npy"), y_train_p)
        np.save(os.path.join(processed_dir, "y_test.npy"),  y_test_p)
        print(f"[Preprocessor] Processed arrays saved to: {processed_dir}")

    @staticmethod
    def load_processed_npy(processed_dir: str):
        """
        Load previously saved preprocessed .npy arrays.

        Args:
            processed_dir: Directory containing x_train.npy etc.

        Returns:
            tuple: (x_train_p, x_test_p, y_train_p, y_test_p)

        Raises:
            FileNotFoundError: If any expected .npy file is missing.
        """
        expected = ["x_train.npy", "x_test.npy", "y_train.npy", "y_test.npy"]
        for fname in expected:
            path = os.path.join(processed_dir, fname)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"[Preprocessor] Missing: {path}. "
                    f"Run Preprocessor.save_processed_npy() first."
                )

        x_train_p = np.load(os.path.join(processed_dir, "x_train.npy"))
        x_test_p  = np.load(os.path.join(processed_dir, "x_test.npy"))
        y_train_p = np.load(os.path.join(processed_dir, "y_train.npy"))
        y_test_p  = np.load(os.path.join(processed_dir, "y_test.npy"))

        print(f"[Preprocessor] Processed arrays loaded from: {processed_dir}")
        return x_train_p, x_test_p, y_train_p, y_test_p
