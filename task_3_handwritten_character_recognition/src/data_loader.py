"""
Task 3 — Handwritten Character Recognition
src/data_loader.py

Purpose:
    Responsible for loading the MNIST dataset using TensorFlow/Keras.
    Returns train/test splits as NumPy arrays, ready for preprocessing.

    This module also provides:
        - Integrity validation (shapes, dtypes, label range)
        - Dataset statistics as a structured dict
        - Saving raw arrays to data/raw/ as .npy files for offline use

    This module does NOT preprocess or transform the data — that is
    handled by preprocessing.py.

Usage:
    from src.data_loader import DataLoader
    (x_train, y_train), (x_test, y_test) = DataLoader.load_mnist()
    stats = DataLoader.get_data_stats(x_train, y_train, x_test, y_test)
    DataLoader.validate(x_train, y_train, x_test, y_test)
"""

import os
import sys


class DataLoader:
    """
    Handles loading and integrity-checking of the raw MNIST dataset.

    Methods:
        load_mnist()                    — Load MNIST via Keras (cached locally).
        validate(x_tr, y_tr, x_te, y_te) — Assert shapes, dtypes, label ranges.
        get_data_stats(...)             — Return a structured dict of statistics.
        get_data_info(...)              — Print a human-readable summary.
        save_raw_npy(...)               — Persist raw arrays to data/raw/.
        load_raw_npy(raw_dir)           — Load from previously saved .npy files.
    """

    # Expected MNIST constants — used in validation
    EXPECTED_TRAIN_N    = 60_000
    EXPECTED_TEST_N     = 10_000
    EXPECTED_IMG_SHAPE  = (28, 28)
    EXPECTED_N_CLASSES  = 10

    # ------------------------------------------------------------------ #
    #  Loading
    # ------------------------------------------------------------------ #

    @staticmethod
    def load_mnist():
        """
        Load the MNIST dataset using tf.keras.datasets.

        On the first call Keras downloads ~11 MB from the internet and
        caches it in  ~/.keras/datasets/mnist.npz.
        Every subsequent call reads the local cache — no internet needed.

        Returns:
            tuple: ((x_train, y_train), (x_test, y_test))
                x_train : np.ndarray  shape (60000, 28, 28)  dtype uint8
                y_train : np.ndarray  shape (60000,)          dtype uint8
                x_test  : np.ndarray  shape (10000, 28, 28)  dtype uint8
                y_test  : np.ndarray  shape (10000,)          dtype uint8

        Raises:
            ImportError : If TensorFlow is not installed.
            RuntimeError: If the dataset cannot be downloaded or loaded.
        """
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is not installed. Run: pip install tensorflow"
            ) from exc

        print("[DataLoader] Loading MNIST dataset via Keras ...")
        try:
            (x_train, y_train), (x_test, y_test) = (
                tf.keras.datasets.mnist.load_data()
            )
        except Exception as exc:
            raise RuntimeError(
                f"[DataLoader] Failed to load MNIST. "
                f"Check your internet connection on first run.\n"
                f"Original error: {exc}"
            ) from exc

        print(f"[DataLoader] [OK] Train - images {x_train.shape}, "
              f"labels {y_train.shape}")
        print(f"[DataLoader] [OK] Test  - images {x_test.shape}, "
              f"labels {y_test.shape}")

        return (x_train, y_train), (x_test, y_test)

    # ------------------------------------------------------------------ #
    #  Validation
    # ------------------------------------------------------------------ #

    @staticmethod
    def validate(x_train, y_train, x_test, y_test) -> bool:
        """
        Run integrity checks on the loaded raw arrays.

        Checks performed:
            1. Training set size   == 60 000
            2. Test set size       == 10 000
            3. Image spatial shape == (28, 28)
            4. Raw dtype           == uint8  (values 0-255)
            5. Label dtype         == uint8
            6. Label range         in [0, 9]
            7. Number of unique classes == 10

        Args:
            x_train, y_train : Raw training arrays.
            x_test,  y_test  : Raw test arrays.

        Returns:
            True if all checks pass.

        Raises:
            AssertionError: Describing the first failed check.
        """
        import numpy as np

        print("\n[DataLoader] Running validation checks ...")

        # --- Sample counts ---
        assert x_train.shape[0] == DataLoader.EXPECTED_TRAIN_N, (
            f"Expected {DataLoader.EXPECTED_TRAIN_N} training images, "
            f"got {x_train.shape[0]}"
        )
        assert x_test.shape[0] == DataLoader.EXPECTED_TEST_N, (
            f"Expected {DataLoader.EXPECTED_TEST_N} test images, "
            f"got {x_test.shape[0]}"
        )

        # --- Spatial dimensions ---
        assert x_train.shape[1:] == DataLoader.EXPECTED_IMG_SHAPE, (
            f"Expected image shape {DataLoader.EXPECTED_IMG_SHAPE}, "
            f"got {x_train.shape[1:]}"
        )
        assert x_test.shape[1:] == DataLoader.EXPECTED_IMG_SHAPE, (
            f"Expected image shape {DataLoader.EXPECTED_IMG_SHAPE}, "
            f"got {x_test.shape[1:]}"
        )

        # --- Data types ---
        assert x_train.dtype == np.uint8, (
            f"Expected x_train dtype uint8, got {x_train.dtype}"
        )
        assert y_train.dtype == np.uint8, (
            f"Expected y_train dtype uint8, got {y_train.dtype}"
        )

        # --- Label range ---
        assert y_train.min() == 0 and y_train.max() == 9, (
            f"Expected labels in [0,9], got [{y_train.min()},{y_train.max()}]"
        )
        assert y_test.min() == 0 and y_test.max() == 9, (
            f"Expected labels in [0,9], got [{y_test.min()},{y_test.max()}]"
        )

        # --- Number of classes ---
        n_classes = len(np.unique(y_train))
        assert n_classes == DataLoader.EXPECTED_N_CLASSES, (
            f"Expected {DataLoader.EXPECTED_N_CLASSES} classes, found {n_classes}"
        )

        print("[DataLoader] [PASS] All validation checks passed.\n")
        return True

    # ------------------------------------------------------------------ #
    #  Statistics
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_data_stats(x_train, y_train, x_test, y_test) -> dict:
        """
        Return a structured statistics dictionary for the raw dataset.

        Args:
            x_train, y_train : Raw training arrays.
            x_test,  y_test  : Raw test arrays.

        Returns:
            dict with keys:
                train_samples, test_samples, total_samples,
                image_shape, pixel_min, pixel_max, dtype_images,
                num_classes, train_class_counts, test_class_counts
        """
        import numpy as np

        train_unique, train_counts = np.unique(y_train, return_counts=True)
        test_unique,  test_counts  = np.unique(y_test,  return_counts=True)

        stats = {
            "train_samples"      : int(x_train.shape[0]),
            "test_samples"       : int(x_test.shape[0]),
            "total_samples"      : int(x_train.shape[0] + x_test.shape[0]),
            "image_shape_raw"    : tuple(x_train.shape[1:]),        # (28,28)
            "image_shape_cnn"    : tuple(x_train.shape[1:]) + (1,), # (28,28,1)
            "pixel_min_raw"      : int(x_train.min()),
            "pixel_max_raw"      : int(x_train.max()),
            "dtype_images"       : str(x_train.dtype),
            "dtype_labels"       : str(y_train.dtype),
            "num_classes"        : int(len(train_unique)),
            "class_names"        : [str(c) for c in train_unique.tolist()],
            "train_class_counts" : dict(zip(
                [str(c) for c in train_unique.tolist()],
                train_counts.tolist()
            )),
            "test_class_counts"  : dict(zip(
                [str(c) for c in test_unique.tolist()],
                test_counts.tolist()
            )),
            "normalization_status": "NOT YET - raw pixel values [0, 255]",
        }
        return stats

    # ------------------------------------------------------------------ #
    #  Human-readable print
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_data_info(x_train, y_train, x_test, y_test) -> None:
        """
        Print a formatted summary of dataset shapes and class distribution.

        Args:
            x_train, y_train : Raw training arrays.
            x_test,  y_test  : Raw test arrays.
        """
        import numpy as np

        stats = DataLoader.get_data_stats(x_train, y_train, x_test, y_test)

        print("\n" + "=" * 55)
        print("  MNIST Dataset Summary")
        print("=" * 55)
        print(f"  Training samples  : {stats['train_samples']:,}")
        print(f"  Test samples      : {stats['test_samples']:,}")
        print(f"  Total samples     : {stats['total_samples']:,}")
        print(f"  Raw image shape   : {stats['image_shape_raw']}")
        print(f"  CNN input shape   : {stats['image_shape_cnn']}")
        print(f"  Pixel range (raw) : [{stats['pixel_min_raw']}, {stats['pixel_max_raw']}]")
        print(f"  Image dtype       : {stats['dtype_images']}")
        print(f"  Label dtype       : {stats['dtype_labels']}")
        print(f"  Num classes       : {stats['num_classes']}")
        print(f"\n  Training class distribution:")
        for cls, cnt in stats["train_class_counts"].items():
            bar = "#" * (cnt // 500)
            print(f"    Digit {cls}: {cnt:,}  {bar}")
        print(f"\n  Normalisation     : {stats['normalization_status']}")
        print("=" * 55 + "\n")

    # ------------------------------------------------------------------ #
    #  Persistence — save / load raw .npy files
    # ------------------------------------------------------------------ #

    @staticmethod
    def save_raw_npy(x_train, y_train, x_test, y_test, raw_dir: str) -> None:
        """
        Save the raw MNIST arrays as .npy files for offline / fast access.

        Files written:
            <raw_dir>/x_train.npy
            <raw_dir>/y_train.npy
            <raw_dir>/x_test.npy
            <raw_dir>/y_test.npy

        Args:
            x_train, y_train : Raw training arrays.
            x_test,  y_test  : Raw test arrays.
            raw_dir          : Destination directory path.
        """
        import numpy as np

        os.makedirs(raw_dir, exist_ok=True)
        np.save(os.path.join(raw_dir, "x_train.npy"), x_train)
        np.save(os.path.join(raw_dir, "y_train.npy"), y_train)
        np.save(os.path.join(raw_dir, "x_test.npy"),  x_test)
        np.save(os.path.join(raw_dir, "y_test.npy"),  y_test)
        print(f"[DataLoader] Raw arrays saved to: {raw_dir}")

    @staticmethod
    def load_raw_npy(raw_dir: str):
        """
        Load previously saved raw .npy arrays from data/raw/.

        Useful when working offline after the first download.

        Args:
            raw_dir: Directory containing x_train.npy etc.

        Returns:
            tuple: ((x_train, y_train), (x_test, y_test))

        Raises:
            FileNotFoundError: If any expected .npy file is missing.
        """
        import numpy as np

        expected = ["x_train.npy", "y_train.npy", "x_test.npy", "y_test.npy"]
        for fname in expected:
            path = os.path.join(raw_dir, fname)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"[DataLoader] Missing file: {path}. "
                    f"Run DataLoader.save_raw_npy() first."
                )

        x_train = np.load(os.path.join(raw_dir, "x_train.npy"))
        y_train = np.load(os.path.join(raw_dir, "y_train.npy"))
        x_test  = np.load(os.path.join(raw_dir, "x_test.npy"))
        y_test  = np.load(os.path.join(raw_dir, "y_test.npy"))

        print(f"[DataLoader] Raw arrays loaded from: {raw_dir}")
        return (x_train, y_train), (x_test, y_test)


# ---------------------------------------------------------------------------
# Quick smoke-test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    (x_tr, y_tr), (x_te, y_te) = DataLoader.load_mnist()
    DataLoader.validate(x_tr, y_tr, x_te, y_te)
    DataLoader.get_data_info(x_tr, y_tr, x_te, y_te)
