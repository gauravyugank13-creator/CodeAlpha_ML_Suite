"""
validate_prediction_pipeline.py
Task 3 — Handwritten Character Recognition

Purpose:
    Validates the entire handwritten character prediction pipeline.
    Does NOT retrain the model.
    Steps:
        1. Generates realistic custom user test images (.png) from MNIST test samples
           scaled up to 112x112 and inverted to black strokes on a white background.
        2. Loads the model from models/best_mnist_model.keras.
        3. Preprocesses the images (testing image_utils.preprocess_image and auto-inversion).
        4. Invokes predict_image(), predict_array(), and predict_with_confidence().
        5. Asserts shapes, probability sum, and interface consistency.
        6. Writes predictions and confidence scores to results/predictions/prediction_examples.txt.
"""

import os
import sys
import numpy as np
import cv2

# Ensure task directory is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from src.config import Config
from src.predictor import Predictor
from src.image_utils import preprocess_image


def generate_test_images():
    """Extract sample digits from MNIST test set and write to results/predictions/ as user PNGs."""
    from src.data_loader import DataLoader

    print("[Validation] Extracting sample digits from raw dataset...")
    # Attempt load
    try:
        (x_train, y_train), (x_test, y_test) = DataLoader.load_mnist()
    except Exception:
        # Load from raw npy if files exist offline
        (x_train, y_train), (x_test, y_test) = DataLoader.load_raw_npy(Config.RAW_DATA_DIR)

    os.makedirs(Config.PREDICTIONS_DIR, exist_ok=True)

    samples = []
    # Find one sample for digits 0, 1, 2, 3, 4
    for digit in range(5):
        idx = np.where(y_test == digit)[0][0]
        img = x_test[idx]
        filename = f"sample_digit_{digit}.png"
        filepath = os.path.join(Config.PREDICTIONS_DIR, filename)

        # Scale image to 112x112 to represent a typical custom user image
        img_resized = cv2.resize(img, (112, 112), interpolation=cv2.INTER_NEAREST)

        # Invert to black-on-white to verify the preprocessor's auto-inversion logic
        img_inverted = cv2.bitwise_not(img_resized)

        cv2.imwrite(filepath, img_inverted)
        print(f"[Validation] Saved test image: {filepath}")
        samples.append((filename, filepath, digit))

    return samples


def main():
    model_path = os.path.join(Config.MODELS_DIR, "best_mnist_model.keras")

    print("\n" + "=" * 60)
    print("  validate_prediction_pipeline.py - Prediction Verification")
    print("=" * 60 + "\n")

    # 1. Generate sample PNGs
    samples = generate_test_images()

    # 2. Load model
    print("\n[Validation] 1. Loading model...")
    model = Predictor.load_model(model_path)

    # 3. Predict and verify
    print("\n[Validation] 2. Running predictions and verifying pipeline...")

    report_lines = [
        "=" * 50,
        "  MNIST Prediction Pipeline Examples Report",
        "=" * 50,
        "",
        f"{'Filename':<30} | {'Prediction':<10} | {'Confidence':<10}",
        "-" * 55,
    ]

    for filename, filepath, expected_digit in samples:
        # Preprocess using image_utils (uses "new" method by default)
        img_array = preprocess_image(filepath)

        # Check preprocessing shape & values
        assert img_array.shape == (1, 28, 28, 1), f"Incorrect shape: {img_array.shape}"
        assert (
            img_array.min() >= 0.0 and img_array.max() <= 1.0
        ), f"Incorrect pixel range: [{img_array.min()}, {img_array.max()}]"

        # Predict using all required interfaces
        digit_arr, conf_arr, probs_arr = Predictor.predict_array(model, img_array)
        digit_img, conf_img, probs_img = Predictor.predict_image(model, filepath)
        digit_conf, conf_conf, probs_conf = Predictor.predict_with_confidence(model, filepath)

        # Validate interface consistency and correctness
        assert digit_arr == digit_img == digit_conf, "Inconsistent predicted digit across interfaces"
        assert digit_arr == expected_digit, f"Incorrect prediction on {filename}: expected {expected_digit}, got {digit_arr}"
        assert np.isclose(conf_arr, conf_img) and np.isclose(
            conf_arr, conf_conf
        ), "Inconsistent confidence score"
        assert np.isclose(
            sum(probs_arr), 1.0, atol=1e-3
        ), f"Softmax probabilities do not sum to 1: {sum(probs_arr)}"

        print(
            f"  [OK] {filename} -> Prediction: {digit_img} (Confidence: {conf_img*100:.2f}%)"
        )
        report_lines.append(f"{filename:<30} | {digit_img:<10} | {conf_img*100:.2f}%")

    # 4. Save examples report
    report_path = os.path.join(Config.PREDICTIONS_DIR, "prediction_examples.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\n[Validation] 3. Prediction examples report saved to: {report_path}")

    # 5. Generate before/after visual comparison plot
    print("\n[Validation] 4. Generating before/after comparison plot...")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(5, 3, figsize=(10, 12))
        
        for i, (filename, filepath, expected_digit) in enumerate(samples):
            # Load raw image (user input drawing style, e.g. black digit on white background)
            raw_img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            
            # Get preprocessed inputs (excluding batch/channel dimensions for display)
            img_old = preprocess_image(filepath, method="old")[0, :, :, 0]
            img_new = preprocess_image(filepath, method="new")[0, :, :, 0]
            
            # Column 1: Original user custom style (usually black ink on white page)
            axes[i, 0].imshow(raw_img, cmap="gray")
            axes[i, 0].set_title(f"Sample {expected_digit} (User Input)")
            axes[i, 0].axis("off")
            
            # Column 2: Old simple direct resizing
            axes[i, 1].imshow(img_old, cmap="gray")
            axes[i, 1].set_title("Old (Distorted & Edge-pinned)")
            axes[i, 1].axis("off")
            
            # Column 3: New aspect-preserved, bounding-box cropped and COM centered
            axes[i, 2].imshow(img_new, cmap="gray")
            axes[i, 2].set_title("New (Aspect Kept & Centered)")
            axes[i, 2].axis("off")
            
        plt.tight_layout()
        comparison_plot_path = os.path.join(Config.PLOTS_DIR, "preprocessing_comparison.png")
        os.makedirs(os.path.dirname(comparison_plot_path), exist_ok=True)
        plt.savefig(comparison_plot_path, dpi=150)
        plt.close()
        print(f"[Validation] Preprocessing comparison plot saved to: {comparison_plot_path}")
    except Exception as e:
        print(f"[Validation Warning] Failed to generate comparison plot: {e}")

    print("\n" + "=" * 60)
    print("  [PASS] Validation Complete - All Checks Passed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
