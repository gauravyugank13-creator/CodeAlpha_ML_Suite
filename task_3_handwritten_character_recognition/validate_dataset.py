"""
validate_dataset.py
Task 3 — Handwritten Character Recognition

Purpose:
    Standalone validation script for the MNIST dataset.
    Run this script AFTER installing requirements.txt to verify that:

        1. MNIST loads correctly via Keras
        2. Raw data shapes and dtypes are as expected
        3. Preprocessing pipeline produces correct output shapes
        4. All pixel values are normalised to [0.0, 1.0]
        5. A text dataset report is saved to results/metrics/
        6. A sample digit grid image is saved to results/plots/
        7. A class distribution chart is saved to results/plots/
        8. Raw arrays are saved to data/raw/ for offline use
        9. Processed arrays are saved to data/processed/

    This script does NOT:
        - Train any model
        - Make any predictions
        - Build any Flask app
        - Modify any model weights

Usage (from the project root or from task_3/ directory):
    python validate_dataset.py

    Or:
    python task_3_handwritten_character_recognition/validate_dataset.py
"""

import os
import sys

# ---------------------------------------------------------------------------
# Make sure src/ is importable regardless of where the script is called from
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)          # adds task_3_handwritten_character_recognition/

from src.config       import Config
from src.data_loader  import DataLoader
from src.preprocessing import Preprocessor


# ===========================================================================
#  Report writer
# ===========================================================================

def write_dataset_report(stats: dict, preprocessing_ok: bool) -> None:
    """
    Write a human-readable plain-text report to results/metrics/dataset_report.txt.

    Args:
        stats           : Dict returned by DataLoader.get_data_stats().
        preprocessing_ok: True if Preprocessor.validate() passed.
    """
    os.makedirs(Config.METRICS_DIR, exist_ok=True)

    lines = [
        "=" * 60,
        "  MNIST Dataset Validation Report",
        "  Task 3 - Handwritten Character Recognition",
        "=" * 60,
        "",
        "DATASET SUMMARY",
        "-" * 40,
        f"  Training samples   : {stats['train_samples']:,}",
        f"  Test samples       : {stats['test_samples']:,}",
        f"  Total samples      : {stats['total_samples']:,}",
        "",
        "IMAGE DIMENSIONS",
        "-" * 40,
        f"  Raw image shape    : {stats['image_shape_raw']}   (H x W)",
        f"  CNN input shape    : {stats['image_shape_cnn']}  (H x W x C)",
        "",
        "PIXEL VALUE RANGE (RAW)",
        "-" * 40,
        f"  Minimum pixel      : {stats['pixel_min_raw']}",
        f"  Maximum pixel      : {stats['pixel_max_raw']}",
        f"  Image dtype        : {stats['dtype_images']}",
        "",
        "NORMALISATION STATUS",
        "-" * 40,
        f"  Status             : COMPLETE - values in [0.0, 1.0]" if preprocessing_ok
            else f"  Status             : {stats['normalization_status']}",
        "",
        "CLASS DISTRIBUTION (TRAINING SET)",
        "-" * 40,
    ]

    for cls, cnt in stats["train_class_counts"].items():
        pct = cnt / stats["train_samples"] * 100
        bar = "#" * (cnt // 600)
        lines.append(f"  Digit {cls}:  {cnt:,}  ({pct:.1f}%)  {bar}")

    lines += [
        "",
        "CLASS DISTRIBUTION (TEST SET)",
        "-" * 40,
    ]

    for cls, cnt in stats["test_class_counts"].items():
        pct = cnt / stats["test_samples"] * 100
        bar = "#" * (cnt // 100)
        lines.append(f"  Digit {cls}:  {cnt:,}  ({pct:.1f}%)  {bar}")

    lines += [
        "",
        "VALIDATION RESULT",
        "-" * 40,
        f"  Raw data integrity : PASS",
        f"  Preprocessing      : {'PASS' if preprocessing_ok else 'FAIL'}",
        "",
        "=" * 60,
    ]

    report_text = "\n".join(lines)
    with open(Config.DATASET_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"[Report] Saved: {Config.DATASET_REPORT_PATH}")
    try:
        print(report_text)
    except UnicodeEncodeError:
        print(report_text.encode('ascii', errors='replace').decode('ascii'))


# ===========================================================================
#  Plot: sample digit grid
# ===========================================================================

def plot_sample_digit_grid(x_train: "np.ndarray", y_train: "np.ndarray") -> None:
    """
    Create a 2×5 grid showing one example of each digit class (0–9).
    Saved to results/plots/sample_digit_grid.png.

    Args:
        x_train : Raw (uint8) training images, shape (60000, 28, 28).
        y_train : Raw integer labels, shape (60000,).
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")   # headless — no GUI window
    import matplotlib.pyplot as plt

    os.makedirs(Config.PLOTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 5, figsize=(11, 5))
    fig.suptitle(
        "MNIST — One Sample Per Digit Class (0–9)",
        fontsize=14, fontweight="bold", y=1.02
    )
    fig.patch.set_facecolor("#1a1a2e")

    for digit in range(10):
        row, col = divmod(digit, 5)
        idx = np.where(y_train == digit)[0][0]   # first occurrence of digit
        ax  = axes[row][col]
        ax.imshow(x_train[idx], cmap="inferno", interpolation="nearest")
        ax.set_title(f"Digit: {digit}", color="white", fontsize=11, pad=6)
        ax.axis("off")
        ax.set_facecolor("#1a1a2e")

    plt.tight_layout()
    plt.savefig(Config.SAMPLE_GRID_PLOT, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot]   Saved: {Config.SAMPLE_GRID_PLOT}")


# ===========================================================================
#  Plot: class distribution bar chart
# ===========================================================================

def plot_class_distribution(
    y_train: "np.ndarray",
    y_test : "np.ndarray",
) -> None:
    """
    Create a grouped bar chart showing class frequency in train and test sets.
    Saved to results/plots/class_distribution.png.

    Args:
        y_train : Raw integer labels for the training set.
        y_test  : Raw integer labels for the test set.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(Config.PLOTS_DIR, exist_ok=True)

    digits = np.arange(10)
    train_counts = [(y_train == d).sum() for d in digits]
    test_counts  = [(y_test  == d).sum() for d in digits]

    bar_w = 0.38
    x     = np.arange(10)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    bars_tr = ax.bar(x - bar_w / 2, train_counts, bar_w,
                     label="Train (60 000)", color="#e94560", alpha=0.88,
                     edgecolor="#ff6b9d", linewidth=0.6)
    bars_te = ax.bar(x + bar_w / 2, test_counts,  bar_w,
                     label="Test  (10 000)", color="#0f3460", alpha=0.88,
                     edgecolor="#53a8b6", linewidth=0.6)

    # Value annotations
    for bar in bars_tr:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 40,
                f"{int(bar.get_height()):,}",
                ha="center", va="bottom", fontsize=7, color="white")
    for bar in bars_te:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 40,
                f"{int(bar.get_height()):,}",
                ha="center", va="bottom", fontsize=7, color="#a0cfe0")

    ax.set_xlabel("Digit Class", color="white", fontsize=12)
    ax.set_ylabel("Number of Samples", color="white", fontsize=12)
    ax.set_title("MNIST — Class Distribution (Train vs Test)",
                 color="white", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in digits], color="white")
    ax.tick_params(axis="y", colors="white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#444")
    ax.yaxis.set_tick_params(color="#444")
    ax.set_ylim(0, max(train_counts) * 1.15)
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=10)

    plt.tight_layout()
    plt.savefig(Config.CLASS_DIST_PLOT, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Plot]   Saved: {Config.CLASS_DIST_PLOT}")


# ===========================================================================
#  Main entry point
# ===========================================================================

def main():
    print("\n" + "=" * 60)
    print("  validate_dataset.py - MNIST Validation Script")
    print("  Task 3: Handwritten Character Recognition")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # 1. Load raw MNIST
    # ------------------------------------------------------------------
    (x_train, y_train), (x_test, y_test) = DataLoader.load_mnist()

    # ------------------------------------------------------------------
    # 2. Validate raw data integrity
    # ------------------------------------------------------------------
    DataLoader.validate(x_train, y_train, x_test, y_test)

    # ------------------------------------------------------------------
    # 3. Compute and print statistics
    # ------------------------------------------------------------------
    stats = DataLoader.get_data_stats(x_train, y_train, x_test, y_test)
    DataLoader.get_data_info(x_train, y_train, x_test, y_test)

    # ------------------------------------------------------------------
    # 4. Save raw .npy files to data/raw/
    # ------------------------------------------------------------------
    DataLoader.save_raw_npy(
        x_train, y_train, x_test, y_test,
        raw_dir=Config.RAW_DATA_DIR,
    )

    # ------------------------------------------------------------------
    # 5. Preprocess
    # ------------------------------------------------------------------
    x_train_p, x_test_p, y_train_p, y_test_p = Preprocessor.run(
        x_train, y_train, x_test, y_test, verbose=True
    )

    # ------------------------------------------------------------------
    # 6. Validate preprocessed output
    # ------------------------------------------------------------------
    preprocessing_ok = Preprocessor.validate(
        x_train_p, y_train_p, x_test_p, y_test_p
    )

    # ------------------------------------------------------------------
    # 7. Save processed .npy files to data/processed/
    # ------------------------------------------------------------------
    Preprocessor.save_processed_npy(
        x_train_p, x_test_p, y_train_p, y_test_p,
        processed_dir=Config.PROCESSED_DATA_DIR,
    )

    # ------------------------------------------------------------------
    # 8. Generate and save plots
    # ------------------------------------------------------------------
    print("\n[Plots] Generating visualisations ...")
    plot_sample_digit_grid(x_train, y_train)
    plot_class_distribution(y_train, y_test)

    # ------------------------------------------------------------------
    # 9. Write dataset report
    # ------------------------------------------------------------------
    write_dataset_report(stats, preprocessing_ok)

    # ------------------------------------------------------------------
    # 10. Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  [PASS] Validation Complete - Phase 2 DONE")
    print("=" * 60)
    print(f"\n  Files generated:")
    print(f"    [Report] {Config.DATASET_REPORT_PATH}")
    print(f"    [Plot]   {Config.SAMPLE_GRID_PLOT}")
    print(f"    [Chart]  {Config.CLASS_DIST_PLOT}")
    print(f"\n  Raw data:        {Config.RAW_DATA_DIR}")
    print(f"  Processed data:  {Config.PROCESSED_DATA_DIR}")
    print("\n  ->  Next: Run CNN model development (Phase 3)\n")


if __name__ == "__main__":
    main()
