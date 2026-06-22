"""
Task 3 — Handwritten Character Recognition
src/evaluator.py

Purpose:
    Evaluates a trained model on test data and generates visual reports.
    Produces:
        - Overall test accuracy and loss
        - Confusion matrix plot
        - Training history plot (loss and accuracy curves)
        - JSON metrics file

Usage:
    from src.evaluator import Evaluator
    Evaluator.evaluate(model, x_test, y_test)
    Evaluator.plot_training_history(history)
    Evaluator.plot_confusion_matrix(model, x_test, y_test)
"""

import os
import json


class Evaluator:
    """
    Evaluates model performance and generates visualisation artifacts.

    Methods:
        evaluate(model, x_test, y_test)          — Compute accuracy & loss.
        plot_training_history(history, save_path) — Plot loss/accuracy curves.
        plot_confusion_matrix(model, x_test, ...)  — Plot confusion matrix.
        save_metrics(metrics_dict, save_path)     — Save metrics as JSON.
    """

    @staticmethod
    def evaluate(model, x_test, y_test) -> dict:
        """
        Compute and print test loss and accuracy.

        Args:
            model:  Trained tf.keras.Model.
            x_test: Preprocessed test images.
            y_test: One-hot encoded test labels.

        Returns:
            dict: {'loss': float, 'accuracy': float}
        """
        print("[Evaluator] Evaluating model on test set...")
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print(f"  Test Loss     : {loss:.4f}")
        print(f"  Test Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
        return {"loss": round(loss, 4), "accuracy": round(accuracy, 4)}

    @staticmethod
    def plot_training_history(history, save_path: str = None) -> None:
        """
        Generate training and validation loss/accuracy curves.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "matplotlib is not installed. Run: pip install matplotlib"
            ) from e

        hist_dict = history.history if hasattr(history, "history") else history

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Accuracy plot
        axes[0].plot(hist_dict["accuracy"], label="Train Accuracy")
        axes[0].plot(hist_dict["val_accuracy"], label="Val Accuracy")
        axes[0].set_title("Model Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend()

        # Loss plot
        axes[1].plot(hist_dict["loss"], label="Train Loss")
        axes[1].plot(hist_dict["val_loss"], label="Val Loss")
        axes[1].set_title("Model Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150)
            print(f"[Evaluator] Training history plot saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    @staticmethod
    def plot_training_accuracy(history, save_path: str) -> None:
        """Plot training and validation accuracy curves separately."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "matplotlib is not installed. Run: pip install matplotlib"
            ) from e

        hist_dict = history.history if hasattr(history, "history") else history

        plt.figure(figsize=(8, 5))
        plt.plot(hist_dict["accuracy"], label="Train Accuracy", color="#e94560", linewidth=2)
        plt.plot(hist_dict["val_accuracy"], label="Val Accuracy", color="#0f3460", linewidth=2)
        plt.title("MNIST Lightweight CNN - Model Accuracy", fontsize=14, fontweight="bold")
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Accuracy", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[Evaluator] Training accuracy plot saved to: {save_path}")

    @staticmethod
    def plot_training_loss(history, save_path: str) -> None:
        """Plot training and validation loss curves separately."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "matplotlib is not installed. Run: pip install matplotlib"
            ) from e

        hist_dict = history.history if hasattr(history, "history") else history

        plt.figure(figsize=(8, 5))
        plt.plot(hist_dict["loss"], label="Train Loss", color="#e94560", linewidth=2)
        plt.plot(hist_dict["val_loss"], label="Val Loss", color="#0f3460", linewidth=2)
        plt.title("MNIST Lightweight CNN - Model Loss", fontsize=14, fontweight="bold")
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[Evaluator] Training loss plot saved to: {save_path}")

    @staticmethod
    def plot_confusion_matrix(
        model,
        x_test,
        y_test,
        class_names: list = None,
        save_path: str = None,
    ) -> None:
        """
        Generate and display a confusion matrix for the test set.
        """
        try:
            import numpy as np
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        except ImportError as e:
            raise ImportError(
                "Required library not installed. Run: pip install matplotlib scikit-learn"
            ) from e

        if class_names is None:
            class_names = [str(i) for i in range(10)]

        print("[Evaluator] Generating confusion matrix...")
        y_pred = model.predict(x_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)

        cm = confusion_matrix(y_true_classes, y_pred_classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

        fig, ax = plt.subplots(figsize=(10, 10))
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title("Confusion Matrix - MNIST Digit Classifier", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150)
            print(f"[Evaluator] Confusion matrix saved to: {save_path}")
        else:
            plt.show()

        plt.close()

    @staticmethod
    def generate_classification_report(
        model,
        x_test,
        y_test,
        save_path: str,
        class_names: list = None,
    ) -> str:
        """
        Generate classification report and write it to results/metrics/classification_report.txt.
        """
        try:
            import numpy as np
            from sklearn.metrics import classification_report
        except ImportError as e:
            raise ImportError(
                "scikit-learn is not installed. Run: pip install scikit-learn"
            ) from e

        if class_names is None:
            class_names = [str(i) for i in range(10)]

        print("[Evaluator] Generating classification report...")
        y_pred = model.predict(x_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)

        report = classification_report(y_true_classes, y_pred_classes, target_names=class_names)
        print("\n" + "=" * 60)
        print("  MNIST Classification Report")
        print("=" * 60)
        print(report)
        print("=" * 60 + "\n")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[Evaluator] Classification report saved to: {save_path}")
        return report

    @staticmethod
    def save_final_metrics(metrics_dict: dict, save_path: str) -> None:
        """
        Save final summary test loss and accuracy to a plain text file.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("=" * 45 + "\n")
            f.write("  MNIST CNN Classifier Final Metrics\n")
            f.write("=" * 45 + "\n")
            f.write(f"Test Loss:     {metrics_dict['loss']:.4f}\n")
            f.write(f"Test Accuracy: {metrics_dict['accuracy']:.4f} ({metrics_dict['accuracy']*100:.2f}%)\n")
            f.write("=" * 45 + "\n")
        print(f"[Evaluator] Final metrics plain text saved to: {save_path}")

    @staticmethod
    def save_metrics(metrics_dict: dict, save_path: str) -> None:
        """
        Persist a metrics dictionary as a JSON file.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(metrics_dict, f, indent=4)
        print(f"[Evaluator] Metrics saved to: {save_path}")


# ---------------------------------------------------------------------------
# TODO: Phase 4 — Evaluation
#   - Add a full_report() method that calls evaluate, plot_training_history,
#     plot_confusion_matrix, and save_metrics in one step.
# ---------------------------------------------------------------------------
