"""
Evaluation module for Task 2: Emotion Recognition from Speech.
Computes metric reports, exports summary statistics, and plots confusion matrices.
"""
import os
import matplotlib
# Enforce Agg backend to prevent GUI loop errors on Windows
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from src.config import Config

class Evaluator:
    """
    Evaluates classification performance on audio test splits.
    """
    
    def calculate_metrics(self, y_true, y_pred) -> dict:
        """
        Computes accuracy, precision, recall, and F1-score averages.
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    def generate_confusion_matrix_plot(self, y_true, y_pred, save_path: str, 
                                       labels: list = Config.EMOTIONS) -> None:
        """
        Generates and saves a confusion matrix heatmap.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt="d", 
            cmap="Purples", 
            xticklabels=labels, 
            yticklabels=labels
        )
        plt.title("Confusion Matrix - Emotion Recognition")
        plt.xlabel("Predicted Emotion")
        plt.ylabel("True Emotion")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[Evaluator] Saved confusion matrix heatmap to: {save_path}")

    def save_classification_report(self, y_true, y_pred, save_path: str, 
                                   labels: list = Config.EMOTIONS) -> None:
        """
        Generates and saves the textual classification report.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        report = classification_report(y_true, y_pred, target_names=labels, zero_division=0)
        
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("==========================================================\n")
            f.write("    Task 2 Speech Emotion Recognition Evaluation Report  \n")
            f.write("==========================================================\n\n")
            f.write(report)
        print(f"[Evaluator] Saved classification report text to: {save_path}")
