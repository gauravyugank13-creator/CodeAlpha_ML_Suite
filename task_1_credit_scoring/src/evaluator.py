"""
Model evaluation module for Task 1: Credit Scoring Model.
Calculates performance classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC) 
and builds comparative summary tables.
"""
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from src.config import Config

class Evaluator:
    """
    Computes performance metrics and aggregates comparative model reports.
    """
    def __init__(self, config=Config):
        self.config = config

    def calculate_metrics(self, y_true, y_pred, y_probs) -> dict:
        """
        Calculates Accuracy, Precision, Recall, F1, and ROC-AUC metrics for predictions.
        """
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred)),
            "recall": float(recall_score(y_true, y_pred)),
            "f1_score": float(f1_score(y_true, y_pred)),
            "roc_auc": float(roc_auc_score(y_true, y_probs))
        }
        return metrics

    def generate_classification_report(self, y_true, y_pred) -> str:
        """
        Generates precision, recall, and F1 reports using scikit-learn.
        """
        return classification_report(y_true, y_pred)

    def generate_confusion_matrix(self, y_true, y_pred) -> list:
        """
        Computes the standard binary classification confusion matrix.
        """
        cm = confusion_matrix(y_true, y_pred)
        return cm.tolist()

    def generate_comparison_df(self, model_results: list) -> pd.DataFrame:
        """
        Aggregates results from multiple models and compiles a comparative DataFrame.
        Each entry in model_results should be a dict with: 'model_name' and the calculated metrics.
        """
        records = []
        for res in model_results:
            row = {
                "Model": res["model_name"],
                "Accuracy": res["metrics"]["accuracy"],
                "Precision": res["metrics"]["precision"],
                "Recall": res["metrics"]["recall"],
                "F1 Score": res["metrics"]["f1_score"],
                "ROC-AUC": res["metrics"]["roc_auc"]
            }
            records.append(row)
            
        df_comp = pd.DataFrame(records)
        return df_comp
