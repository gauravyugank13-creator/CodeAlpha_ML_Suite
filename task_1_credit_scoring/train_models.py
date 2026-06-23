"""
Baseline Model Development and Comparative Evaluation Pipeline for Task 1.
Trains Logistic Regression, Decision Tree, and Random Forest models on preprocessed 
data splits, generates confusion matrices and comparison charts, and saves checkpoints.
"""
import os
import sys
import matplotlib
# Use headless backend to prevent GUI dialogs on Windows
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix

# Add task directory to path to enable package imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.config import Config
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor, stratified_train_test_split
from src.model_builder import ModelBuilder
from src.trainer import Trainer
from src.evaluator import Evaluator

def main():
    print("==========================================================")
    print("  train_models.py - Baseline Model Development Pipeline   ")
    print("==========================================================")
    
    # 1. Load Dataset
    loader = DataLoader()
    try:
        df = loader.load_dataset()
        print(f"[Pipeline] Ingested raw dataset shape: {df.shape}")
    except Exception as e:
        print(f"[Error] Failed to load dataset: {e}")
        sys.exit(1)
        
    # 2. Stratified Train/Test Split
    X_train, X_test, y_train, y_test = stratified_train_test_split(df)
    print(f"[Pipeline] Stratified split shapes - Train: {X_train.shape}, Test: {X_test.shape}")
    
    # 3. Fit and Apply Preprocessing Pipeline
    preprocessor = Preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train, y_train)
    X_test_trans = preprocessor.transform(X_test)
    print(f"[Pipeline] Feature transformations finished. Matrix size: {X_train_trans.shape}")
    
    # 4. Instantiate Models
    models = {
        "logistic_regression": ModelBuilder.build_logistic_regression(),
        "decision_tree": ModelBuilder.build_decision_tree(),
        "random_forest": ModelBuilder.build_random_forest()
    }
    
    trainer = Trainer()
    evaluator = Evaluator()
    results_list = []
    roc_curves_data = {}
    
    os.makedirs(Config.PLOT_DIR, exist_ok=True)
    os.makedirs(Config.METRICS_DIR, exist_ok=True)
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    
    # 5. Train & Evaluate Models
    for name, model in models.items():
        print(f"\n--- Training Model: {name} ---")
        
        # Fit model on training split
        trainer.train_model(model, X_train_trans, y_train)
        
        # Save model checkpoint alongside the fitted preprocessor
        checkpoint_path = os.path.join(Config.MODEL_DIR, f"{name}.joblib")
        trainer.save_model(model, preprocessor, checkpoint_path)
        
        # Load back checkpoint as a sanity check validation
        loaded_checkpoint = trainer.load_model(checkpoint_path)
        assert loaded_checkpoint["model"] is not None, f"Checkpoint validation failed for {name}"
        
        # Predict on test split
        y_pred = model.predict(X_test_trans)
        y_probs = model.predict_proba(X_test_trans)[:, 1]
        
        # Compute stats
        metrics = evaluator.calculate_metrics(y_test, y_pred, y_probs)
        results_list.append({"model_name": name, "metrics": metrics})
        
        # Save ROC data for comparative charting later
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_curves_data[name] = {"fpr": fpr, "tpr": tpr, "auc": metrics["roc_auc"]}
        
        # Save model-specific classification text reports
        report_path = os.path.join(Config.METRICS_DIR, f"{name}_report.txt")
        class_rep = evaluator.generate_classification_report(y_test, y_pred)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"==========================================================\n")
            f.write(f"        Classification Report: {name}                     \n")
            f.write(f"==========================================================\n\n")
            f.write(f"Model Parameters:\n  {model}\n\n")
            f.write(f"Quantitative Evaluation Metrics:\n")
            f.write(f"  - Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"  - Precision: {metrics['precision']:.4f}\n")
            f.write(f"  - Recall:    {metrics['recall']:.4f}\n")
            f.write(f"  - F1 Score:  {metrics['f1_score']:.4f}\n")
            f.write(f"  - ROC-AUC:   {metrics['roc_auc']:.4f}\n\n")
            f.write("Scikit-Learn Classification Report:\n")
            f.write(class_rep)
        print(f"[Pipeline] Wrote metrics report file to: {report_path}")
        
        # Plot model-specific confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["No Risk (0)", "Risk (1)"],
                    yticklabels=["No Risk (0)", "Risk (1)"])
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted Risk Label")
        plt.ylabel("True Risk Label")
        plt.tight_layout()
        cm_plot_path = os.path.join(Config.PLOT_DIR, f"{name}_confusion_matrix.png")
        plt.savefig(cm_plot_path, dpi=150)
        plt.close()
        print(f"[Pipeline] Saved confusion matrix plot to: {cm_plot_path}")

    # 6. Generate Comparative Summary Table
    df_comp = evaluator.generate_comparison_df(results_list)
    comp_csv_path = os.path.join(Config.METRICS_DIR, "model_comparison.csv")
    df_comp.to_csv(comp_csv_path, index=False)
    print(f"\n[Pipeline] Saved summary model comparison table to: {comp_csv_path}")
    print(df_comp.to_string(index=False))
    
    # 7. Model Ranking calculations
    # Ranking hierarchy: (1) ROC-AUC, (2) F1 Score, (3) Recall, (4) Accuracy
    df_ranked = df_comp.sort_values(
        by=["ROC-AUC", "F1 Score", "Recall", "Accuracy"],
        ascending=False
    ).reset_index(drop=True)
    
    ranking_txt_path = os.path.join(Config.METRICS_DIR, "model_ranking.txt")
    with open(ranking_txt_path, "w", encoding="utf-8") as f:
        f.write("==========================================================\n")
        f.write("        Task 1 Credit Scoring Model Ranking Report        \n")
        f.write("==========================================================\n\n")
        f.write("Primary evaluation hierarchy:\n")
        f.write("  1. ROC-AUC (Core discriminative capacity)\n")
        f.write("  2. F1 Score (Balance of precision/recall)\n")
        f.write("  3. Recall (Minimizing defaults)\n")
        f.write("  4. Accuracy\n\n")
        
        f.write("Sorted Model Leaderboard:\n")
        for idx, row in df_ranked.iterrows():
            f.write(f"  Rank [{idx+1}] {row['Model']}\n")
            f.write(f"    - ROC-AUC:   {row['ROC-AUC']:.4f}\n")
            f.write(f"    - F1 Score:  {row['F1 Score']:.4f}\n")
            f.write(f"    - Recall:    {row['Recall']:.4f}\n")
            f.write(f"    - Accuracy:  {row['Accuracy']:.4f}\n\n")
            
        best_model = df_ranked.loc[0, "Model"]
        f.write(f"Conclusion: Selected Best Model is '{best_model}' based on evaluation metric rankings.\n")
    print(f"[Pipeline] Saved leaderboard model ranking to: {ranking_txt_path}")
    
    # 8. Plot ROC Curves Comparison
    roc_plot_path = os.path.join(Config.PLOT_DIR, "roc_curve_comparison.png")
    plt.figure(figsize=(7, 6))
    for name, data in roc_curves_data.items():
        plt.plot(data["fpr"], data["tpr"], label=f"{name} (AUC = {data['auc']:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess (AUC = 0.5000)")
    plt.title("Credit Risk Classifiers - ROC Curve Comparison")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(roc_plot_path, dpi=150)
    plt.close()
    print(f"[Pipeline] Saved comparative ROC curves chart to: {roc_plot_path}")
    
    # 9. Plot Bar Chart Metrics Comparison
    metrics_plot_path = os.path.join(Config.PLOT_DIR, "model_metrics_comparison.png")
    df_melted = df_comp.melt(id_vars="Model", var_name="Metric", value_name="Score")
    
    plt.figure(figsize=(9, 5))
    sns.barplot(x="Metric", y="Score", hue="Model", data=df_melted, palette="Set2")
    plt.title("Credit Risk Classifiers - Quantitative Metrics Comparison")
    plt.xlabel("Evaluation Metric")
    plt.ylabel("Score Value")
    plt.ylim(0, 1.05)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(metrics_plot_path, dpi=150)
    plt.close()
    print(f"[Pipeline] Saved comparative bar metrics chart to: {metrics_plot_path}")
    
    print("\n==========================================================")
    print(f"  [PASS] Training pipeline executed successfully!        ")
    print(f"  Best selected baseline model: {df_ranked.loc[0, 'Model']} ")
    print("==========================================================")

if __name__ == "__main__":
    main()
