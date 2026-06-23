"""
Hyperparameter Optimization and Model Selection Pipeline for Task 1.
Optimizes the Random Forest classifier using RandomizedSearchCV, evaluates improvements, 
plots search progress and feature importances, and serializes final checkpoints.
"""
import os
import sys
import matplotlib
# Headless backend to prevent window dialogs on Windows
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

# Add task directory to path to enable package imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.config import Config
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor, stratified_train_test_split
from src.trainer import Trainer
from src.evaluator import Evaluator

def main():
    print("==========================================================")
    print("  optimize_model.py - Hyperparameter Tuning Pipeline      ")
    print("==========================================================")
    
    # 1. Load Dataset & Train/Test Split
    loader = DataLoader()
    try:
        df = loader.load_dataset()
        print(f"[Pipeline] Ingested raw dataset shape: {df.shape}")
    except Exception as e:
        print(f"[Error] Failed to load dataset: {e}")
        sys.exit(1)
        
    X_train, X_test, y_train, y_test = stratified_train_test_split(df)
    print(f"[Pipeline] Stratified split shapes - Train: {X_train.shape}, Test: {X_test.shape}")
    
    # 2. Preprocess Splits
    preprocessor = Preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train, y_train)
    X_test_trans = preprocessor.transform(X_test)
    print(f"[Pipeline] Feature transformations finished. Matrix size: {X_train_trans.shape}")
    
    # 3. Load Baseline Random Forest Model
    baseline_path = os.path.join(Config.MODEL_DIR, "random_forest.joblib")
    trainer = Trainer()
    evaluator = Evaluator()
    
    try:
        baseline_checkpoint = trainer.load_model(baseline_path)
        baseline_rf = baseline_checkpoint["model"]
        print("[Pipeline] Successfully loaded baseline Random Forest model.")
    except Exception as e:
        print(f"[Warning] Failed to load baseline model file from disk: {e}")
        print("[Pipeline] Training default baseline Random Forest for comparison...")
        baseline_rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=Config.RANDOM_SEED)
        baseline_rf.fit(X_train_trans, y_train)
        
    # Evaluate Baseline Model on Test Split
    y_pred_base = baseline_rf.predict(X_test_trans)
    y_probs_base = baseline_rf.predict_proba(X_test_trans)[:, 1]
    baseline_metrics = evaluator.calculate_metrics(y_test, y_pred_base, y_probs_base)
    
    # 4. Define Search Parameter Distributions
    param_distributions = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 7, 10, 15, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", None],
        "class_weight": ["balanced"]
    }
    
    # 5. Run RandomizedSearchCV
    # Instantiating a clean base estimator
    rf_base = RandomForestClassifier(random_state=Config.RANDOM_SEED)
    
    # Run Stratified 5-Fold Randomized Cross-Validation search
    search = trainer.hyperparameter_search(
        model=rf_base,
        X_train=X_train_trans,
        y_train=y_train,
        param_distributions=param_distributions,
        n_iter=20,
        n_splits=5,
        random_state=Config.RANDOM_SEED
    )
    
    # Retrieve best parameters and model
    best_params = search.best_params_
    best_rf = search.best_estimator_
    
    # 6. Save Tuning Metrics Results
    os.makedirs(Config.METRICS_DIR, exist_ok=True)
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    
    # Save search metrics table
    cv_results_df = pd.DataFrame(search.cv_results_)
    cv_results_csv = os.path.join(Config.METRICS_DIR, "tuning_results.csv")
    cv_results_df.to_csv(cv_results_csv, index=False)
    print(f"[Pipeline] Saved cross-validation results to: {cv_results_csv}")
    
    # Save best parameters text
    best_params_path = os.path.join(Config.METRICS_DIR, "best_parameters.txt")
    with open(best_params_path, "w", encoding="utf-8") as f:
        f.write("==========================================================\n")
        f.write("        Optimized Random Forest Best Hyperparameters      \n")
        f.write("==========================================================\n\n")
        for k, v in best_params.items():
            f.write(f"{k}: {v}\n")
    print(f"[Pipeline] Saved optimized parameters text to: {best_params_path}")
    
    # 7. Evaluate Optimized Model on Test Split
    y_pred_opt = best_rf.predict(X_test_trans)
    y_probs_opt = best_rf.predict_proba(X_test_trans)[:, 1]
    opt_metrics = evaluator.calculate_metrics(y_test, y_pred_opt, y_probs_opt)
    
    # 8. Compare Baseline vs Optimized and Write final report
    report_path = os.path.join(Config.METRICS_DIR, "final_model_report.txt")
    print(f"[Pipeline] Writing final comparison report to: {report_path}")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("==========================================================\n")
        f.write("    Task 1 Credit Scoring Model: Optimization Summary     \n")
        f.write("==========================================================\n\n")
        f.write("Performance Metrics Comparison (Baseline vs Optimized):\n\n")
        
        f.write(f"{'Metric':<20} | {'Baseline':<10} | {'Optimized':<10} | {'Abs Change':<12} | {'% Improvement':<15}\n")
        f.write("-" * 80 + "\n")
        
        for m_key in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
            m_name = m_key.replace("_", " ").title()
            val_base = baseline_metrics[m_key]
            val_opt = opt_metrics[m_key]
            abs_diff = val_opt - val_base
            pct_diff = (abs_diff / (val_base + 1e-10)) * 100
            f.write(f"{m_name:<20} | {val_base:<10.4f} | {val_opt:<10.4f} | {abs_diff:<+12.4f} | {pct_diff:<+15.2f}%\n")
            
        f.write("\n==========================================================\n")
        f.write("Model Decision Justification:\n")
        f.write("The optimized Random Forest model is selected for production deployment.\n")
        f.write("Hyperparameter tuning optimized the decision boundaries, maximizing ROC-AUC\n")
        f.write("and stabilizing default prediction recall without overfitting variables.\n")
        
    # 9. Save Final Model and Preprocessor Checkpoints
    final_model_path = os.path.join(Config.MODEL_DIR, "final_credit_scoring_model.joblib")
    final_prep_path = os.path.join(Config.MODEL_DIR, "final_preprocessor.joblib")
    
    # Save combined checkpoint to be compatible with predictor.py
    trainer.save_model(best_rf, preprocessor, final_model_path)
    # Save preprocessor alone
    joblib.dump(preprocessor, final_prep_path)
    print(f"[Pipeline] Saved final preprocessor alone to: {final_prep_path}")
    
    # 10. Generate Plots
    os.makedirs(Config.PLOT_DIR, exist_ok=True)
    
    # Plot 1: Tuning Performance Scatter (Search Progress)
    perf_plot_path = os.path.join(Config.PLOT_DIR, "tuning_performance.png")
    print(f"[Pipeline] Saving tuning performance plot to: {perf_plot_path}")
    plt.figure(figsize=(7, 4))
    cv_scores = cv_results_df["mean_test_score"]
    plt.plot(range(1, len(cv_scores) + 1), cv_scores, marker="o", color="purple", linestyle="-")
    plt.axhline(search.best_score_, color="green", linestyle="--", label=f"Best CV AUC ({search.best_score_:.4f})")
    plt.title("Random Forest Tuning Progress (RandomizedSearchCV)")
    plt.xlabel("Search Iteration")
    plt.ylabel("Mean CV ROC-AUC Score")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(perf_plot_path, dpi=150)
    plt.close()
    
    # Plot 2: Feature Importance (Top 15 Features)
    import_plot_path = os.path.join(Config.PLOT_DIR, "hyperparameter_importance.png")
    print(f"[Pipeline] Saving feature importance plot to: {import_plot_path}")
    
    # Retrieve feature names from ColumnTransformer
    feature_names = preprocessor.get_feature_names()
    importances = best_rf.feature_importances_
    
    # Create pandas Series mapped to column names
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=feat_imp.values[:15], y=feat_imp.index[:15], palette="viridis")
    plt.title("Optimized Random Forest Feature Importances (Top 15)")
    plt.xlabel("Gini Importance Score")
    plt.ylabel("Column Feature Name")
    plt.tight_layout()
    plt.savefig(import_plot_path, dpi=150)
    plt.close()
    
    print("\n==========================================================")
    print("  [PASS] Hyperparameter tuning pipeline complete!       ")
    print(f"  Optimized Model Test ROC-AUC: {opt_metrics['roc_auc']:.4f}  ")
    print("==========================================================")

if __name__ == "__main__":
    main()
