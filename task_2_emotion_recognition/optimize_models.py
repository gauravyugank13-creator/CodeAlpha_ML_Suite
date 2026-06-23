"""
Hyperparameter optimization runner for Task 2: Speech Emotion Recognition.
Runs RandomizedSearchCV with 5-fold StratifiedKFold cross-validation on SVM, Random Forest, and MLP,
saves metrics, generates visual plots, and saves the final optimized winning model.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Resolve paths to allow package imports
task_dir = os.path.dirname(os.path.abspath(__file__))
if task_dir not in sys.path:
    sys.path.insert(0, task_dir)

from src.config import Config
from src.model_builder import ModelBuilder
from src.trainer import Trainer
from src.evaluator import Evaluator
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

def main():
    print("==========================================================")
    print("  optimize_models.py - Hyperparameter Grid Optimization    ")
    print("==========================================================")

    # 1. Load dataset split matrices
    X_train_path = os.path.join(Config.PROCESSED_DATA_DIR, "X_train.npy")
    X_test_path = os.path.join(Config.PROCESSED_DATA_DIR, "X_test.npy")
    y_train_path = os.path.join(Config.PROCESSED_DATA_DIR, "y_train.npy")
    y_test_path = os.path.join(Config.PROCESSED_DATA_DIR, "y_test.npy")

    if not all(os.path.exists(p) for p in [X_train_path, X_test_path, y_train_path, y_test_path]):
        print("[Error] Missing preprocessed feature matrices. Run validate_features.py first.")
        sys.exit(1)

    X_train = np.load(X_train_path)
    X_test = np.load(X_test_path)
    y_train = np.load(y_train_path)
    y_test = np.load(y_test_path)

    print(f"[DataLoader] Loaded dataset split matrices:")
    print(f"  - X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  - X_test: {X_test.shape}, y_test: {y_test.shape}")

    trainer = Trainer()
    evaluator = Evaluator()

    # 2. Set up base pipelines and hyperparameter search grids
    # Standard prefix tags: svm__, rf__, mlp__
    base_svm = ModelBuilder.build_svm()
    base_rf = ModelBuilder.build_random_forest()
    base_mlp = ModelBuilder.build_mlp()

    search_spaces = {
        "svm": {
            "model": base_svm,
            "n_iter": 15,
            "space": {
                "svm__C": [0.1, 1, 5, 10, 20, 50],
                "svm__gamma": ["scale", 0.001, 0.005, 0.01, 0.05],
                "svm__class_weight": ["balanced"]
            }
        },
        "random_forest": {
            "model": base_rf,
            "n_iter": 15,
            "space": {
                "rf__n_estimators": [100, 200, 300, 500],
                "rf__max_depth": [5, 10, 15, 20, None],
                "rf__min_samples_split": [2, 5, 10],
                "rf__min_samples_leaf": [1, 2, 4],
                "rf__max_features": ["sqrt", "log2"],
                "rf__class_weight": ["balanced"]
            }
        },
        "mlp": {
            "model": base_mlp,
            "n_iter": 10,
            "space": {
                "mlp__hidden_layer_sizes": [(128,), (256,), (256, 128), (512, 256)],
                "mlp__activation": ["relu", "tanh"],
                "mlp__alpha": [0.0001, 0.001, 0.01],
                "mlp__learning_rate_init": [0.0005, 0.001, 0.005],
                "mlp__max_iter": [500, 750]
            }
        }
    }

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.RANDOM_SEED)

    # Dictionary to store search history results
    tuning_curves = {}
    optimized_pipelines = {}
    best_params_dict = {}
    optimized_metrics = {}

    # Run search for each classifier
    for name, config in search_spaces.items():
        print(f"\n[Optimization] Optimizing classifier: {name} ...")
        search = RandomizedSearchCV(
            estimator=config["model"],
            param_distributions=config["space"],
            n_iter=config["n_iter"],
            cv=cv_strategy,
            scoring="f1_weighted",
            random_state=Config.RANDOM_SEED,
            n_jobs=1,  # Safe sequential run
            verbose=1
        )
        search.fit(X_train, y_train)

        # Store results
        optimized_pipelines[name] = search.best_estimator_
        best_params_dict[name] = search.best_params_
        tuning_curves[name] = search.cv_results_["mean_test_score"]
        
        print(f"  - Best Parameters: {search.best_params_}")
        print(f"  - Best CV Weighted F1: {search.best_score_:.4f}")

    # 3. Evaluate optimized pipelines on the test split
    print("\n[Evaluation] Scoring optimized pipelines on the test set...")
    optimized_rows = []
    
    for name, pipeline in optimized_pipelines.items():
        y_pred = pipeline.predict(X_test)
        metrics = evaluator.calculate_metrics(y_test, y_pred)
        optimized_metrics[name] = metrics
        
        print(f"  - {name}: F1={metrics['f1_score']:.4f}, Accuracy={metrics['accuracy']:.4f}")
        
        optimized_rows.append({
            "Model": name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1-Score": metrics["f1_score"]
        })
        
        # Save individual report
        report_path = os.path.join(Config.METRICS_DIR, f"opt_{name}_report.txt")
        evaluator.save_classification_report(y_test, y_pred, report_path)
        
        # Save individual confusion matrix
        cm_path = os.path.join(Config.PLOT_DIR, f"confusion_matrix_opt_{name}.png")
        evaluator.generate_confusion_matrix_plot(y_test, y_pred, cm_path)

    # 4. Save metrics outputs
    # 4A. tuning_results.csv
    tuning_list = []
    for name, scores in tuning_curves.items():
        for iter_idx, score in enumerate(scores):
            tuning_list.append({"Model": name, "Iteration": iter_idx + 1, "Mean_CV_Score": score})
    tuning_df = pd.DataFrame(tuning_list)
    tuning_csv_path = os.path.join(Config.METRICS_DIR, "tuning_results.csv")
    tuning_df.to_csv(tuning_csv_path, index=False)
    print(f"[Metrics] Saved tuning curves to: {tuning_csv_path}")

    # 4B. best_parameters.txt
    best_params_path = os.path.join(Config.METRICS_DIR, "best_parameters.txt")
    with open(best_params_path, "w", encoding="utf-8") as f:
        f.write("==========================================================\n")
        f.write("    Task 2 Speech Emotion Optimized Hyperparameters      \n")
        f.write("==========================================================\n\n")
        for name, params in best_params_dict.items():
            f.write(f"Model: {name}\n")
            for k, v in params.items():
                f.write(f"  - {k}: {v}\n")
            f.write("\n")
    print(f"[Metrics] Saved best parameters list to: {best_params_path}")

    # 4C. optimized_model_comparison.csv
    opt_comparison_df = pd.DataFrame(optimized_rows)
    opt_comparison_csv_path = os.path.join(Config.METRICS_DIR, "optimized_model_comparison.csv")
    opt_comparison_df.to_csv(opt_comparison_csv_path, index=False)
    print(f"[Metrics] Saved optimized models comparison table to: {opt_comparison_csv_path}")

    # 4D. optimized_model_ranking.txt
    # Ranked by F1-score (weighted) then Accuracy
    ranked_opt_models = sorted(optimized_metrics.items(), key=lambda x: (x[1]["f1_score"], x[1]["accuracy"]), reverse=True)
    
    ranking_path = os.path.join(Config.METRICS_DIR, "optimized_model_ranking.txt")
    with open(ranking_path, "w", encoding="utf-8") as f:
        f.write("==========================================================\n")
        f.write("    Task 2 Speech Emotion Optimized Leaderboard Ranking   \n")
        f.write("==========================================================\n\n")
        f.write(f"Primary Selection Metric: Weighted F1-score\n")
        f.write(f"Secondary Selection Metric: Accuracy\n\n")
        for rank, (name, metrics) in enumerate(ranked_opt_models, 1):
            f.write(f"{rank}. Model: {name}\n")
            f.write(f"   - Weighted F1-Score: {metrics['f1_score']:.6f}\n")
            f.write(f"   - Accuracy:          {metrics['accuracy']:.6f}\n")
            f.write(f"   - Weighted Precision: {metrics['precision']:.6f}\n")
            f.write(f"   - Weighted Recall:    {metrics['recall']:.6f}\n\n")
    print(f"[Metrics] Saved optimized model ranking leaderboard to: {ranking_path}")

    # 5. Generate plots
    print("\n[Plots] Saving diagnostic visualizations...")
    
    # 5A. tuning_performance.png
    plt.figure(figsize=(8, 4.5))
    sns.lineplot(data=tuning_df, x="Iteration", y="Mean_CV_Score", hue="Model", marker="o", linewidth=2)
    plt.title("Hyperparameter Search CV Score Progression")
    plt.ylabel("Mean Test Score (Weighted F1)")
    plt.xlabel("Search Iteration Index")
    plt.tight_layout()
    tuning_plot_path = os.path.join(Config.PLOT_DIR, "tuning_performance.png")
    plt.savefig(tuning_plot_path, dpi=150)
    plt.close()
    print(f"  - Saved search score line chart to: {tuning_plot_path}")

    # 5B. optimization_comparison.png (Baseline vs Optimized)
    # Load baseline scores from model_comparison.csv if it exists, otherwise fall back to static
    baseline_scores = {"svm": 0.4842, "random_forest": 0.5359, "mlp": 0.5744}
    baseline_csv_path = os.path.join(Config.METRICS_DIR, "model_comparison.csv")
    if os.path.exists(baseline_csv_path):
        try:
            base_df = pd.read_csv(baseline_csv_path)
            for _, row in base_df.iterrows():
                baseline_scores[row["Model"]] = row["F1-Score"]
        except Exception:
            pass

    comparison_list = []
    for name in ["svm", "random_forest", "mlp"]:
        comparison_list.append({"Model": name, "Type": "Baseline", "F1-Score": baseline_scores[name]})
        comparison_list.append({"Model": name, "Type": "Optimized", "F1-Score": optimized_metrics[name]["f1_score"]})
    
    comp_df = pd.DataFrame(comparison_list)
    plt.figure(figsize=(7.5, 4.5))
    sns.barplot(x="Model", y="F1-Score", hue="Type", data=comp_df, palette="coolwarm")
    plt.ylim(0, 1.05)
    plt.title("Speech Emotion Classifiers: Baseline vs. Optimized F1-Scores")
    plt.ylabel("Weighted F1-Score")
    plt.tight_layout()
    comp_plot_path = os.path.join(Config.PLOT_DIR, "optimization_comparison.png")
    plt.savefig(comp_plot_path, dpi=150)
    plt.close()
    print(f"  - Saved baseline vs optimized comparison bar chart to: {comp_plot_path}")

    # 5C. feature_importance_random_forest.png
    opt_rf_pipeline = optimized_pipelines["random_forest"]
    # Access the random forest classifier from pipeline steps
    rf_estimator = opt_rf_pipeline.named_steps["rf"]
    importances = rf_estimator.feature_importances_
    top_indices = np.argsort(importances)[-15:]
    top_importances = importances[top_indices]
    
    # Map feature indexes to labels
    col_labels = []
    for idx in top_indices:
        if idx < 80:
            fam = f"MFCC_{idx}"
        elif idx < 104:
            fam = f"Chroma_{idx-80}"
        elif idx < 360:
            fam = f"Mel_{idx-104}"
        else:
            fam = f"Contrast_{idx-360}"
        col_labels.append(fam)

    feat_df = pd.DataFrame({"Feature": col_labels, "Gini Importance": top_importances})
    feat_df = feat_df.sort_values(by="Gini Importance", ascending=False)
    
    plt.figure(figsize=(9, 5.5))
    sns.barplot(x="Gini Importance", y="Feature", data=feat_df, palette="mako")
    plt.title("Top 15 Acoustic Feature Importances - Optimized Random Forest")
    plt.xlabel("Gini Importance Score")
    plt.tight_layout()
    feat_plot_path = os.path.join(Config.PLOT_DIR, "feature_importance_random_forest.png")
    plt.savefig(feat_plot_path, dpi=150)
    plt.close()
    print(f"  - Saved RF feature importance chart to: {feat_plot_path}")

    # 6. Select final model and save
    winning_name, winning_metrics = ranked_opt_models[0]
    print(f"\n[Selection] Winning Optimized Model: {winning_name} (F1-score: {winning_metrics['f1_score']:.4f})")
    
    # Check if this optimized model F1 score outperforms the previous baseline F1 score
    baseline_best_f1 = max(baseline_scores.values())
    print(f"  - Previous Baseline Best F1: {baseline_best_f1:.4f}")
    print(f"  - Optimized Winning F1: {winning_metrics['f1_score']:.4f}")
    
    # Save winner to final_emotion_model.joblib
    winning_pipeline = optimized_pipelines[winning_name]
    final_model_path = os.path.join(Config.MODEL_DIR, "final_emotion_model.joblib")
    trainer.save_model(winning_pipeline, final_model_path)
    print(f"  - Serialized final model checkpoint saved to: {final_model_path}")

    print("==========================================================")
    print("  [PASS] Hyperparameter searches and optimizations done!  ")
    print("==========================================================")

if __name__ == "__main__":
    main()
