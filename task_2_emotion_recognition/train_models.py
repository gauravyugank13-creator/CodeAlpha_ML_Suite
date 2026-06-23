"""
Model training, evaluation, comparison, and serialization coordinator for Task 2: Speech Emotion Recognition.
Trains SVM, Random Forest, and MLP models, outputs metrics and plots, and saves the final winner.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Resolve paths to allow package imports
task_dir = os.path.dirname(os.path.abspath(__file__))
if task_dir not in sys.path:
    sys.path.insert(0, task_dir)

from src.config import Config
from src.model_builder import ModelBuilder
from src.trainer import Trainer
from src.evaluator import Evaluator

def main():
    print("==========================================================")
    print("  train_models.py - Speech Emotion Classifier Training    ")
    print("==========================================================")

    # 1. Load preprocessed dataset matrices
    X_train_path = os.path.join(Config.PROCESSED_DATA_DIR, "X_train.npy")
    X_test_path = os.path.join(Config.PROCESSED_DATA_DIR, "X_test.npy")
    y_train_path = os.path.join(Config.PROCESSED_DATA_DIR, "y_train.npy")
    y_test_path = os.path.join(Config.PROCESSED_DATA_DIR, "y_test.npy")

    if not all(os.path.exists(p) for p in [X_train_path, X_test_path, y_train_path, y_test_path]):
        print("[Error] Missing preprocessed feature matrices. Please run validate_features.py first.")
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

    # 2. Build model pipelines
    print("[Pipeline] Assembling model pipelines with standard scaling...")
    models = {
        "svm": ModelBuilder.build_svm(),
        "random_forest": ModelBuilder.build_random_forest(),
        "mlp": ModelBuilder.build_mlp()
    }

    # Dict to collect results
    results = {}
    predictions = {}

    # 3. Train and evaluate each classifier
    for name, pipeline in models.items():
        print(f"\n[Training] Fitting classifier: {name} ...")
        # Train model
        trainer.train_model(pipeline, X_train, y_train)

        # Predict
        y_pred = pipeline.predict(X_test)
        predictions[name] = y_pred

        # Compute metrics
        metrics = evaluator.calculate_metrics(y_test, y_pred)
        results[name] = metrics
        print(f"[Evaluation] {name} metrics on test set:")
        print(f"  - Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  - Precision: {metrics['precision']:.4f}")
        print(f"  - Recall:    {metrics['recall']:.4f}")
        print(f"  - F1-score:  {metrics['f1_score']:.4f}")

        # Save individual classification report text
        report_filename = f"{name}_report.txt"
        report_path = os.path.join(Config.METRICS_DIR, report_filename)
        evaluator.save_classification_report(y_test, y_pred, report_path)

        # Save individual confusion matrix heatmap
        cm_filename = f"confusion_matrix_{name}.png"
        cm_path = os.path.join(Config.PLOT_DIR, cm_filename)
        evaluator.generate_confusion_matrix_plot(y_test, y_pred, cm_path)

        # Save model training run metadata
        metadata_filename = f"{name}_training_metadata.json"
        metadata_path = os.path.join(Config.METRICS_DIR, metadata_filename)
        metadata = {
            "model_name": name,
            "train_samples": len(y_train),
            "test_samples": len(y_test),
            "feature_dimension": X_train.shape[1],
            "hyperparameters": {k: str(v) for k, v in pipeline.get_params().items()},
            "evaluation_metrics": metrics
        }
        trainer.save_training_metadata(metadata, metadata_path)

    # 4. Generate comparison output tables
    print("\n[Comparison] Formulating comparative evaluation results...")
    comparison_rows = []
    for name, metrics in results.items():
        comparison_rows.append({
            "Model": name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1-Score": metrics["f1_score"]
        })
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_csv_path = os.path.join(Config.METRICS_DIR, "model_comparison.csv")
    comparison_df.to_csv(comparison_csv_path, index=False)
    print(f"  - Saved model comparison table to: {comparison_csv_path}")

    # 5. Determine model ranking and save ranking text file
    # Primary: F1-score (weighted), Secondary: Accuracy
    ranked_models = sorted(results.items(), key=lambda x: (x[1]["f1_score"], x[1]["accuracy"]), reverse=True)
    
    ranking_path = os.path.join(Config.METRICS_DIR, "model_ranking.txt")
    with open(ranking_path, "w", encoding="utf-8") as f:
        f.write("==========================================================\n")
        f.write("    Task 2 Speech Emotion Classifier Leaderboard Ranking  \n")
        f.write("==========================================================\n\n")
        f.write(f"Primary Selection Metric: Weighted F1-score\n")
        f.write(f"Secondary Selection Metric: Accuracy\n\n")
        
        for rank, (name, metrics) in enumerate(ranked_models, 1):
            f.write(f"{rank}. Model: {name}\n")
            f.write(f"   - Weighted F1-Score: {metrics['f1_score']:.6f}\n")
            f.write(f"   - Accuracy:          {metrics['accuracy']:.6f}\n")
            f.write(f"   - Weighted Precision: {metrics['precision']:.6f}\n")
            f.write(f"   - Weighted Recall:    {metrics['recall']:.6f}\n\n")
            
    print(f"  - Saved model ranking leaderboard to: {ranking_path}")

    # 6. Select winning model and save
    best_model_name, best_metrics = ranked_models[0]
    print(f"\n[Selection] Winning Model: {best_model_name} (F1-score: {best_metrics['f1_score']:.4f})")
    
    winning_pipeline = models[best_model_name]
    final_model_path = os.path.join(Config.MODEL_DIR, "final_emotion_model.joblib")
    trainer.save_model(winning_pipeline, final_model_path)
    print(f"  - Serialized winning model checkpoint saved to: {final_model_path}")

    # 7. Generate comparison visualizations: model_metrics_comparison.png
    print("\n[Plots] Generating comparison metrics chart...")
    metrics_plot_path = os.path.join(Config.PLOT_DIR, "model_metrics_comparison.png")
    
    # Restructure dataframe for plotting
    plot_df = pd.melt(comparison_df, id_vars="Model", var_name="Metric", value_name="Score")
    
    plt.figure(figsize=(9, 5))
    sns.set_theme(style="whitegrid")
    sns.barplot(x="Metric", y="Score", hue="Model", data=plot_df, palette="viridis")
    plt.ylim(0, 1.05)
    plt.title("Acoustic Model Comparison - Speech Emotion Classification")
    plt.ylabel("Performance Score")
    plt.xlabel("Evaluation Metric")
    plt.legend(title="Classifier", loc="lower right")
    plt.tight_layout()
    plt.savefig(metrics_plot_path, dpi=150)
    plt.close()
    print(f"  - Saved comparative metrics bar chart to: {metrics_plot_path}")

    print("==========================================================")
    print("  [PASS] Speech emotion model training complete!          ")
    print("==========================================================")

if __name__ == "__main__":
    main()
