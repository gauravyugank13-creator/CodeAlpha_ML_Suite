"""
Verification and Exploratory Data Analysis (EDA) execution pipeline for Task 1.
Loads, validates, and visualizes stats for the German Credit Dataset.
"""
import os
import sys
import matplotlib
# Force Agg backend to prevent interactive Tkinter GUI windows on Windows
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Add the project directory to path to enable package imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.config import Config
from src.data_loader import DataLoader

def run_validation():
    print("==========================================================")
    print("  validate_dataset.py - Dataset Validation & Initial EDA  ")
    print("==========================================================")
    
    # 1. Initialize Loader & Load Dataset
    loader = DataLoader()
    try:
        df = loader.load_dataset()
        print(f"[Validation] Dataset loaded successfully. Shape: {df.shape}")
    except Exception as e:
        print(f"[Error] Failed to load dataset: {e}")
        sys.exit(1)
        
    # 2. Run assertion verification checks
    try:
        loader.validate_dataset(df)
        print("[Validation] [PASS] All data integrity checks completed.")
    except AssertionError as err:
        print(f"[Error] Assertion Check Failed: {err}")
        sys.exit(1)
        
    # 3. Compute Summary Statistics
    summary = loader.get_dataset_summary(df)
    
    # Ensure results paths exist
    os.makedirs(Config.PLOT_DIR, exist_ok=True)
    os.makedirs(Config.METRICS_DIR, exist_ok=True)
    
    # 4. Generate Report Document
    report_path = os.path.join(Config.METRICS_DIR, "dataset_report.txt")
    print(f"[Validation] Writing dataset statistics report to: {report_path}")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("==========================================================\n")
        f.write("      German Credit Dataset Validation & Summary Stats    \n")
        f.write("==========================================================\n\n")
        f.write(f"Row Count (Samples): {summary['row_count']}\n")
        f.write(f"Column Count (Features): {summary['column_count']}\n")
        f.write(f"Duplicate Count: {summary['duplicate_count']}\n\n")
        
        f.write("Target Class Distribution (Risk):\n")
        for cls, count in summary["class_distribution"].items():
            pct = (count / summary["row_count"]) * 100
            f.write(f"  - {cls}: {count} ({pct:.2f}%)\n")
        f.write("\n")
        
        f.write("Missing Value Summary Counts:\n")
        for col, nulls in summary["missing_values"].items():
            pct = (nulls / summary["row_count"]) * 100
            f.write(f"  - {col}: {nulls} ({pct:.2f}%)\n")
        f.write("\n")
        
        f.write("Feature Datatype Breakdown:\n")
        for col, dtype in summary["datatype_map"].items():
            f.write(f"  - {col}: {dtype}\n")
            
    # 5. Plot 1: Target Class Distribution
    dist_plot_path = os.path.join(Config.PLOT_DIR, "class_distribution.png")
    print(f"[Validation] Saving class distribution plot to: {dist_plot_path}")
    plt.figure(figsize=(6, 4))
    sns.countplot(x=Config.TARGET_COLUMN, data=df, palette="muted")
    plt.title("Credit Risk Class Distribution")
    plt.xlabel("Risk Category")
    plt.ylabel("Sample Count")
    plt.tight_layout()
    plt.savefig(dist_plot_path, dpi=150)
    plt.close()
    
    # 6. Plot 2: Missing Values
    missing_plot_path = os.path.join(Config.PLOT_DIR, "missing_values.png")
    print(f"[Validation] Saving missing values plot to: {missing_plot_path}")
    plt.figure(figsize=(8, 4))
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if len(null_counts) > 0:
        sns.barplot(x=null_counts.index, y=null_counts.values, palette="rocket")
    else:
        plt.text(0.5, 0.5, "No missing values found", ha="center", va="center")
    plt.title("Missing Value Counts Per Feature")
    plt.xlabel("Features")
    plt.ylabel("Number of Missing Records")
    plt.tight_layout()
    plt.savefig(missing_plot_path, dpi=150)
    plt.close()
    
    # 7. Plot 3: Numerical Correlation Heatmap
    corr_plot_path = os.path.join(Config.PLOT_DIR, "correlation_heatmap.png")
    print(f"[Validation] Saving correlation heatmap plot to: {corr_plot_path}")
    plt.figure(figsize=(6, 5))
    numeric_df = df[Config.NUMERICAL_FEATURES]
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".3f", square=True)
    plt.title("Numeric Features Correlation Matrix")
    plt.tight_layout()
    plt.savefig(corr_plot_path, dpi=150)
    plt.close()
    
    print("==========================================================")
    print("  [PASS] Validation and EDA completed successfully!       ")
    print("==========================================================")

if __name__ == "__main__":
    run_validation()
