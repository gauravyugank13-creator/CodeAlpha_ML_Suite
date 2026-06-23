"""
Verification script for Preprocessing and Feature Engineering Pipelines.
Applies transformations, audits data integrity, generates summary reports, 
and outputs visual distribution and correlation plots.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
# Use headless backend to prevent window popup errors on Windows systems
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root directory to path to enable package import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.config import Config
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor, stratified_train_test_split

def run_preprocessing_validation():
    print("==========================================================")
    print("  validate_preprocessing.py - Pipeline Verification Checks  ")
    print("==========================================================")
    
    # 1. Load Dataset
    loader = DataLoader()
    try:
        df = loader.load_dataset()
        print(f"[Verification] Raw dataset loaded. Shape: {df.shape}")
    except Exception as e:
        print(f"[Error] Failed to load dataset: {e}")
        sys.exit(1)
        
    # Store original statistics
    original_shape = df.shape
    missing_before = df.isnull().sum().to_dict()
    
    # 2. Stratified Train/Test Split
    try:
        X_train, X_test, y_train, y_test = stratified_train_test_split(df)
        print(f"[Verification] Stratified Train split shape: {X_train.shape}, Test split shape: {X_test.shape}")
        
        # Verify stratification balance
        train_bad_pct = (y_train == 1).mean() * 100
        test_bad_pct = (y_test == 1).mean() * 100
        print(f"[Verification] Train target 'bad' risk: {train_bad_pct:.2f}%, Test 'bad' risk: {test_bad_pct:.2f}%")
        
        # Ensure splits are correct sizes
        assert X_train.shape[0] == 800, "Expected 800 train samples."
        assert X_test.shape[0] == 200, "Expected 200 test samples."
        assert abs(train_bad_pct - test_bad_pct) < 1e-2, "Stratification mismatch detected."
        print("[Verification] [PASS] Train/test split stratified successfully.")
    except Exception as e:
        print(f"[Error] Train/Test Split failed: {e}")
        sys.exit(1)
        
    # 3. Fit and Transform Preprocessing Pipeline
    preprocessor = Preprocessor()
    try:
        # Fit on training data and transform train split
        X_train_trans = preprocessor.fit_transform(X_train, y_train)
        # Transform test data (reusing fitted pipeline)
        X_test_trans = preprocessor.transform(X_test)
        
        print(f"[Verification] Transformed matrix shapes - Train: {X_train_trans.shape}, Test: {X_test_trans.shape}")
    except Exception as e:
        print(f"[Error] Pipeline transformation failed: {e}")
        sys.exit(1)
        
    # 4. Run Pipeline Assertions and Checks
    # - Ensure no missing values (NaN) remain in processed outputs
    assert not np.isnan(X_train_trans).any(), "Assertion Error: Transformed training data contains NaNs."
    assert not np.isnan(X_test_trans).any(), "Assertion Error: Transformed test data contains NaNs."
    print("[Verification] [PASS] Imputation successful. Zero NaN values remain.")
    
    # - Verify numerical scaling works (mean ~ 0, std ~ 1)
    # The first 5 columns correspond to numerical variables:
    # ["Age", "Credit amount", "Duration", "credit_per_month", "credit_to_age_ratio"]
    num_means = X_train_trans[:, :5].mean(axis=0)
    num_stds = X_train_trans[:, :5].std(axis=0)
    print(f"[Verification] Scaled features mean: {num_means}, std: {num_stds}")
    for i, (m, s) in enumerate(zip(num_means, num_stds)):
        assert abs(m) < 1e-1, f"Assertion Error: Feature index {i} mean is not close to 0 ({m})"
        assert abs(s - 1.0) < 1e-1, f"Assertion Error: Feature index {i} std is not close to 1 ({s})"
    print("[Verification] [PASS] Numerical scaling verification passed.")
    
    # - Check One-hot encoded column count
    feature_names = preprocessor.get_feature_names()
    print(f"[Verification] Generated feature count: {len(feature_names)}")
    
    # 5. Generate Text Report
    os.makedirs(Config.METRICS_DIR, exist_ok=True)
    report_path = os.path.join(Config.METRICS_DIR, "preprocessing_report.txt")
    print(f"[Verification] Writing preprocessing metrics report to: {report_path}")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("==========================================================\n")
        f.write("  German Credit Preprocessing & Feature Engineering Report  \n")
        f.write("==========================================================\n\n")
        f.write(f"Original Raw Shape: {original_shape}\n")
        f.write(f"Processed Train Shape: {X_train_trans.shape}\n")
        f.write(f"Processed Test Shape: {X_test_trans.shape}\n")
        f.write(f"Total Encoded/Transformed Columns: {X_train_trans.shape[1]}\n\n")
        
        f.write("Missing Value Counts Before Imputation:\n")
        for col, nulls in missing_before.items():
            f.write(f"  - {col}: {nulls}\n")
        f.write("\n")
        
        f.write("Missing Value Counts After Preprocessing Pipeline:\n")
        f.write(f"  - X_train transformed NaN count: {np.isnan(X_train_trans).sum()}\n")
        f.write(f"  - X_test transformed NaN count: {np.isnan(X_test_trans).sum()}\n\n")
        
        f.write("Engineered Features Created:\n")
        f.write("  1. credit_per_month (Credit amount / Duration)\n")
        f.write("  2. age_group (young, adult, middle_age, senior)\n")
        f.write("  3. credit_to_age_ratio (Credit amount / Age)\n\n")
        
        f.write("Final Feature Columns Generated:\n")
        for idx, col_name in enumerate(feature_names):
            f.write(f"  [{idx:02d}] {col_name}\n")
            
    # 6. Plotting Visualizations
    os.makedirs(Config.PLOT_DIR, exist_ok=True)
    
    # Plot 1: Age Distribution Histogram
    age_plot_path = os.path.join(Config.PLOT_DIR, "age_distribution.png")
    print(f"[Verification] Saving age distribution plot to: {age_plot_path}")
    plt.figure(figsize=(6, 4))
    sns.histplot(df["Age"], kde=True, color="skyblue")
    plt.title("Applicant Age Distribution")
    plt.xlabel("Age (years)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(age_plot_path, dpi=150)
    plt.close()
    
    # Plot 2: Credit Amount Distribution Histogram
    credit_plot_path = os.path.join(Config.PLOT_DIR, "credit_amount_distribution.png")
    print(f"[Verification] Saving credit amount distribution plot to: {credit_plot_path}")
    plt.figure(figsize=(6, 4))
    sns.histplot(df["Credit amount"], kde=True, color="salmon")
    plt.title("Requested Credit Amount Distribution")
    plt.xlabel("Credit Amount (DM)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(credit_plot_path, dpi=150)
    plt.close()
    
    # Plot 3: Numerical and Engineered Features Correlation matrix
    corr_plot_path = os.path.join(Config.PLOT_DIR, "engineered_feature_correlations.png")
    print(f"[Verification] Saving engineered feature correlations to: {corr_plot_path}")
    
    # Extract numerical features + run engineering steps on entire df to plot
    from src.feature_engineering import FeatureEngineer
    fe = FeatureEngineer()
    df_engineered = fe.transform(df)
    
    numeric_cols = ["Age", "Credit amount", "Duration", "credit_per_month", "credit_to_age_ratio"]
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_engineered[numeric_cols].corr(), annot=True, cmap="mako", fmt=".3f", square=True)
    plt.title("Correlation Heatmap (Numerical & Engineered Features)")
    plt.tight_layout()
    plt.savefig(corr_plot_path, dpi=150)
    plt.close()
    
    print("==========================================================")
    print("  [PASS] Preprocessing validation executed successfully!  ")
    print("==========================================================")

if __name__ == "__main__":
    run_preprocessing_validation()
