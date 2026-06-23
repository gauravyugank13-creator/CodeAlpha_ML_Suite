"""
Configuration module for Task 1: Credit Scoring Model.
Defines paths, hyperparameters, split settings, features, and seeds.
"""
import os

# Base directory of the Task 1 module
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Config:
    # Random Seed for reproducibility
    RANDOM_SEED = 42
    
    # Train / Test split options
    TEST_SIZE = 0.2
    
    # Data Paths
    DATA_DIR = os.path.join(BASE_DIR, "data")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    
    # Model Storage Paths
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_credit_model.joblib")
    
    # Results Directories
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    PLOT_DIR = os.path.join(RESULTS_DIR, "plots")
    METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
    PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")
    
    # Feature Names (defined based on German Credit dataset schema)
    NUMERICAL_FEATURES = ["Age", "Credit amount", "Duration"]
    CATEGORICAL_FEATURES = ["Sex", "Job", "Housing", "Saving accounts", "Checking account", "Purpose"]
    TARGET_COLUMN = "Risk"
    
    # Dataset URL
    DATASET_URL = "https://raw.githubusercontent.com/ziadasal/Credit-Risk-Assessment/main/german_credit_data.csv"
