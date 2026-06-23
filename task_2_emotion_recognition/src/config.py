"""
Configuration module for Task 2: Emotion Recognition from Speech.
Defines paths, random seeds, audio properties, and feature extraction constants.
"""
import os

# Base directory of the Task 2 module
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
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_emotion_model.joblib")
    
    # Results Directories
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    PLOT_DIR = os.path.join(RESULTS_DIR, "plots")
    METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
    PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")
    
    # Audio Feature Parameters
    SAMPLE_RATE = 22050         # Standard sampling rate in Hz
    DURATION = 3.0              # Max duration to trim/pad audio to (in seconds)
    
    # Feature Dimensions
    N_MFCC = 40                 # Number of MFCC coefficients to extract
    N_MELS = 128                # Number of Mel bands
    N_CHROMA = 12               # Number of Chroma bins
    
    # Emotion Classes (Standard RAVDESS mapping example)
    EMOTIONS = [
        "neutral", 
        "calm", 
        "happy", 
        "sad", 
        "angry", 
        "fearful", 
        "disgust", 
        "surprised"
    ]
