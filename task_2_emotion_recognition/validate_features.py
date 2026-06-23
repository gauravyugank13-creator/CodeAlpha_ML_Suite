"""
Audio preprocessing and feature extraction pipeline verification script for Task 2.
Ingests RAVDESS files, applies preprocessing, extracts 374 features, splits 80/20 train/test,
audits NaNs/Infs, plots profiling charts, and writes a metrics report.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
# Headless matplotlib backend for Windows console runs
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Resolve paths to allow package imports
task_dir = os.path.dirname(os.path.abspath(__file__))
if task_dir not in sys.path:
    sys.path.insert(0, task_dir)

from src.config import Config
from src.data_loader import AudioDataLoader
from src.audio_preprocessing import trim_silence, normalize_amplitude, pad_or_truncate
from src.feature_extraction import extract_combined_features

def run_feature_extraction_pipeline():
    print("==========================================================")
    print("  validate_features.py - Feature Ingestion & Auditing     ")
    print("==========================================================")
    
    loader = AudioDataLoader()
    catalog = loader.scan_audio_files()
    
    if len(catalog) == 0:
        print("[Error] No WAV audio files cataloged. Please run validate_dataset.py first.")
        sys.exit(1)
        
    print(f"[Pipeline] Loaded catalog containing {len(catalog)} audio records.")
    
    # Lists to store features, labels, and metadata
    all_features = []
    all_labels = []
    metadata = []
    
    # Track preprocessing metrics
    original_durations = []
    trimmed_durations = []
    scaling_factors = []
    
    print("[Pipeline] Running preprocessing and feature extraction (this will take 1-2 minutes)...")
    
    import soundfile as sf
    total_files = len(catalog)
    
    for idx, item in enumerate(catalog):
        file_path = item["file_path"]
        emotion = item["emotion"]
        actor_id = item["actor_id"]
        
        # Log progress every 200 files
        if (idx + 1) % 200 == 0 or (idx + 1) == total_files:
            print(f"  Processed {idx + 1}/{total_files} clips...")
            
        try:
            # 1. Track original properties
            info = sf.info(file_path)
            orig_dur = info.duration
            original_durations.append(orig_dur)
            
            # 2. Ingest waveform
            waveform, sr = loader.load_audio_waveform(file_path)
            
            # 3. Apply preprocessing
            # Trim margins
            trimmed = trim_silence(waveform)
            trim_dur = len(trimmed) / sr
            trimmed_durations.append(trim_dur)
            
            # Normalise amplitude
            max_val = np.max(np.abs(trimmed))
            scaling_factors.append(1.0 / max_val if max_val > 0 else 1.0)
            normalized = normalize_amplitude(trimmed)
            
            # Pad / Truncate to exact 3.0s duration
            processed = pad_or_truncate(normalized, target_sr=sr)
            
            # 4. Extract 374 combined features (means and stds)
            features = extract_combined_features(processed, sr)
            
            # Append features and labels
            all_features.append(features)
            all_labels.append(emotion)
            
            # Add metadata dict
            metadata.append({
                "file_name": item["file_name"],
                "emotion": emotion,
                "actor_id": actor_id,
                "gender": item["gender"],
                "intensity": item["intensity"]
            })
            
        except Exception as e:
            print(f"[Error] Failed to process clip: {file_path}. Error: {e}")
            sys.exit(1)
            
    # Convert lists to numpy arrays
    X = np.array(all_features)
    y = np.array(all_labels)
    
    print(f"[Pipeline] Extraction finished. Matrix shape: {X.shape}, Labels shape: {y.shape}")
    
    # 5. Perform validation and audits
    print("[Pipeline] Running data integrity validation audits...")
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    
    assert nan_count == 0, f"Assertion Error: Matrix contains {nan_count} NaN values."
    assert inf_count == 0, f"Assertion Error: Matrix contains {inf_count} infinite values."
    assert X.shape[1] == 374, f"Assertion Error: Dimension mismatch, expected 374 features, got {X.shape[1]}"
    print("[PASS] Zero NaNs or Infs detected. Dimensions match 374 feature variables.")
    
    # 6. Save master processed outputs
    os.makedirs(Config.PROCESSED_DATA_DIR, exist_ok=True)
    
    features_npy_path = os.path.join(Config.PROCESSED_DATA_DIR, "features.npy")
    labels_npy_path = os.path.join(Config.PROCESSED_DATA_DIR, "labels.npy")
    np.save(features_npy_path, X)
    np.save(labels_npy_path, y)
    print(f"[Pipeline] Saved master features.npy and labels.npy to: {Config.PROCESSED_DATA_DIR}")
    
    # Formulate features.csv
    features_csv_path = os.path.join(Config.PROCESSED_DATA_DIR, "features.csv")
    meta_df = pd.DataFrame(metadata)
    feat_cols = [f"feat_{i}" for i in range(X.shape[1])]
    feat_df = pd.DataFrame(X, columns=feat_cols)
    csv_df = pd.concat([meta_df, feat_df], axis=1)
    csv_df.to_csv(features_csv_path, index=False)
    print(f"[Pipeline] Saved tabular features.csv to: {features_csv_path}")
    
    # 7. Implement Stratified Train/Test Split
    print("[Pipeline] Creating stratified 80/20 train/test split...")
    
    # We pass indices or map splits
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_SEED,
        stratify=y
    )
    
    # Save splits
    x_train_path = os.path.join(Config.PROCESSED_DATA_DIR, "X_train.npy")
    x_test_path = os.path.join(Config.PROCESSED_DATA_DIR, "X_test.npy")
    y_train_path = os.path.join(Config.PROCESSED_DATA_DIR, "y_train.npy")
    y_test_path = os.path.join(Config.PROCESSED_DATA_DIR, "y_test.npy")
    
    np.save(x_train_path, X_train)
    np.save(x_test_path, X_test)
    np.save(y_train_path, y_train)
    np.save(y_test_path, y_test)
    print(f"[Pipeline] Saved split matrices to: {Config.PROCESSED_DATA_DIR}")
    print(f"  - X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  - X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # 8. Save textual feature report
    report_path = os.path.join(Config.METRICS_DIR, "feature_report.txt")
    print(f"[Pipeline] Saving feature report to: {report_path}")
    
    # Preprocessing stats
    avg_trimmed_pct = np.mean((np.array(original_durations) - np.array(trimmed_durations)) / np.array(original_durations)) * 100
    avg_scaling_factor = np.mean(scaling_factors)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("==========================================================\n")
        f.write("    RAVDESS Speech Preprocessing & Feature Extraction     \n")
        f.write("==========================================================\n\n")
        
        f.write("Data Dimensions:\n")
        f.write(f"  - Master Features Matrix: {X.shape}\n")
        f.write(f"  - Master Labels Vector:   {y.shape}\n")
        f.write(f"  - X_train Shape:          {X_train.shape}\n")
        f.write(f"  - X_test Shape:           {X_test.shape}\n")
        f.write(f"  - y_train Shape:          {y_train.shape}\n")
        f.write(f"  - y_test Shape:           {y_test.shape}\n\n")
        
        f.write("Acoustic Preprocessing Audits:\n")
        f.write(f"  - Avg Original Clip Length: {np.mean(original_durations):.3f}s\n")
        f.write(f"  - Avg Trimmed Silence Clip Length: {np.mean(trimmed_durations):.3f}s\n")
        f.write(f"  - Avg Trimming Percentage:  {avg_trimmed_pct:.2f}%\n")
        f.write(f"  - Avg Normalisation Scaling Factor: {avg_scaling_factor:.3f}\n")
        f.write(f"  - Fixed Duration Padding/Truncating: {Config.DURATION}s at {Config.SAMPLE_RATE}Hz\n\n")
        
        f.write("Feature Vectors Configuration:\n")
        f.write(f"  - MFCC (Temporal Mean + Std):   80 features (feats 0-79)\n")
        f.write(f"  - Chroma (Temporal Mean + Std): 24 features (feats 80-103)\n")
        f.write(f"  - Mel-Spectrogram (Mean + Std): 256 features (feats 104-359)\n")
        f.write(f"  - Spectral Contrast (Mean + Std): 14 features (feats 360-373)\n")
        f.write(f"  - Total Feature Size:           374 features\n\n")
        
        f.write("Split Class Balances (Emotion Stratification):\n")
        train_emo_s = pd.Series(y_train).value_counts()
        test_emo_s = pd.Series(y_test).value_counts()
        
        f.write(f"  {'Emotion':<12} | {'Train Count':<12} | {'Train %':<10} | {'Test Count':<12} | {'Test %':<10}\n")
        f.write("-" * 65 + "\n")
        for emo in Config.EMOTIONS:
            tr_cnt = train_emo_s.get(emo, 0)
            ts_cnt = test_emo_s.get(emo, 0)
            f.write(f"  {emo:<12} | {tr_cnt:<12d} | {tr_cnt/len(y_train)*100:<10.2f}% | {ts_cnt:<12d} | {ts_cnt/len(y_test)*100:<10.2f}%\n")
        f.write("\n")
        
        f.write("Data Verification Audits:\n")
        f.write(f"  - NaN values found: {nan_count}\n")
        f.write(f"  - Infinite values found: {inf_count}\n")
        f.write("  - Overall Audit Status: PASS\n")
        
    # 9. Generate Plots
    print("[Pipeline] Generating validation plots...")
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Split Class Balance
    split_plot_path = os.path.join(Config.PLOT_DIR, "emotion_distribution_split.png")
    print(f"  - Saving split distribution plot to: {split_plot_path}")
    
    train_counts = pd.Series(y_train).value_counts().reset_index()
    train_counts.columns = ["Emotion", "Count"]
    train_counts["Split"] = "Train (80%)"
    
    test_counts = pd.Series(y_test).value_counts().reset_index()
    test_counts.columns = ["Emotion", "Count"]
    test_counts["Split"] = "Test (20%)"
    
    split_df = pd.concat([train_counts, test_counts])
    
    plt.figure(figsize=(9, 5))
    sns.barplot(x="Emotion", y="Count", hue="Split", data=split_df, palette="muted")
    plt.title("Emotion Class Balance - Train vs Test Splits")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(split_plot_path, dpi=150)
    plt.close()
    
    # Compute variances for variance & correlation plotting
    variances = np.var(X, axis=0)
    
    # Plot 2: Feature Variance Histogram
    var_plot_path = os.path.join(Config.PLOT_DIR, "feature_variance_distribution.png")
    print(f"  - Saving feature variance histogram to: {var_plot_path}")
    plt.figure(figsize=(7, 4.5))
    # Log-scaling the variance values since they span different magnitudes (Mel spec energy vs chroma)
    sns.histplot(variances, kde=True, color="teal", log_scale=True)
    plt.title("Variance Distribution across 374 Acoustic Features")
    plt.xlabel("Feature Variance (Log Scale)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(var_plot_path, dpi=150)
    plt.close()
    
    # Plot 3: Feature Correlation Heatmap (Top 15 features with highest variance)
    corr_plot_path = os.path.join(Config.PLOT_DIR, "feature_correlation_heatmap.png")
    print(f"  - Saving feature correlation heatmap to: {corr_plot_path}")
    top_indices = np.argsort(variances)[-15:]
    top_features = X[:, top_indices]
    
    # Map index names to feature family tags
    # MFCC: 0-79, Chroma: 80-103, Mel: 104-359, Contrast: 360-373
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
        
    top_df = pd.DataFrame(top_features, columns=col_labels)
    plt.figure(figsize=(9, 8))
    sns.heatmap(top_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", square=True, annot_kws={"size": 8})
    plt.title("Pearson Correlation Heatmap (Top 15 High-Variance Features)")
    plt.tight_layout()
    plt.savefig(corr_plot_path, dpi=150)
    plt.close()
    
    print("==========================================================")
    print("  [PASS] Feature extraction pipeline completed!          ")
    print("  Report: results/metrics/feature_report.txt              ")
    print("  Plots saved inside: results/plots/                      ")
    print("  Split matrices saved under data/processed/              ")
    print("==========================================================")

if __name__ == "__main__":
    run_feature_extraction_pipeline()
