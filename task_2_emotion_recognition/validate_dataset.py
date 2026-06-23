"""
Dataset acquisition, validation, profiling, and reporting script for Task 2: Speech Emotion Recognition.
Downloads RAVDESS Speech audio, verifies integrity (corruption, duplicates), plots profiles, and writes reports.
"""
import os
import sys
import hashlib
import numpy as np
import pandas as pd
import soundfile as sf
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

def compute_md5(file_path: str) -> str:
    """
    Computes MD5 checksum hash of file bytes to detect exact duplicates.
    """
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def run_dataset_validation():
    print("==========================================================")
    print("  validate_dataset.py - Dataset Acquisition & Validation  ")
    print("==========================================================")
    
    # 1. Trigger dataset download and extraction
    loader = AudioDataLoader()
    try:
        loader.download_and_extract_dataset()
    except Exception as e:
        print(f"[Error] Failed during dataset acquisition phase: {e}")
        sys.exit(1)
        
    # 2. Walk directory structure and verify files
    print("[Pipeline] Cataloging and verifying files...")
    all_files = []
    unsupported_files = []
    
    for root, _, files in os.walk(Config.RAW_DATA_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            # Skip documentation files or folders config files
            if file.lower() in [".gitkeep", "dataset_source.txt", "desktop.ini"]:
                continue
            
            # Check format
            if not file.lower().endswith(".wav"):
                unsupported_files.append(file_path)
            else:
                all_files.append(file_path)
                
    total_found_files = len(all_files)
    print(f"[Pipeline] Found {total_found_files} WAV recordings.")
    
    # 3. File integrity verification (corruptions & duplicates)
    corrupted_files = []
    duplicate_records = {}
    md5_hashes = {}
    
    durations = []
    sample_rates = []
    
    print("[Pipeline] Performing checksum and integrity audits (this may take a few seconds)...")
    for file_path in all_files:
        # Check duplicate via MD5 hash
        try:
            file_hash = compute_md5(file_path)
            if file_hash in md5_hashes:
                duplicate_records[file_path] = md5_hashes[file_hash]
            else:
                md5_hashes[file_hash] = file_path
        except Exception as e:
            print(f"[Warning] Failed to hash file: {file_path}. Error: {e}")
            corrupted_files.append(file_path)
            continue
            
        # Check corruption and load duration/sample rate info
        try:
            info = sf.info(file_path)
            durations.append(info.duration)
            sample_rates.append(info.samplerate)
        except Exception as e:
            print(f"[Warning] Corrupted WAV file detected: {file_path}. Error: {e}")
            corrupted_files.append(file_path)
            
    # 4. Catalog successfully scanned metadata
    catalog = loader.scan_audio_files()
    
    # Filter catalog by non-corrupted lists
    non_corrupt_catalog = [item for item in catalog if item["file_path"] not in corrupted_files]
    df = pd.DataFrame(non_corrupt_catalog)
    
    # Assert size requirements
    total_valid = len(df)
    print(f"[Pipeline] Audit finished. Valid samples: {total_valid}, Corrupted: {len(corrupted_files)}, Duplicates: {len(duplicate_records)}")
    
    # 5. Extract statistics
    if total_valid == 0:
        print("[Error] Zero valid files found. Validation failed.")
        sys.exit(1)
        
    emotion_counts = df["emotion"].value_counts().to_dict()
    actor_counts = df["actor_id"].value_counts().to_dict()
    gender_counts = df["gender"].value_counts().to_dict()
    intensity_counts = df["intensity"].value_counts().to_dict()
    
    duration_stats = {
        "min": float(np.min(durations)),
        "max": float(np.max(durations)),
        "mean": float(np.mean(durations)),
        "std": float(np.std(durations))
    }
    
    # 6. Save textual dataset report
    os.makedirs(Config.METRICS_DIR, exist_ok=True)
    report_path = os.path.join(Config.METRICS_DIR, "dataset_report.txt")
    print(f"[Pipeline] Writing dataset metrics report to: {report_path}")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("==========================================================\n")
        f.write("        RAVDESS Speech Dataset Verification Report       \n")
        f.write("==========================================================\n\n")
        f.write(f"Total Files Cataloged:   {total_found_files}\n")
        f.write(f"Valid Waveform Samples:  {total_valid}\n")
        f.write(f"Corrupted File Count:   {len(corrupted_files)}\n")
        f.write(f"Duplicate Files Found:  {len(duplicate_records)}\n")
        f.write(f"Unsupported Extensions: {len(unsupported_files)}\n\n")
        
        f.write("Audio Length Statistics (seconds):\n")
        f.write(f"  - Minimum Duration: {duration_stats['min']:.3f}s\n")
        f.write(f"  - Maximum Duration: {duration_stats['max']:.3f}s\n")
        f.write(f"  - Average Duration: {duration_stats['mean']:.3f}s (± {duration_stats['std']:.3f}s)\n\n")
        
        f.write("Vocal Channel Sample Rates:\n")
        rate_counts = pd.Series(sample_rates).value_counts().to_dict()
        for rate, count in rate_counts.items():
            f.write(f"  - {rate} Hz: {count} clips\n")
        f.write("\n")
        
        f.write("Emotion Distribution:\n")
        for emo, count in sorted(emotion_counts.items()):
            f.write(f"  - {emo:<10}: {count} ({count/total_valid*100:.2f}%)\n")
        f.write("\n")
        
        f.write("Gender Class Distribution:\n")
        for gen, count in sorted(gender_counts.items()):
            f.write(f"  - {gen:<10}: {count} ({count/total_valid*100:.2f}%)\n")
        f.write("\n")
        
        f.write("Emotional Intensity Balance:\n")
        for inten, count in sorted(intensity_counts.items()):
            f.write(f"  - {inten:<10}: {count} ({count/total_valid*100:.2f}%)\n")
        f.write("\n")
        
        if corrupted_files:
            f.write("Corrupted Files Listing:\n")
            for item in corrupted_files:
                f.write(f"  - {os.path.basename(item)}\n")
            f.write("\n")
            
        if duplicate_records:
            f.write("Duplicate Files Listing:\n")
            for item, orig in duplicate_records.items():
                f.write(f"  - {os.path.basename(item)} (Duplicate of: {os.path.basename(orig)})\n")
            f.write("\n")
            
        if unsupported_files:
            f.write("Unsupported Files Listing:\n")
            for item in unsupported_files:
                f.write(f"  - {os.path.basename(item)}\n")
            f.write("\n")

    # 7. Generate Distribution Visualizations
    os.makedirs(Config.PLOT_DIR, exist_ok=True)
    sns.set_theme(style="darkgrid")
    
    # Plot 1: Emotion Distribution counts
    print("[Pipeline] Plotting emotion distribution chart...")
    plt.figure(figsize=(8, 4.5))
    emo_df = df["emotion"].value_counts().reset_index()
    emo_df.columns = ["Emotion", "Count"]
    sns.barplot(x="Count", y="Emotion", data=emo_df, palette="crest")
    plt.title("RAVDESS Emotion Class Distribution")
    plt.tight_layout()
    plot_emo_path = os.path.join(Config.PLOT_DIR, "emotion_distribution.png")
    plt.savefig(plot_emo_path, dpi=150)
    plt.close()
    
    # Plot 2: Actor distribution counts
    print("[Pipeline] Plotting actor distribution chart...")
    plt.figure(figsize=(10, 4))
    actor_df = df["actor_id"].value_counts().sort_index().reset_index()
    actor_df.columns = ["Actor ID", "Count"]
    sns.barplot(x="Actor ID", y="Count", data=actor_df, palette="magma")
    plt.title("Speech Samples Count per Actor ID")
    plt.xlabel("Actor Identifier (1 - 24)")
    plt.ylabel("Audio Clip Count")
    plt.tight_layout()
    plot_act_path = os.path.join(Config.PLOT_DIR, "actor_distribution.png")
    plt.savefig(plot_act_path, dpi=150)
    plt.close()
    
    # Plot 3: Duration distribution histogram
    print("[Pipeline] Plotting duration distribution chart...")
    plt.figure(figsize=(7, 4.5))
    sns.histplot(durations, kde=True, color="purple", bins=20)
    plt.title("Acoustic Audio Duration Distribution")
    plt.xlabel("Audio Clip Duration (seconds)")
    plt.ylabel("Frequency Count")
    plt.tight_layout()
    plot_dur_path = os.path.join(Config.PLOT_DIR, "duration_distribution.png")
    plt.savefig(plot_dur_path, dpi=150)
    plt.close()
    
    print("==========================================================")
    print("  [PASS] Dataset validation completed successfully!       ")
    print("  Report: results/metrics/dataset_report.txt              ")
    print("  Plots saved inside: results/plots/                      ")
    print("==========================================================")

if __name__ == "__main__":
    run_dataset_validation()
