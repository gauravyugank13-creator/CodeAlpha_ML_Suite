"""
Data loading, validation, and dataset ingestion module for Task 2: Emotion Recognition.
Downloads Zenodo dataset archives, extracts files, and parses RAVDESS naming rules.
"""
import os
import re
import sys
import zipfile
import requests
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import Config

class AudioDataLoader:
    """
    Handles download, extraction, parsing, and cataloging of RAVDESS audio files.
    """
    def __init__(self, raw_data_dir: str = Config.RAW_DATA_DIR):
        self.raw_data_dir = raw_data_dir
        
        # RAVDESS Emotion mapping dictionary
        self.emotion_map = {
            "01": "neutral",
            "02": "calm",
            "03": "happy",
            "04": "sad",
            "05": "angry",
            "06": "fearful",
            "07": "disgust",
            "08": "surprised"
        }

    def download_and_extract_dataset(self, url: str = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip") -> None:
        """
        Checks if dataset is present locally. If not, downloads the zip archive from Zenodo
        with a stdout progress bar and extracts it to data/raw/.
        """
        os.makedirs(self.raw_data_dir, exist_ok=True)
        
        # We check if there are already .wav files inside raw/ (meaning it was already extracted)
        wav_files = []
        for root, _, files in os.walk(self.raw_data_dir):
            for file in files:
                if file.lower().endswith(".wav"):
                    wav_files.append(file)
                    
        if len(wav_files) > 0:
            print(f"[DataLoader] Dataset already exists locally ({len(wav_files)} WAV files found). Skipping download.")
            return

        zip_path = os.path.join(os.path.dirname(self.raw_data_dir), "ravdess_speech.zip")
        print(f"[DataLoader] Downloading RAVDESS Speech dataset from: {url} ...")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            
            block_size = 1024 * 1024 # 1 MB chunks
            downloaded = 0
            
            with open(zip_path, "wb") as f:
                for data in response.iter_content(block_size):
                    f.write(data)
                    downloaded += len(data)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        # Print progress
                        sys.stdout.write(f"\r[DataLoader] Progress: {percent:.1f}% ({downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB)")
                        sys.stdout.flush()
            print("\n[DataLoader] Download complete. Starting extraction...")
            
            # Extract zip
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.raw_data_dir)
            print(f"[DataLoader] Extraction complete! Files saved to: {self.raw_data_dir}")
            
        except Exception as e:
            print(f"\n[Error] Dataset download or extraction failed: {e}")
            # Clean up partial files
            if os.path.exists(zip_path):
                os.remove(zip_path)
            raise e
        finally:
            # Clean up zip file
            if os.path.exists(zip_path):
                os.remove(zip_path)

    def parse_ravdess_filename(self, filename: str) -> dict:
        """
        Parses RAVDESS filename syntax:
        Modality - VocalChannel - Emotion - Intensity - Statement - Repetition - Actor.wav
        Example: 03-01-06-01-02-01-02.wav
        """
        base_name = os.path.splitext(filename)[0]
        parts = base_name.split("-")
        
        if len(parts) != 7:
            raise ValueError(f"Invalid RAVDESS filename format: {filename}")
            
        modality = parts[0]
        vocal_channel = parts[1]
        emotion_code = parts[2]
        intensity_code = parts[3]
        statement_code = parts[4]
        repetition_code = parts[5]
        actor_code = parts[6]
        
        emotion = self.emotion_map.get(emotion_code, "unknown")
        actor_id = int(actor_code)
        gender = "male" if actor_id % 2 == 1 else "female"
        intensity = "normal" if intensity_code == "01" else "strong"
        statement = "talking" if statement_code == "01" else "sitting"
        repetition = "1st" if repetition_code == "01" else "2nd"
        
        return {
            "file_name": filename,
            "modality": modality,
            "vocal_channel": vocal_channel,
            "emotion": emotion,
            "intensity": intensity,
            "statement": statement,
            "repetition": repetition,
            "actor_id": actor_id,
            "gender": gender
        }

    def scan_audio_files(self) -> list:
        """
        Walks the raw/ directory, finds all WAV files, parses their names, 
        and returns a catalog of samples.
        """
        catalog = []
        if not os.path.exists(self.raw_data_dir):
            return catalog
            
        for root, _, files in os.walk(self.raw_data_dir):
            for file in files:
                if file.lower().endswith(".wav"):
                    file_path = os.path.join(root, file)
                    try:
                        meta = self.parse_ravdess_filename(file)
                        meta["file_path"] = os.path.abspath(file_path)
                        catalog.append(meta)
                    except ValueError as e:
                        # Exclude files with invalid naming formats or print warnings
                        print(f"[Warning] Skipping file {file} due to: {e}")
                        
        return catalog

    def load_audio_waveform(self, file_path: str, target_sr: int = Config.SAMPLE_RATE) -> tuple:
        """
        Loads a single audio file waveform array.
        """
        import librosa
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        waveform, sr = librosa.load(file_path, sr=target_sr)
        return waveform, sr

    def split_train_test(self, files_list: list, test_size: float = Config.TEST_SIZE, 
                         random_state: int = Config.RANDOM_SEED) -> tuple:
        """
        Splits lists of file metadata into balanced train and test splits.
        """
        if len(files_list) == 0:
            return [], []
            
        df = pd.DataFrame(files_list)
        
        # Enforce stratified split on emotion classes
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=df["emotion"] if len(df["emotion"].unique()) > 1 else None
        )
        
        return train_df.to_dict("records"), test_df.to_dict("records")
