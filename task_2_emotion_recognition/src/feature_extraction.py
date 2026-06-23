"""
Feature extraction module for Task 2: Emotion Recognition from Speech.
Extracts acoustic descriptors (MFCCs, Chroma, Mel Spectrogram, Spectral Contrast) 
from audio waveforms. For each feature family, computes both temporal mean and 
standard deviation, generating a consolidated feature vector per audio clip.
"""
import numpy as np
from src.config import Config

def extract_mfcc(waveform: np.ndarray, sr: int = Config.SAMPLE_RATE, 
                 n_mfcc: int = Config.N_MFCC) -> np.ndarray:
    """
    Extracts Mel-Frequency Cepstral Coefficients (MFCCs) from waveform.
    Returns concatenated mean (n_mfcc,) and std (n_mfcc,) arrays over time frames.
    """
    import librosa
    mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    return np.concatenate([mfcc_mean, mfcc_std])

def extract_chroma(waveform: np.ndarray, sr: int = Config.SAMPLE_RATE, 
                   n_chroma: int = Config.N_CHROMA) -> np.ndarray:
    """
    Extracts Chroma STFT features (pitch classes).
    Returns concatenated mean (n_chroma,) and std (n_chroma,) arrays over time frames.
    """
    import librosa
    chroma = librosa.feature.chroma_stft(y=waveform, sr=sr, n_chroma=n_chroma)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)
    return np.concatenate([chroma_mean, chroma_std])

def extract_mel_spectrogram(waveform: np.ndarray, sr: int = Config.SAMPLE_RATE, 
                            n_mels: int = Config.N_MELS) -> np.ndarray:
    """
    Extracts Mel-scaled spectrogram power.
    Returns concatenated mean (n_mels,) and std (n_mels,) arrays over time frames.
    """
    import librosa
    mel = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=n_mels)
    mel_mean = np.mean(mel, axis=1)
    mel_std = np.std(mel, axis=1)
    return np.concatenate([mel_mean, mel_std])

def extract_spectral_contrast(waveform: np.ndarray, sr: int = Config.SAMPLE_RATE) -> np.ndarray:
    """
    Extracts spectral contrast (peaks/valleys difference).
    Returns concatenated mean (7,) and std (7,) arrays over time frames.
    """
    import librosa
    contrast = librosa.feature.spectral_contrast(y=waveform, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)
    contrast_std = np.std(contrast, axis=1)
    return np.concatenate([contrast_mean, contrast_std])

def extract_combined_features(waveform: np.ndarray, sr: int = Config.SAMPLE_RATE) -> np.ndarray:
    """
    Extracts and concatenates MFCC, Chroma, Mel Spectrogram, and Spectral Contrast 
    mean and standard deviation vectors to generate a unified 1D acoustic representation (374 features).
    """
    mfcc_feats = extract_mfcc(waveform, sr)
    chroma_feats = extract_chroma(waveform, sr)
    mel_feats = extract_mel_spectrogram(waveform, sr)
    contrast_feats = extract_spectral_contrast(waveform, sr)
    
    return np.concatenate([
        mfcc_feats,
        chroma_feats,
        mel_feats,
        contrast_feats
    ])
