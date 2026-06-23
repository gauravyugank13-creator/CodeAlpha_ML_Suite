"""
Audio preprocessing and waveform sanitisation module for Task 2: Emotion Recognition.
Provides helper functions for trimming silences, padding, normalisation, and augmentation.
"""
import numpy as np
from src.config import Config

def trim_silence(waveform: np.ndarray, top_db: float = 30.0) -> np.ndarray:
    """
    Trims silent margins at the boundaries of the audio sample.
    """
    import librosa
    trimmed_wave, _ = librosa.effects.trim(waveform, top_db=top_db)
    return trimmed_wave

def normalize_amplitude(waveform: np.ndarray) -> np.ndarray:
    """
    Scales the wave amplitude to have peak value of 1.0.
    """
    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        return waveform / max_val
    return waveform

def pad_or_truncate(waveform: np.ndarray, target_sr: int = Config.SAMPLE_RATE, 
                     target_duration: float = Config.DURATION) -> np.ndarray:
    """
    Truncates a waveform to target_length or pads with zeros to fit exact duration.
    """
    target_samples = int(target_sr * target_duration)
    current_samples = len(waveform)
    
    if current_samples > target_samples:
        # Truncate
        return waveform[:target_samples]
    elif current_samples < target_samples:
        # Pad with zeros
        padding = target_samples - current_samples
        return np.pad(waveform, (0, padding), mode="constant")
    
    return waveform

def add_white_noise(waveform: np.ndarray, noise_level: float = 0.005) -> np.ndarray:
    """
    Adds low-level white noise to the audio wave for model regularization.
    """
    noise = np.random.normal(0, noise_level, len(waveform))
    return waveform + noise
