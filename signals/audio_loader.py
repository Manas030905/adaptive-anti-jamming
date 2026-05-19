"""
Audio Loader Module
Handles loading and preprocessing of communication audio signals.
"""

import numpy as np
import soundfile as sf


def load_audio(path):
    """
    Load audio file and convert to mono normalized signal.
    
    Args:
        path (str): Path to .wav file
        
    Returns:
        tuple: (signal (ndarray), sampling_frequency (int))
            - signal: Normalized audio signal (mono, float32, -1 to 1)
            - sampling_frequency: Sample rate in Hz
    """
    try:
        # Load audio file
        audio_data, fs = sf.read(path)
        
        # Convert stereo to mono if needed
        if len(audio_data.shape) > 1:
            # Average stereo channels
            audio_data = np.mean(audio_data, axis=1)
        
        # Normalize to [-1, 1] range
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
        
        # Convert to float32
        audio_data = audio_data.astype(np.float32)
        
        print(f"✓ Audio loaded: {path}")
        print(f"  Duration: {len(audio_data) / fs:.2f} seconds")
        print(f"  Sampling frequency: {fs} Hz")
        print(f"  Samples: {len(audio_data)}")
        
        return audio_data, fs
        
    except FileNotFoundError:
        print(f"✗ Error: Audio file not found at {path}")
        raise
    except Exception as e:
        print(f"✗ Error loading audio: {e}")
        raise


def generate_dummy_audio(duration=3.0, fs=16000, freq=440):
    """
    Generate a dummy audio signal for testing.
    
    Args:
        duration (float): Duration in seconds
        fs (int): Sampling frequency in Hz
        freq (float): Frequency of test signal in Hz
        
    Returns:
        tuple: (signal, fs)
    """
    n_samples = int(duration * fs)
    t = np.arange(n_samples) / fs
    
    # Generate multi-frequency signal
    signal = (0.5 * np.sin(2 * np.pi * freq * t) +
              0.3 * np.sin(2 * np.pi * (freq * 1.5) * t) +
              0.2 * np.sin(2 * np.pi * (freq * 2) * t))
    
    # Normalize
    signal = signal / np.max(np.abs(signal))
    
    return signal.astype(np.float32), fs
