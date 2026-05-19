"""
Jammer Generator Module
Generates various types of jamming signals for anti-jamming simulation.
"""

import numpy as np
from scipy import signal as sp_signal


def tone_jammer(n_samples, fs, frequency, amplitude):
    """
    Generate a sinusoidal tone jammer.
    
    Args:
        n_samples (int): Number of samples
        fs (int): Sampling frequency in Hz
        frequency (float): Jammer frequency in Hz
        amplitude (float): Jammer amplitude (0-1)
        
    Returns:
        ndarray: Tone jammer signal
    """
    t = np.arange(n_samples) / fs
    jammer = amplitude * np.sin(2 * np.pi * frequency * t)
    return jammer.astype(np.float32)


def white_noise_jammer(n_samples, level=0.3):
    """
    Generate white noise jammer.
    
    Args:
        n_samples (int): Number of samples
        level (float): Noise power level (0-1)
        
    Returns:
        ndarray: White noise jammer signal
    """
    jammer = level * np.random.randn(n_samples)
    # Normalize to prevent clipping
    jammer = jammer / (np.max(np.abs(jammer)) + 1e-8) * level
    return jammer.astype(np.float32)


def sweep_jammer(n_samples, fs, f_start=50, f_end=500, amplitude=0.3):
    """
    Generate sweep (chirp) jammer.
    
    Args:
        n_samples (int): Number of samples
        fs (int): Sampling frequency in Hz
        f_start (float): Starting frequency in Hz
        f_end (float): Ending frequency in Hz
        amplitude (float): Jammer amplitude (0-1)
        
    Returns:
        ndarray: Sweep jammer signal
    """
    t = np.arange(n_samples) / fs
    
    # Generate chirp signal
    jammer = amplitude * sp_signal.chirp(
        t, 
        f0=f_start, 
        f1=f_end, 
        t1=t[-1], 
        method='linear'
    )
    
    return jammer.astype(np.float32)


def burst_jammer(n_samples, fs, burst_duration=0.1, burst_amplitude=0.8, burst_interval=0.5):
    """
    Generate burst jammer (high-energy intermittent interference).
    
    Args:
        n_samples (int): Number of samples
        fs (int): Sampling frequency in Hz
        burst_duration (float): Duration of each burst in seconds
        burst_amplitude (float): Burst amplitude (0-1)
        burst_interval (float): Interval between bursts in seconds
        
    Returns:
        ndarray: Burst jammer signal
    """
    jammer = np.zeros(n_samples)
    
    burst_samples = int(burst_duration * fs)
    interval_samples = int(burst_interval * fs)
    
    burst_idx = 0
    while burst_idx + burst_samples < n_samples:
        # Generate high-frequency burst
        burst_signal = burst_amplitude * np.sin(
            2 * np.pi * 2000 * np.arange(burst_samples) / fs
        )
        jammer[burst_idx:burst_idx + burst_samples] = burst_signal
        burst_idx += interval_samples
    
    return jammer.astype(np.float32)


def combined_jammer(n_samples, fs, tone_freq=180, 
                   tone_amp=0.2, noise_level=0.15, 
                   sweep_amp=0.2, burst_amp=0.15):
    """
    Generate combined jammer (tone + noise + sweep + burst).
    
    Args:
        n_samples (int): Number of samples
        fs (int): Sampling frequency in Hz
        tone_freq (float): Tone frequency in Hz
        tone_amp (float): Tone amplitude
        noise_level (float): Noise level
        sweep_amp (float): Sweep amplitude
        burst_amp (float): Burst amplitude
        
    Returns:
        ndarray: Combined jammer signal
    """
    # Generate each component
    tone = tone_jammer(n_samples, fs, tone_freq, tone_amp)
    noise = white_noise_jammer(n_samples, noise_level)
    sweep = sweep_jammer(n_samples, fs, amplitude=sweep_amp)
    burst = burst_jammer(n_samples, fs, burst_amplitude=burst_amp)
    
    # Combine all jammers
    combined = tone + noise + sweep + burst
    
    # Normalize to prevent excessive clipping
    max_val = np.max(np.abs(combined))
    if max_val > 1:
        combined = combined / max_val
    
    return combined.astype(np.float32)


def get_jammer(n_samples, fs, jammer_type='combined', **kwargs):
    """
    Get jammer signal by type.
    
    Args:
        n_samples (int): Number of samples
        fs (int): Sampling frequency in Hz
        jammer_type (str): Type of jammer ('tone', 'noise', 'sweep', 'burst', 'combined')
        **kwargs: Additional parameters for specific jammer types
        
    Returns:
        ndarray: Jammer signal
    """
    # Set defaults for tone jammer
    if jammer_type == 'tone':
        frequency = kwargs.get('frequency', 180)
        amplitude = kwargs.get('amplitude', 0.2)
        return tone_jammer(n_samples, fs, frequency, amplitude)
    elif jammer_type == 'noise':
        level = kwargs.get('level', 0.3)
        return white_noise_jammer(n_samples, level)
    elif jammer_type == 'sweep':
        f_start = kwargs.get('f_start', 50)
        f_end = kwargs.get('f_end', 500)
        amplitude = kwargs.get('amplitude', 0.3)
        return sweep_jammer(n_samples, fs, f_start, f_end, amplitude)
    elif jammer_type == 'burst':
        burst_duration = kwargs.get('burst_duration', 0.1)
        burst_amplitude = kwargs.get('burst_amplitude', 0.8)
        burst_interval = kwargs.get('burst_interval', 0.5)
        return burst_jammer(n_samples, fs, burst_duration, burst_amplitude, burst_interval)
    elif jammer_type == 'combined':
        return combined_jammer(n_samples, fs, **kwargs)
    else:
        raise ValueError(f"Unknown jammer type: {jammer_type}")
