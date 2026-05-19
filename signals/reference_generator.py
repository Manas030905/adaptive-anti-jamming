"""
Reference Noise Generator Module
Generates correlated reference signals for adaptive filter training.
"""

import numpy as np


def generate_reference_noise(jammer_signal, correlation=0.9, noise_variance=0.01):
    """
    Generate reference noise signal correlated with jammer.
    
    Used as reference input to adaptive filters.
    Reference is designed to be partially correlated with the jammer
    so that the filter can learn and cancel it.
    
    Args:
        jammer_signal (ndarray): Original jammer signal
        correlation (float): Correlation coefficient (0.5-1.0)
        noise_variance (float): Variance of added random variation
        
    Returns:
        ndarray: Reference noise signal
    """
    # Scale jammer with correlation coefficient
    reference = correlation * jammer_signal.copy()
    
    # Add random variation (small perturbation)
    perturbation = np.sqrt(noise_variance) * np.random.randn(len(jammer_signal))
    reference = reference + perturbation
    
    # Normalize to similar power as jammer
    jammer_power = np.mean(jammer_signal ** 2)
    ref_power = np.mean(reference ** 2)
    
    if ref_power > 0:
        reference = reference * np.sqrt(jammer_power / ref_power)
    
    return reference.astype(np.float32)


def generate_reference_from_sensor(sensor_signal, delay=0, correlation=0.85):
    """
    Generate reference signal from auxiliary sensor.
    
    In practical anti-jamming systems, a second antenna/sensor
    can provide reference jammer signal.
    
    Args:
        sensor_signal (ndarray): Signal from auxiliary sensor
        delay (int): Time delay in samples between sensors
        correlation (float): Expected correlation level
        
    Returns:
        ndarray: Aligned reference signal
    """
    reference = sensor_signal.copy()
    
    # Apply delay
    if delay > 0:
        reference = np.concatenate([np.zeros(delay), reference[:-delay]])
    elif delay < 0:
        reference = np.concatenate([reference[-delay:], np.zeros(-delay)])
    
    # Add small noise
    noise = 0.02 * np.random.randn(len(reference))
    reference = reference + noise
    
    # Scale by correlation
    signal_power = np.mean(sensor_signal ** 2)
    ref_power = np.mean(reference ** 2)
    
    if ref_power > 0:
        reference = reference * np.sqrt(signal_power / ref_power) * correlation
    
    return reference.astype(np.float32)


def generate_band_limited_reference(jammer_signal, fs, cutoff_freq=2000, correlation=0.9):
    """
    Generate band-limited reference signal.
    
    Args:
        jammer_signal (ndarray): Original jammer signal
        fs (int): Sampling frequency in Hz
        cutoff_freq (float): Cutoff frequency for low-pass filter in Hz
        correlation (float): Correlation coefficient
        
    Returns:
        ndarray: Band-limited reference signal
    """
    from scipy import signal as sp_signal
    
    # Design low-pass filter
    nyquist = fs / 2
    normalized_cutoff = cutoff_freq / nyquist
    
    if normalized_cutoff >= 1:
        normalized_cutoff = 0.99
    
    # Create filter
    b, a = sp_signal.butter(4, normalized_cutoff, btype='low')
    
    # Filter jammer signal
    filtered_jammer = sp_signal.filtfilt(b, a, jammer_signal)
    
    # Apply correlation scaling
    reference = correlation * filtered_jammer
    
    # Add small random variation
    reference = reference + 0.01 * np.random.randn(len(reference))
    
    # Normalize power
    jammer_power = np.mean(jammer_signal ** 2)
    ref_power = np.mean(reference ** 2)
    
    if ref_power > 0:
        reference = reference * np.sqrt(jammer_power / ref_power)
    
    return reference.astype(np.float32)
