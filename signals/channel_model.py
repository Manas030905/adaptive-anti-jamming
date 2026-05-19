"""
Channel Model Module
Simulates communication channel with jammer and noise corruption.
"""

import numpy as np


def generate_received_signal(clean_audio, jammer, channel_noise=None, snr_db=20):
    """
    Generate received signal corrupted by jammer and channel noise.
    
    Equation:
        received = clean_audio + jammer + channel_noise
    
    Args:
        clean_audio (ndarray): Original communication signal
        jammer (ndarray): Jamming interference signal
        channel_noise (ndarray or None): Additive channel noise. If None, generated as white Gaussian noise
        snr_db (float): Signal-to-Noise Ratio in dB for channel noise
        
    Returns:
        dict: Dictionary containing:
            - 'received': Received corrupted signal
            - 'clean': Clean signal
            - 'jammer': Jammer signal
            - 'noise': Channel noise signal
            - 'input_snr': Input SNR in dB
    """
    n_samples = len(clean_audio)
    
    # Generate channel noise if not provided
    if channel_noise is None:
        # Generate white Gaussian noise
        noise_power = np.mean(clean_audio ** 2) / (10 ** (snr_db / 10))
        channel_noise = np.sqrt(noise_power) * np.random.randn(n_samples)
    
    # Ensure all signals have same length
    channel_noise = channel_noise[:n_samples]
    jammer_signal = jammer[:n_samples]
    
    # Generate received signal
    received = clean_audio + jammer_signal + channel_noise
    
    # Calculate input SNR (Signal vs Jammer+Noise)
    signal_power = np.mean(clean_audio ** 2)
    interference_power = np.mean(jammer_signal ** 2) + np.mean(channel_noise ** 2)
    
    if interference_power > 0:
        input_snr = 10 * np.log10(signal_power / interference_power)
    else:
        input_snr = np.inf
    
    return {
        'received': received.astype(np.float32),
        'clean': clean_audio,
        'jammer': jammer_signal,
        'noise': channel_noise,
        'input_snr': input_snr
    }


def multipath_channel(signal, delays, gains):
    """
    Apply multipath fading channel model.
    
    Args:
        signal (ndarray): Input signal
        delays (list): Delay in samples for each path
        gains (list): Gain for each path
        
    Returns:
        ndarray: Multipath-affected signal
    """
    output = np.zeros_like(signal)
    
    for delay, gain in zip(delays, gains):
        if delay < len(signal):
            padded_signal = np.concatenate([np.zeros(delay), signal[:-delay]])
            output += gain * padded_signal
    
    return output.astype(np.float32)


def calculate_sinr(clean_signal, jammer_signal, noise_signal=None):
    """
    Calculate SINR (Signal-to-Interference-plus-Noise Ratio).
    
    Args:
        clean_signal (ndarray): Original signal
        jammer_signal (ndarray): Interference signal
        noise_signal (ndarray or None): Noise signal
        
    Returns:
        float: SINR in dB
    """
    signal_power = np.mean(clean_signal ** 2)
    jammer_power = np.mean(jammer_signal ** 2)
    
    if noise_signal is not None:
        noise_power = np.mean(noise_signal ** 2)
    else:
        noise_power = 0
    
    interference_power = jammer_power + noise_power
    
    if interference_power > 1e-10:
        sinr = 10 * np.log10(signal_power / interference_power)
    else:
        sinr = np.inf
    
    return sinr
