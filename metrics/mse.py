"""
Mean Squared Error (MSE) Calculation Module
"""

import numpy as np


def calculate_mse(signal1, signal2):
    """
    Calculate Mean Squared Error between two signals.
    
    Args:
        signal1 (ndarray): First signal
        signal2 (ndarray): Second signal
        
    Returns:
        float: MSE value
    """
    if len(signal1) != len(signal2):
        min_len = min(len(signal1), len(signal2))
        signal1 = signal1[:min_len]
        signal2 = signal2[:min_len]
    
    mse = np.mean((signal1 - signal2) ** 2)
    return mse


def calculate_rmse(signal1, signal2):
    """
    Calculate Root Mean Squared Error.
    
    Args:
        signal1 (ndarray): First signal
        signal2 (ndarray): Second signal
        
    Returns:
        float: RMSE value
    """
    mse = calculate_mse(signal1, signal2)
    rmse = np.sqrt(mse)
    return rmse


def calculate_error_metrics(clean_signal, corrupted_signal, recovered_signal):
    """
    Calculate comprehensive error metrics.
    
    Args:
        clean_signal (ndarray): Original clean signal
        corrupted_signal (ndarray): Corrupted received signal
        recovered_signal (ndarray): Recovered signal after filtering
        
    Returns:
        dict: Error metrics
    """
    # Error before filtering
    error_before = calculate_mse(clean_signal, corrupted_signal)
    rmse_before = calculate_rmse(clean_signal, corrupted_signal)
    
    # Error after filtering
    error_after = calculate_mse(clean_signal, recovered_signal)
    rmse_after = calculate_rmse(clean_signal, recovered_signal)
    
    # Error reduction
    error_reduction = error_before - error_after
    error_reduction_percent = (error_reduction / error_before * 100) if error_before > 0 else 0
    
    return {
        'mse_before': error_before,
        'rmse_before': rmse_before,
        'mse_after': error_after,
        'rmse_after': rmse_after,
        'mse_reduction': error_reduction,
        'error_reduction_percent': error_reduction_percent
    }


def calculate_mad(signal1, signal2):
    """
    Calculate Mean Absolute Deviation.
    
    Args:
        signal1 (ndarray): First signal
        signal2 (ndarray): Second signal
        
    Returns:
        float: MAD value
    """
    if len(signal1) != len(signal2):
        min_len = min(len(signal1), len(signal2))
        signal1 = signal1[:min_len]
        signal2 = signal2[:min_len]
    
    mad = np.mean(np.abs(signal1 - signal2))
    return mad
