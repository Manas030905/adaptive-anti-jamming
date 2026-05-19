"""
SNR (Signal-to-Noise Ratio) Calculation Module
Computes SNR metrics for signal quality assessment.
"""

import numpy as np


def calculate_snr(signal, noise):
    """
    Calculate Signal-to-Noise Ratio (SNR).
    
    SNR = 10 * log10(P_signal / P_noise)
    
    Args:
        signal (ndarray): Signal component
        noise (ndarray): Noise/interference component
        
    Returns:
        float: SNR in dB
    """
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    
    if noise_power > 1e-10:
        snr = 10 * np.log10(signal_power / noise_power)
    else:
        snr = np.inf
    
    return snr


def calculate_snr_improvement(clean_signal, jammed_signal, recovered_signal):
    """
    Calculate SNR improvement after filtering.
    
    Args:
        clean_signal (ndarray): Original clean signal
        jammed_signal (ndarray): Jammed (corrupted) signal
        recovered_signal (ndarray): Filtered/recovered signal
        
    Returns:
        dict: SNR metrics
            - input_snr: SNR of jammed signal
            - output_snr: SNR of recovered signal
            - improvement: SNR improvement in dB
    """
    # Calculate input SNR (jammed vs clean)
    input_noise = jammed_signal - clean_signal
    input_snr = calculate_snr(clean_signal, input_noise)
    
    # Calculate output SNR (recovered vs clean)
    output_noise = recovered_signal - clean_signal
    output_snr = calculate_snr(clean_signal, output_noise)
    
    # SNR improvement
    improvement = output_snr - input_snr
    
    return {
        'input_snr_db': input_snr,
        'output_snr_db': output_snr,
        'improvement_db': improvement
    }


def calculate_segmental_snr(clean_signal, recovered_signal, segment_length=512):
    """
    Calculate segmental SNR (average SNR over segments).
    
    More robust to outliers than global SNR.
    
    Args:
        clean_signal (ndarray): Original clean signal
        recovered_signal (ndarray): Recovered signal
        segment_length (int): Length of each segment
        
    Returns:
        dict: Segmental SNR metrics
    """
    n_segments = len(clean_signal) // segment_length
    segmental_snrs = []
    
    for i in range(n_segments):
        start = i * segment_length
        end = start + segment_length
        
        clean_seg = clean_signal[start:end]
        recovered_seg = recovered_signal[start:end]
        
        error = recovered_seg - clean_seg
        seg_snr = calculate_snr(clean_seg, error)
        
        if np.isfinite(seg_snr):
            segmental_snrs.append(seg_snr)
    
    if segmental_snrs:
        avg_segmental_snr = np.mean(segmental_snrs)
        std_segmental_snr = np.std(segmental_snrs)
    else:
        avg_segmental_snr = 0
        std_segmental_snr = 0
    
    return {
        'mean_segmental_snr': avg_segmental_snr,
        'std_segmental_snr': std_segmental_snr,
        'segmental_snrs': segmental_snrs
    }


def calculate_pesq_like_metric(clean_signal, recovered_signal):
    """
    Calculate a simplified PESQ-like quality metric.
    
    Based on correlation and error statistics.
    
    Args:
        clean_signal (ndarray): Original clean signal
        recovered_signal (ndarray): Recovered signal
        
    Returns:
        float: Quality metric (0-5 scale, higher is better)
    """
    # Normalize signals
    clean_norm = clean_signal / (np.max(np.abs(clean_signal)) + 1e-8)
    recovered_norm = recovered_signal / (np.max(np.abs(recovered_signal)) + 1e-8)
    
    # Correlation
    correlation = np.corrcoef(clean_norm, recovered_norm)[0, 1]
    
    # Mean Squared Error
    mse = np.mean((clean_norm - recovered_norm) ** 2)
    
    # Combine metrics into 0-5 score
    # Correlation: -1 to 1 → 0 to 2.5
    correlation_score = (correlation + 1) * 1.25
    
    # MSE: normalize and invert
    # MSE score: 0 to 2.5
    mse_score = 2.5 * np.exp(-10 * mse)
    
    pesq_like = correlation_score + mse_score
    pesq_like = np.clip(pesq_like, 0, 5)
    
    return pesq_like
