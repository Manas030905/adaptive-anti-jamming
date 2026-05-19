"""
Algorithm Comparison and Experiments Module
Runs comprehensive comparisons between LMS, NLMS, and RLS filters.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import time

from signals.audio_loader import generate_dummy_audio
from signals.jammer_generator import get_jammer
from signals.channel_model import generate_received_signal
from signals.reference_generator import generate_reference_noise
from filters.lms_filter import LMSFilter
from filters.nlms_filter import NLMSFilter
from filters.rls_filter import RLSFilter
from metrics.snr import calculate_snr_improvement, calculate_pesq_like_metric
from metrics.mse import calculate_error_metrics
from visualization.plots import SignalPlotter


def run_single_experiment(clean_audio, fs, jammer_type='combined', 
                         filter_order=64, output_dir='outputs'):
    """
    Run a single experiment comparing all three algorithms.
    
    Args:
        clean_audio (ndarray): Clean audio signal
        fs (int): Sampling frequency
        jammer_type (str): Type of jammer to apply
        filter_order (int): Adaptive filter order
        output_dir (str): Output directory
        
    Returns:
        dict: Experimental results
    """
    print(f"\n{'='*60}")
    print(f"Experiment: Jammer Type = {jammer_type}, Filter Order = {filter_order}")
    print(f"{'='*60}")
    
    # Generate jammer
    print(f"Generating {jammer_type} jammer...")
    jammer = get_jammer(len(clean_audio), fs, jammer_type=jammer_type)
    
    # Generate received signal
    print("Creating received signal...")
    channel_data = generate_received_signal(clean_audio, jammer, snr_db=15)
    received_signal = channel_data['received']
    
    # Generate reference noise
    print("Generating reference noise...")
    reference_noise = generate_reference_noise(jammer, correlation=0.9)
    
    # Initialize results
    results = {
        'jammer_type': jammer_type,
        'filter_order': filter_order,
        'algorithms': {}
    }
    
    # LMS Filter
    print("\n→ Running LMS filter...")
    lms_filter = LMSFilter(filter_order=filter_order, mu=0.01)
    lms_result = lms_filter.update(reference_noise, received_signal)
    
    lms_snr = calculate_snr_improvement(clean_audio, received_signal, lms_result['recovered'])
    lms_mse = calculate_error_metrics(clean_audio, received_signal, lms_result['recovered'])
    lms_pesq = calculate_pesq_like_metric(clean_audio, lms_result['recovered'])
    
    results['algorithms']['LMS'] = {
        'recovered': lms_result['recovered'],
        'snr': lms_snr,
        'mse': lms_mse,
        'pesq': lms_pesq,
        'execution_time': lms_result['convergence']['execution_time'],
        'final_mse': lms_result['convergence']['final_mse'],
        'mse_history': lms_result['convergence']['mse_history']
    }
    
    print(f"  Input SNR:  {lms_snr['input_snr_db']:.2f} dB")
    print(f"  Output SNR: {lms_snr['output_snr_db']:.2f} dB")
    print(f"  Improvement: {lms_snr['improvement_db']:.2f} dB")
    print(f"  MSE Reduction: {lms_mse['error_reduction_percent']:.1f}%")
    print(f"  Execution Time: {lms_result['convergence']['execution_time']:.4f}s")
    
    # NLMS Filter
    print("\n→ Running NLMS filter...")
    nlms_filter = NLMSFilter(filter_order=filter_order, mu=0.5)
    nlms_result = nlms_filter.update(reference_noise, received_signal)
    
    nlms_snr = calculate_snr_improvement(clean_audio, received_signal, nlms_result['recovered'])
    nlms_mse = calculate_error_metrics(clean_audio, received_signal, nlms_result['recovered'])
    nlms_pesq = calculate_pesq_like_metric(clean_audio, nlms_result['recovered'])
    
    results['algorithms']['NLMS'] = {
        'recovered': nlms_result['recovered'],
        'snr': nlms_snr,
        'mse': nlms_mse,
        'pesq': nlms_pesq,
        'execution_time': nlms_result['convergence']['execution_time'],
        'final_mse': nlms_result['convergence']['final_mse'],
        'mse_history': nlms_result['convergence']['mse_history']
    }
    
    print(f"  Input SNR:  {nlms_snr['input_snr_db']:.2f} dB")
    print(f"  Output SNR: {nlms_snr['output_snr_db']:.2f} dB")
    print(f"  Improvement: {nlms_snr['improvement_db']:.2f} dB")
    print(f"  MSE Reduction: {nlms_mse['error_reduction_percent']:.1f}%")
    print(f"  Execution Time: {nlms_result['convergence']['execution_time']:.4f}s")
    
    # RLS Filter
    print("\n→ Running RLS filter...")
    rls_filter = RLSFilter(filter_order=filter_order, lambda_param=0.99)
    rls_result = rls_filter.update(reference_noise, received_signal)
    
    rls_snr = calculate_snr_improvement(clean_audio, received_signal, rls_result['recovered'])
    rls_mse = calculate_error_metrics(clean_audio, received_signal, rls_result['recovered'])
    rls_pesq = calculate_pesq_like_metric(clean_audio, rls_result['recovered'])
    
    results['algorithms']['RLS'] = {
        'recovered': rls_result['recovered'],
        'snr': rls_snr,
        'mse': rls_mse,
        'pesq': rls_pesq,
        'execution_time': rls_result['convergence']['execution_time'],
        'final_mse': rls_result['convergence']['final_mse'],
        'mse_history': rls_result['convergence']['mse_history']
    }
    
    print(f"  Input SNR:  {rls_snr['input_snr_db']:.2f} dB")
    print(f"  Output SNR: {rls_snr['output_snr_db']:.2f} dB")
    print(f"  Improvement: {rls_snr['improvement_db']:.2f} dB")
    print(f"  MSE Reduction: {rls_mse['error_reduction_percent']:.1f}%")
    print(f"  Execution Time: {rls_result['convergence']['execution_time']:.4f}s")
    
    # Store signals for visualization
    results['signals'] = {
        'clean': clean_audio,
        'jammer': jammer,
        'received': received_signal
    }
    
    return results


def create_comparison_table(all_results, output_dir='outputs'):
    """
    Create comprehensive comparison table.
    
    Args:
        all_results (list): List of experimental results
        output_dir (str): Output directory
        
    Returns:
        pd.DataFrame: Comparison results
    """
    data = []
    
    for result in all_results:
        jammer_type = result['jammer_type']
        
        for algo_name, algo_data in result['algorithms'].items():
            row = {
                'Jammer_Type': jammer_type,
                'Algorithm': algo_name,
                'Input_SNR_dB': algo_data['snr']['input_snr_db'],
                'Output_SNR_dB': algo_data['snr']['output_snr_db'],
                'SNR_Improvement_dB': algo_data['snr']['improvement_db'],
                'MSE_Before': algo_data['mse']['mse_before'],
                'MSE_After': algo_data['mse']['mse_after'],
                'MSE_Reduction_Percent': algo_data['mse']['error_reduction_percent'],
                'PESQ_Score': algo_data['pesq'],
                'Execution_Time_s': algo_data['execution_time']
            }
            data.append(row)
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    csv_path = Path(output_dir) / 'comparison_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved comparison table: {csv_path}")
    
    return df


def analyze_stability(fs=16000, filter_orders=[32, 64, 128]):
    """
    Analyze filter stability and compute maximum stable step-size.
    
    Args:
        fs (int): Sampling frequency
        filter_orders (list): Filter orders to test
        
    Returns:
        dict: Stability analysis results
    """
    print("\n" + "="*60)
    print("STABILITY ANALYSIS")
    print("="*60)
    
    results = {}
    
    # Generate test signal
    clean, _ = generate_dummy_audio(duration=2, fs=fs)
    jammer = get_jammer(len(clean), fs, jammer_type='combined')
    channel_data = generate_received_signal(clean, jammer, snr_db=15)
    received = channel_data['received']
    reference = generate_reference_noise(jammer, correlation=0.9)
    
    for order in filter_orders:
        print(f"\nFilter Order: {order}")
        
        # LMS maximum stable step-size
        # Empirical rule: mu_max ≈ 1 / (10 * input_power * filter_order)
        input_power = np.mean(reference ** 2)
        mu_max_lms = 1.0 / (10 * input_power * order)
        
        print(f"  LMS:  mu_max ≈ {mu_max_lms:.6f}")
        
        # NLMS is more stable, typically mu_max ≈ 1
        print(f"  NLMS: mu_max ≈ 1.0")
        
        # RLS typically stable for 0 < lambda < 1
        print(f"  RLS:  lambda ∈ (0, 1), typically 0.99")
        
        results[order] = {
            'lms_mu_max': mu_max_lms,
            'nlms_mu_max': 1.0,
            'rls_lambda_range': (0.9, 0.99)
        }
    
    return results
