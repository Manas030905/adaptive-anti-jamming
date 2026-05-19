"""
Parameter Tuning Script for NLMS and RLS Optimization
Tests different configurations to improve NLMS and RLS performance.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import time

from signals.audio_loader import generate_dummy_audio
from signals.jammer_generator import combined_jammer
from signals.channel_model import generate_received_signal
from signals.reference_generator import generate_reference_noise
from filters.lms_filter import LMSFilter
from filters.nlms_filter import NLMSFilter
from filters.rls_filter import RLSFilter
from metrics.snr import calculate_snr_improvement
from metrics.mse import calculate_error_metrics


def run_test(config_name, lms_mu, nlms_mu, rls_lambda, filter_order, duration=3.0, fs=16000, input_snr=15):
    """Run a single configuration test."""
    
    print(f"\n{'='*70}")
    print(f"CONFIG: {config_name}")
    print(f"{'='*70}")
    print(f"Filter Order: {filter_order} | Duration: {duration}s | Input SNR: {input_snr}dB")
    print(f"LMS μ={lms_mu} | NLMS μ={nlms_mu} | RLS λ={rls_lambda}")
    print("-"*70)
    
    # Generate signals
    clean_audio, _ = generate_dummy_audio(duration=duration, fs=fs)
    jammer = combined_jammer(len(clean_audio), fs)
    channel_data = generate_received_signal(clean_audio, jammer, snr_db=input_snr)
    received_signal = channel_data['received']
    reference_noise = generate_reference_noise(jammer, correlation=0.9)
    
    results = {'Config': config_name, 'Filter_Order': filter_order, 'Input_SNR': input_snr}
    
    # Test LMS
    lms_filter = LMSFilter(filter_order=filter_order, mu=lms_mu)
    lms_result = lms_filter.update(reference_noise, received_signal)
    lms_snr = calculate_snr_improvement(clean_audio, received_signal, lms_result['recovered'])
    lms_error = calculate_error_metrics(clean_audio, received_signal, lms_result['recovered'])
    
    results['LMS_SNR_Imp'] = lms_snr['improvement_db']
    results['LMS_MSE_Red'] = lms_error['error_reduction_percent']
    results['LMS_Time'] = lms_result['convergence']['execution_time']
    
    print(f"LMS:  SNR +{lms_snr['improvement_db']:.2f}dB | MSE -{lms_error['error_reduction_percent']:.1f}% | {lms_result['convergence']['execution_time']:.4f}s")
    
    # Test NLMS
    nlms_filter = NLMSFilter(filter_order=filter_order, mu=nlms_mu)
    nlms_result = nlms_filter.update(reference_noise, received_signal)
    nlms_snr = calculate_snr_improvement(clean_audio, received_signal, nlms_result['recovered'])
    nlms_error = calculate_error_metrics(clean_audio, received_signal, nlms_result['recovered'])
    
    results['NLMS_SNR_Imp'] = nlms_snr['improvement_db']
    results['NLMS_MSE_Red'] = nlms_error['error_reduction_percent']
    results['NLMS_Time'] = nlms_result['convergence']['execution_time']
    
    print(f"NLMS: SNR +{nlms_snr['improvement_db']:.2f}dB | MSE -{nlms_error['error_reduction_percent']:.1f}% | {nlms_result['convergence']['execution_time']:.4f}s")
    
    # Test RLS
    rls_filter = RLSFilter(filter_order=filter_order, lambda_param=rls_lambda)
    rls_result = rls_filter.update(reference_noise, received_signal)
    rls_snr = calculate_snr_improvement(clean_audio, received_signal, rls_result['recovered'])
    rls_error = calculate_error_metrics(clean_audio, received_signal, rls_result['recovered'])
    
    results['RLS_SNR_Imp'] = rls_snr['improvement_db']
    results['RLS_MSE_Red'] = rls_error['error_reduction_percent']
    results['RLS_Time'] = rls_result['convergence']['execution_time']
    
    print(f"RLS:  SNR +{rls_snr['improvement_db']:.2f}dB | MSE -{rls_error['error_reduction_percent']:.1f}% | {rls_result['convergence']['execution_time']:.4f}s")
    
    # Ranking
    snr_ranking = sorted([
        ('LMS', lms_snr['improvement_db']),
        ('NLMS', nlms_snr['improvement_db']),
        ('RLS', rls_snr['improvement_db'])
    ], key=lambda x: x[1], reverse=True)
    
    print(f"\nSNR Ranking: {snr_ranking[0][0]}(+{snr_ranking[0][1]:.2f}dB) > {snr_ranking[1][0]} > {snr_ranking[2][0]}")
    
    return results


def main():
    """Run parameter tuning tests."""
    
    print("\n" + "="*70)
    print(" PARAMETER TUNING FOR ADAPTIVE FILTER OPTIMIZATION")
    print("="*70)
    
    all_results = []
    
    # Test 1: BASELINE (Current config)
    all_results.append(run_test(
        "BASELINE (Current)",
        lms_mu=0.01, nlms_mu=0.5, rls_lambda=0.99,
        filter_order=64, duration=3.0, input_snr=15
    ))
    
    # Test 2: Reduced filter order (better for RLS)
    all_results.append(run_test(
        "REDUCED FILTER ORDER (32)",
        lms_mu=0.01, nlms_mu=0.5, rls_lambda=0.99,
        filter_order=32, duration=3.0, input_snr=15
    ))
    
    # Test 3: Optimized NLMS μ (higher)
    all_results.append(run_test(
        "NLMS OPTIMIZED (μ=0.8)",
        lms_mu=0.01, nlms_mu=0.8, rls_lambda=0.99,
        filter_order=64, duration=3.0, input_snr=15
    ))
    
    # Test 4: Optimized RLS λ (more aggressive forgetting)
    all_results.append(run_test(
        "RLS OPTIMIZED (λ=0.95)",
        lms_mu=0.01, nlms_mu=0.5, rls_lambda=0.95,
        filter_order=64, duration=3.0, input_snr=15
    ))
    
    # Test 5: Combined optimizations (reduced order + optimized NLMS/RLS)
    all_results.append(run_test(
        "COMBINED (Order=32, NLMS=0.8, RLS=0.95)",
        lms_mu=0.01, nlms_mu=0.8, rls_lambda=0.95,
        filter_order=32, duration=3.0, input_snr=15
    ))
    
    # Test 6: Higher LMS μ to make it more competitive
    all_results.append(run_test(
        "ALL ALGORITHMS BOOSTED (μ_lms=0.02, μ_nlms=1.0, λ=0.9)",
        lms_mu=0.02, nlms_mu=1.0, rls_lambda=0.9,
        filter_order=32, duration=3.0, input_snr=15
    ))
    
    # Test 7: Longer signal to show convergence
    all_results.append(run_test(
        "LONGER SIGNAL (Duration=5s, Order=64)",
        lms_mu=0.01, nlms_mu=0.5, rls_lambda=0.99,
        filter_order=64, duration=5.0, input_snr=15
    ))
    
    # Test 8: Lower input SNR (harder problem)
    all_results.append(run_test(
        "HARDER PROBLEM (Input SNR=5dB)",
        lms_mu=0.01, nlms_mu=0.5, rls_lambda=0.99,
        filter_order=64, duration=3.0, input_snr=5
    ))
    
    # Save results
    results_df = pd.DataFrame(all_results)
    
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    results_csv = output_dir / 'parameter_tuning_results.csv'
    results_df.to_csv(results_csv, index=False)
    
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(results_df.to_string(index=False))
    print(f"\nResults saved to: {results_csv}")


if __name__ == "__main__":
    main()
