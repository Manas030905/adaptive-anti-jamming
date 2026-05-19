"""
Comprehensive Algorithm Comparison with Optimal Scenarios
Tests each algorithm with parameters specifically tuned for best performance.
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
from metrics.snr import calculate_snr_improvement
from metrics.mse import calculate_error_metrics


def test_scenario(scenario_name, jammer_type, filter_order, lms_mu, nlms_mu, rls_lambda, input_snr):
    """Test a specific scenario with all three algorithms."""
    
    print(f"\n{scenario_name}")
    print(f"  Jammer: {jammer_type} | Order: {filter_order} | Input SNR: {input_snr}dB")
    print(f"  μ_LMS={lms_mu} | μ_NLMS={nlms_mu} | λ_RLS={rls_lambda}")
    
    # Generate signals
    clean_audio, fs = generate_dummy_audio(duration=3.0, fs=16000)
    
    # Get specific jammer type
    if jammer_type == "tone":
        jammer = get_jammer(len(clean_audio), fs, "tone", frequency=180, amplitude=0.2)
    elif jammer_type == "noise":
        jammer = get_jammer(len(clean_audio), fs, "noise", amplitude=0.3)
    elif jammer_type == "sweep":
        jammer = get_jammer(len(clean_audio), fs, "sweep", start_freq=50, end_freq=500, amplitude=0.3)
    
    channel_data = generate_received_signal(clean_audio, jammer, snr_db=input_snr)
    received_signal = channel_data['received']
    input_snr_actual = channel_data['input_snr']
    
    # Use jammer as reference (corrupted)
    reference_signal = jammer + 0.1 * np.random.randn(len(jammer))
    reference_signal = reference_signal / (np.std(reference_signal) + 1e-8)
    
    results = {
        'Scenario': scenario_name,
        'Jammer': jammer_type,
        'Order': filter_order,
        'Input_SNR': f"{input_snr_actual:.2f}"
    }
    
    try:
        # LMS
        lms_filter = LMSFilter(filter_order=filter_order, mu=lms_mu)
        lms_result = lms_filter.update(reference_signal, received_signal)
        lms_snr = calculate_snr_improvement(clean_audio, received_signal, lms_result['recovered'])
        lms_error = calculate_error_metrics(clean_audio, received_signal, lms_result['recovered'])
        results['LMS_SNR'] = f"{lms_snr['improvement_db']:.2f}"
        results['LMS_MSE_Red'] = f"{lms_error['error_reduction_percent']:.1f}%"
        print(f"  LMS:  SNR +{lms_snr['improvement_db']:.2f}dB | MSE {lms_error['error_reduction_percent']:.1f}%")
    except Exception as e:
        results['LMS_SNR'] = "ERROR"
        results['LMS_MSE_Red'] = "ERROR"
        print(f"  LMS:  ERROR - {str(e)[:30]}")
    
    try:
        # NLMS
        nlms_filter = NLMSFilter(filter_order=filter_order, mu=nlms_mu)
        nlms_result = nlms_filter.update(reference_signal, received_signal)
        nlms_snr = calculate_snr_improvement(clean_audio, received_signal, nlms_result['recovered'])
        nlms_error = calculate_error_metrics(clean_audio, received_signal, nlms_result['recovered'])
        results['NLMS_SNR'] = f"{nlms_snr['improvement_db']:.2f}"
        results['NLMS_MSE_Red'] = f"{nlms_error['error_reduction_percent']:.1f}%"
        print(f"  NLMS: SNR +{nlms_snr['improvement_db']:.2f}dB | MSE {nlms_error['error_reduction_percent']:.1f}%")
    except Exception as e:
        results['NLMS_SNR'] = "ERROR"
        results['NLMS_MSE_Red'] = "ERROR"
        print(f"  NLMS: ERROR - {str(e)[:30]}")
    
    try:
        # RLS
        rls_filter = RLSFilter(filter_order=filter_order, lambda_param=rls_lambda)
        rls_result = rls_filter.update(reference_signal, received_signal)
        rls_snr = calculate_snr_improvement(clean_audio, received_signal, rls_result['recovered'])
        rls_error = calculate_error_metrics(clean_audio, received_signal, rls_result['recovered'])
        results['RLS_SNR'] = f"{rls_snr['improvement_db']:.2f}"
        results['RLS_MSE_Red'] = f"{rls_error['error_reduction_percent']:.1f}%"
        print(f"  RLS:  SNR +{rls_snr['improvement_db']:.2f}dB | MSE {rls_error['error_reduction_percent']:.1f}%")
    except Exception as e:
        results['RLS_SNR'] = "ERROR"
        results['RLS_MSE_Red'] = "ERROR"
        print(f"  RLS:  ERROR - {str(e)[:30]}")
    
    return results


def main():
    """Run comprehensive algorithm comparison."""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE ADAPTIVE FILTER COMPARISON")
    print("Testing LMS vs NLMS vs RLS with Optimized Scenarios")
    print("="*80)
    
    all_results = []
    
    # CATEGORY 1: SINGLE TONE JAMMER (LMS-friendly baseline)
    print("\n[CATEGORY 1] SINGLE TONE JAMMER - Narrowband Interference")
    all_results.append(test_scenario(
        "Tone: Conservative (Order=16)",
        "tone", filter_order=16, lms_mu=0.01, nlms_mu=0.3, rls_lambda=0.99, input_snr=15
    ))
    
    all_results.append(test_scenario(
        "Tone: Moderate (Order=32)",
        "tone", filter_order=32, lms_mu=0.01, nlms_mu=0.4, rls_lambda=0.95, input_snr=15
    ))
    
    all_results.append(test_scenario(
        "Tone: Aggressive RLS (Order=16, λ=0.85)",
        "tone", filter_order=16, lms_mu=0.01, nlms_mu=0.3, rls_lambda=0.85, input_snr=15
    ))
    
    # CATEGORY 2: WHITE NOISE JAMMER (More challenging)
    print("\n[CATEGORY 2] WHITE NOISE JAMMER - Broadband Interference")
    all_results.append(test_scenario(
        "Noise: Standard (Order=32)",
        "noise", filter_order=32, lms_mu=0.01, nlms_mu=0.5, rls_lambda=0.99, input_snr=15
    ))
    
    all_results.append(test_scenario(
        "Noise: RLS Optimized (Order=24, λ=0.90)",
        "noise", filter_order=24, lms_mu=0.01, nlms_mu=0.4, rls_lambda=0.90, input_snr=15
    ))
    
    # CATEGORY 3: SWEEP JAMMER (Frequency-varying)
    print("\n[CATEGORY 3] SWEEP JAMMER - Frequency-Varying Interference")
    all_results.append(test_scenario(
        "Sweep: Standard (Order=32)",
        "sweep", filter_order=32, lms_mu=0.01, nlms_mu=0.5, rls_lambda=0.99, input_snr=15
    ))
    
    all_results.append(test_scenario(
        "Sweep: NLMS Optimized (Order=32, μ=0.6)",
        "sweep", filter_order=32, lms_mu=0.01, nlms_mu=0.6, rls_lambda=0.99, input_snr=15
    ))
    
    # CATEGORY 4: LOW SNR SCENARIOS (Harder problem)
    print("\n[CATEGORY 4] CHALLENGING: LOW INPUT SNR")
    all_results.append(test_scenario(
        "Low SNR (5dB, Tone, Order=24)",
        "tone", filter_order=24, lms_mu=0.015, nlms_mu=0.3, rls_lambda=0.85, input_snr=5
    ))
    
    all_results.append(test_scenario(
        "Low SNR (5dB, Noise, Order=24)",
        "noise", filter_order=24, lms_mu=0.015, nlms_mu=0.4, rls_lambda=0.90, input_snr=5
    ))
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    results_csv = output_dir / 'comprehensive_comparison.csv'
    results_df.to_csv(results_csv, index=False)
    
    # Display results
    print(f"\n{'='*80}")
    print("COMPREHENSIVE RESULTS TABLE")
    print(f"{'='*80}")
    print(results_df.to_string(index=False))
    
    print(f"\n{'='*80}")
    print("KEY FINDINGS")
    print(f"{'='*80}")
    
    # Try to convert and find best performers
    try:
        results_numeric = results_df.copy()
        results_numeric['LMS_SNR_num'] = pd.to_numeric(results_numeric['LMS_SNR'], errors='coerce')
        results_numeric['NLMS_SNR_num'] = pd.to_numeric(results_numeric['NLMS_SNR'], errors='coerce')
        results_numeric['RLS_SNR_num'] = pd.to_numeric(results_numeric['RLS_SNR'], errors='coerce')
        
        best_lms_idx = results_numeric['LMS_SNR_num'].idxmax()
        best_nlms_idx = results_numeric['NLMS_SNR_num'].idxmax()
        best_rls_idx = results_numeric['RLS_SNR_num'].idxmax()
        
        print(f"\n✓ Best LMS: {results_df.loc[best_lms_idx, 'Scenario']}")
        print(f"  SNR Improvement: {results_df.loc[best_lms_idx, 'LMS_SNR']} dB")
        
        print(f"\n✓ Best NLMS: {results_df.loc[best_nlms_idx, 'Scenario']}")
        print(f"  SNR Improvement: {results_df.loc[best_nlms_idx, 'NLMS_SNR']} dB")
        
        print(f"\n✓ Best RLS: {results_df.loc[best_rls_idx, 'Scenario']}")
        print(f"  SNR Improvement: {results_df.loc[best_rls_idx, 'RLS_SNR']} dB")
    except:
        pass
    
    print(f"\nResults saved to: {results_csv}")


if __name__ == "__main__":
    main()
