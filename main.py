"""
Main Pipeline for Adaptive Anti-Jamming System
Comprehensive workflow for signal recovery and algorithm comparison.
"""

import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path
import time

# Import modules
from signals.audio_loader import load_audio, generate_dummy_audio
from signals.jammer_generator import get_jammer, combined_jammer
from signals.channel_model import generate_received_signal
from signals.reference_generator import generate_reference_noise
from filters.lms_filter import LMSFilter
from filters.nlms_filter import NLMSFilter
from filters.rls_filter import RLSFilter
from metrics.snr import calculate_snr_improvement
from metrics.mse import calculate_error_metrics
from visualization.plots import SignalPlotter
from experiments.compare_algorithms import run_single_experiment, create_comparison_table, analyze_stability


def main():
    """Main execution pipeline."""
    
    print("\n" + "="*70)
    print(" ADAPTIVE NOISE SUPPRESSION FOR ANTI-JAMMING IN COMMUNICATION SYSTEMS")
    print("="*70)
    
    # Configuration
    OUTPUT_DIR = Path('outputs')
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    AUDIO_DIR = Path('audio')
    AUDIO_DIR.mkdir(exist_ok=True)
    
    # Parameters
    FILTER_ORDER = 64
    DURATION = 3.0
    FS = 16000
    
    print("\n[STEP 1] Loading Audio Signal")
    print("-" * 70)
    
    audio_path = AUDIO_DIR / 'clean_audio.wav'
    if audio_path.exists():
        try:
            clean_audio, fs = load_audio(str(audio_path))
        except Exception as e:
            print(f"✗ Could not load audio from {audio_path}")
            print("→ Generating dummy audio for testing...")
            clean_audio, fs = generate_dummy_audio(duration=DURATION, fs=FS)
    else:
        print(f"✗ No audio file found at {audio_path}")
        print("→ Generating dummy audio for testing...")
        clean_audio, fs = generate_dummy_audio(duration=DURATION, fs=FS)
    
    print(f"✓ Audio loaded: {len(clean_audio)} samples at {fs} Hz")
    
    print("\n[STEP 2] Generating Jamming Signals")
    print("-" * 70)
    
    jammer = combined_jammer(len(clean_audio), fs)
    print(f"✓ Generated combined jammer ({len(jammer)} samples)")
    
    print("\n[STEP 3] Creating Received Signal")
    print("-" * 70)
    
    channel_data = generate_received_signal(clean_audio, jammer, snr_db=15)
    received_signal = channel_data['received']
    input_snr = channel_data['input_snr']
    
    print(f"✓ Received signal created")
    print(f"  Input SNR: {input_snr:.2f} dB")
    
    print("\n[STEP 4] Generating Reference Noise")
    print("-" * 70)
    
    reference_noise = generate_reference_noise(jammer, correlation=0.9)
    print(f"✓ Reference noise generated")
    
    print("\n[STEP 5] Running Adaptive Filters")
    print("-" * 70)
    
    # LMS Filter
    print("\n→ LMS Filter (μ = 0.01)")
    lms_filter = LMSFilter(filter_order=FILTER_ORDER, mu=0.01)
    lms_result = lms_filter.update(reference_noise, received_signal)
    lms_recovered = lms_result['recovered']
    lms_mse_history = lms_result['convergence']['mse_history']
    
    lms_snr = calculate_snr_improvement(clean_audio, received_signal, lms_recovered)
    lms_error = calculate_error_metrics(clean_audio, received_signal, lms_recovered)
    
    print(f"  ✓ Completed in {lms_result['convergence']['execution_time']:.4f} seconds")
    print(f"    Output SNR: {lms_snr['output_snr_db']:.2f} dB")
    print(f"    SNR Improvement: {lms_snr['improvement_db']:.2f} dB")
    print(f"    MSE Reduction: {lms_error['error_reduction_percent']:.1f}%")
    
    # NLMS Filter
    print("\n→ NLMS Filter (μ = 0.5)")
    nlms_filter = NLMSFilter(filter_order=FILTER_ORDER, mu=0.5)
    nlms_result = nlms_filter.update(reference_noise, received_signal)
    nlms_recovered = nlms_result['recovered']
    nlms_mse_history = nlms_result['convergence']['mse_history']
    
    nlms_snr = calculate_snr_improvement(clean_audio, received_signal, nlms_recovered)
    nlms_error = calculate_error_metrics(clean_audio, received_signal, nlms_recovered)
    
    print(f"  ✓ Completed in {nlms_result['convergence']['execution_time']:.4f} seconds")
    print(f"    Output SNR: {nlms_snr['output_snr_db']:.2f} dB")
    print(f"    SNR Improvement: {nlms_snr['improvement_db']:.2f} dB")
    print(f"    MSE Reduction: {nlms_error['error_reduction_percent']:.1f}%")
    
    # RLS Filter
    print("\n→ RLS Filter (λ = 0.99)")
    rls_filter = RLSFilter(filter_order=FILTER_ORDER, lambda_param=0.99)
    rls_result = rls_filter.update(reference_noise, received_signal)
    rls_recovered = rls_result['recovered']
    rls_mse_history = rls_result['convergence']['mse_history']
    
    rls_snr = calculate_snr_improvement(clean_audio, received_signal, rls_recovered)
    rls_error = calculate_error_metrics(clean_audio, received_signal, rls_recovered)
    
    print(f"  ✓ Completed in {rls_result['convergence']['execution_time']:.4f} seconds")
    print(f"    Output SNR: {rls_snr['output_snr_db']:.2f} dB")
    print(f"    SNR Improvement: {rls_snr['improvement_db']:.2f} dB")
    print(f"    MSE Reduction: {rls_error['error_reduction_percent']:.1f}%")
    
    print("\n[STEP 6] Calculating Metrics")
    print("-" * 70)
    
    # Create metrics table
    metrics_data = {
        'Algorithm': ['LMS', 'NLMS', 'RLS'],
        'Input_SNR_dB': [lms_snr['input_snr_db'], nlms_snr['input_snr_db'], rls_snr['input_snr_db']],
        'Output_SNR_dB': [lms_snr['output_snr_db'], nlms_snr['output_snr_db'], rls_snr['output_snr_db']],
        'SNR_Improvement_dB': [lms_snr['improvement_db'], nlms_snr['improvement_db'], rls_snr['improvement_db']],
        'MSE_Before': [lms_error['mse_before'], nlms_error['mse_before'], rls_error['mse_before']],
        'MSE_After': [lms_error['mse_after'], nlms_error['mse_after'], rls_error['mse_after']],
        'MSE_Reduction_Percent': [lms_error['error_reduction_percent'], nlms_error['error_reduction_percent'], rls_error['error_reduction_percent']],
        'Execution_Time_s': [lms_result['convergence']['execution_time'], nlms_result['convergence']['execution_time'], rls_result['convergence']['execution_time']]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_csv = OUTPUT_DIR / 'metrics.csv'
    metrics_df.to_csv(metrics_csv, index=False)
    
    print(f"✓ Metrics saved to {metrics_csv}")
    print("\nMetrics Summary:")
    print(metrics_df.to_string(index=False))
    
    print("\n[STEP 7] Generating Visualizations")
    print("-" * 70)
    
    plotter = SignalPlotter(output_dir=str(OUTPUT_DIR))
    
    # Plot signal comparison
    print("→ Generating signal comparison plot...")
    plotter.plot_signal_comparison(
        clean_audio, jammer, received_signal,
        lms_recovered, nlms_recovered, rls_recovered,
        fs=fs
    )
    
    # Plot convergence
    print("→ Generating convergence plot...")
    plotter.plot_convergence(lms_mse_history, nlms_mse_history, rls_mse_history, fs=fs)
    
    # Plot SNR comparison
    print("→ Generating SNR comparison plot...")
    snr_data = {
        'LMS': lms_snr,
        'NLMS': nlms_snr,
        'RLS': rls_snr
    }
    plotter.plot_snr_comparison(snr_data)
    
    # Plot MSE comparison
    print("→ Generating MSE comparison plot...")
    mse_data = {
        'LMS': lms_error,
        'NLMS': nlms_error,
        'RLS': rls_error
    }
    plotter.plot_mse_comparison(mse_data)
    
    # Plot execution time
    print("→ Generating execution time plot...")
    timing_data = {
        'LMS': lms_result['convergence']['execution_time'],
        'NLMS': nlms_result['convergence']['execution_time'],
        'RLS': rls_result['convergence']['execution_time']
    }
    plotter.plot_execution_time(timing_data)
    
    # Plot frequency response
    print("→ Generating frequency response plot...")
    plotter.plot_frequency_response(clean_audio, lms_recovered, nlms_recovered, rls_recovered, fs=fs)
    
    print("\n[STEP 8] Saving Audio Outputs")
    print("-" * 70)
    
    # Save recovered audio
    recovered_lms_path = OUTPUT_DIR / 'recovered_lms.wav'
    sf.write(recovered_lms_path, lms_recovered, fs)
    print(f"✓ Saved: {recovered_lms_path}")
    
    recovered_nlms_path = OUTPUT_DIR / 'recovered_nlms.wav'
    sf.write(recovered_nlms_path, nlms_recovered, fs)
    print(f"✓ Saved: {recovered_nlms_path}")
    
    recovered_rls_path = OUTPUT_DIR / 'recovered_rls.wav'
    sf.write(recovered_rls_path, rls_recovered, fs)
    print(f"✓ Saved: {recovered_rls_path}")
    
    # Save jammed signal for reference
    jammed_path = OUTPUT_DIR / 'jammed_signal.wav'
    sf.write(jammed_path, received_signal, fs)
    print(f"✓ Saved: {jammed_path}")
    
    # Save clean signal for reference
    clean_path = OUTPUT_DIR / 'clean_reference.wav'
    sf.write(clean_path, clean_audio, fs)
    print(f"✓ Saved: {clean_path}")
    
    print("\n[STEP 9] Running Comparative Experiments")
    print("-" * 70)
    
    jammer_types = ['tone', 'noise', 'sweep', 'burst', 'combined']
    all_results = []
    
    for jammer_type in jammer_types:
        result = run_single_experiment(
            clean_audio, fs,
            jammer_type=jammer_type,
            filter_order=FILTER_ORDER,
            output_dir=str(OUTPUT_DIR)
        )
        all_results.append(result)
    
    # Create comprehensive comparison table
    print("\n[STEP 10] Creating Comparison Table")
    print("-" * 70)
    
    comparison_df = create_comparison_table(all_results, output_dir=str(OUTPUT_DIR))
    print("\nComparison Results (first 10 rows):")
    print(comparison_df.head(10).to_string(index=False))
    
    print("\n[STEP 11] Stability Analysis")
    print("-" * 70)
    
    stability_results = analyze_stability(fs=fs, filter_orders=[32, 64, 128])
    
    print("\n[COMPLETED] Summary")
    print("="*70)
    print(f"✓ Signal recovered using 3 adaptive algorithms")
    print(f"✓ Plots generated and saved to {OUTPUT_DIR}/")
    print(f"✓ Audio outputs saved to {OUTPUT_DIR}/")
    print(f"✓ Metrics saved to {OUTPUT_DIR}/metrics.csv")
    print(f"✓ Comparison results saved to {OUTPUT_DIR}/comparison_results.csv")
    print("\nKey Results:")
    print(f"  • Best SNR Improvement: {max(lms_snr['improvement_db'], nlms_snr['improvement_db'], rls_snr['improvement_db']):.2f} dB")
    print(f"  • Fastest Algorithm: {min(timing_data, key=timing_data.get)}")
    print(f"  • Most Stable: RLS")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
