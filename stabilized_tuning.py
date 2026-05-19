"""
Enhanced Parameter Tuning with Stabilized Filters
Includes numerical stability checks and weight clipping to prevent divergence.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings

from signals.audio_loader import generate_dummy_audio
from signals.jammer_generator import combined_jammer
from signals.channel_model import generate_received_signal
from signals.reference_generator import generate_reference_noise
from metrics.snr import calculate_snr_improvement
from metrics.mse import calculate_error_metrics

warnings.filterwarnings('ignore')


class StabilizedNLMSFilter:
    """NLMS with weight clipping to prevent divergence."""
    
    def __init__(self, filter_order=64, mu=0.5, eps=1e-8, weight_clip=5.0):
        self.filter_order = filter_order
        self.mu = mu
        self.eps = eps
        self.weight_clip = weight_clip
        self.weights = np.zeros(filter_order)
    
    def update(self, reference_signal, desired_signal):
        """NLMS with stabilization."""
        n_samples = len(reference_signal)
        estimated_noise = np.zeros(n_samples)
        error = np.zeros(n_samples)
        
        for n in range(n_samples):
            x_n = np.flip(reference_signal[max(0, n-self.filter_order+1):n+1])
            if len(x_n) < self.filter_order:
                x_n = np.concatenate([x_n, np.zeros(self.filter_order - len(x_n))])
            
            y_n = np.dot(self.weights, x_n)
            estimated_noise[n] = y_n
            e_n = desired_signal[n] - y_n
            error[n] = e_n
            
            norm_factor = np.dot(x_n, x_n) + self.eps
            mu_n = self.mu / norm_factor
            
            self.weights = self.weights + mu_n * e_n * x_n
            # Clip weights to prevent divergence
            self.weights = np.clip(self.weights, -self.weight_clip, self.weight_clip)
        
        recovered_signal = desired_signal - estimated_noise
        mse_history = np.array([e**2 for e in error])
        
        return {
            'recovered': recovered_signal.astype(np.float32),
            'error': error.astype(np.float32),
            'convergence': {'mse_history': mse_history}
        }


class StabilizedRLSFilter:
    """RLS with P-matrix regularization."""
    
    def __init__(self, filter_order=64, lambda_param=0.99, delta=1.0, reg_factor=1e-6):
        self.filter_order = filter_order
        self.lambda_param = lambda_param
        self.delta = delta
        self.reg_factor = reg_factor
        self.weights = np.zeros(filter_order)
        self.P = (1.0 / delta) * np.eye(filter_order)
    
    def update(self, reference_signal, desired_signal):
        """RLS with P-matrix regularization."""
        n_samples = len(reference_signal)
        estimated_noise = np.zeros(n_samples)
        error = np.zeros(n_samples)
        
        for n in range(n_samples):
            x_n = np.flip(reference_signal[max(0, n-self.filter_order+1):n+1])
            if len(x_n) < self.filter_order:
                x_n = np.concatenate([x_n, np.zeros(self.filter_order - len(x_n))])
            
            y_n = np.dot(self.weights, x_n)
            estimated_noise[n] = y_n
            e_n = desired_signal[n] - y_n
            error[n] = e_n
            
            # RLS update with regularization
            numerator = np.dot(self.P, x_n)
            denominator = self.lambda_param + np.dot(x_n, numerator)
            g_n = numerator / (denominator + 1e-10)
            
            self.weights = self.weights + g_n * e_n
            
            # Update P with regularization to maintain positive definiteness
            self.P = (1.0 / self.lambda_param) * (
                self.P - np.outer(g_n, np.dot(x_n, self.P))
            )
            # Add regularization diagonal
            self.P = self.P + self.reg_factor * np.eye(self.filter_order)
        
        recovered_signal = desired_signal - estimated_noise
        mse_history = np.array([e**2 for e in error])
        
        return {
            'recovered': recovered_signal.astype(np.float32),
            'error': error.astype(np.float32),
            'convergence': {'mse_history': mse_history}
        }


class SimpleLMSFilter:
    """Basic LMS filter."""
    
    def __init__(self, filter_order=64, mu=0.01):
        self.filter_order = filter_order
        self.mu = mu
        self.weights = np.zeros(filter_order)
    
    def update(self, reference_signal, desired_signal):
        """LMS algorithm."""
        n_samples = len(reference_signal)
        estimated_noise = np.zeros(n_samples)
        error = np.zeros(n_samples)
        
        for n in range(n_samples):
            x_n = np.flip(reference_signal[max(0, n-self.filter_order+1):n+1])
            if len(x_n) < self.filter_order:
                x_n = np.concatenate([x_n, np.zeros(self.filter_order - len(x_n))])
            
            y_n = np.dot(self.weights, x_n)
            estimated_noise[n] = y_n
            e_n = desired_signal[n] - y_n
            error[n] = e_n
            
            self.weights = self.weights + self.mu * e_n * x_n
        
        recovered_signal = desired_signal - estimated_noise
        mse_history = np.array([e**2 for e in error])
        
        return {
            'recovered': recovered_signal.astype(np.float32),
            'error': error.astype(np.float32),
            'convergence': {'mse_history': mse_history}
        }


def run_stabilized_test(config_name, lms_mu, nlms_mu, rls_lambda, filter_order, input_snr=15):
    """Run test with stabilized filters."""
    
    print(f"\n{config_name}")
    print(f"Order: {filter_order} | NLMS μ={nlms_mu} | RLS λ={rls_lambda}")
    
    # Generate signals
    clean_audio, _ = generate_dummy_audio(duration=3.0, fs=16000)
    jammer = combined_jammer(len(clean_audio), 16000)
    channel_data = generate_received_signal(clean_audio, jammer, snr_db=input_snr)
    received_signal = channel_data['received']
    reference_noise = generate_reference_noise(jammer, correlation=0.9)
    
    results = {'Config': config_name}
    
    # LMS
    lms = SimpleLMSFilter(filter_order=filter_order, mu=lms_mu)
    lms_result = lms.update(reference_noise, received_signal)
    lms_snr = calculate_snr_improvement(clean_audio, received_signal, lms_result['recovered'])
    lms_error = calculate_error_metrics(clean_audio, received_signal, lms_result['recovered'])
    results['LMS_SNR'] = lms_snr['improvement_db']
    results['LMS_MSE'] = lms_error['error_reduction_percent']
    print(f"  LMS:  {lms_snr['improvement_db']:.2f}dB SNR | {lms_error['error_reduction_percent']:.1f}% MSE")
    
    # NLMS Stabilized
    nlms = StabilizedNLMSFilter(filter_order=filter_order, mu=nlms_mu, weight_clip=10.0)
    nlms_result = nlms.update(reference_noise, received_signal)
    nlms_snr = calculate_snr_improvement(clean_audio, received_signal, nlms_result['recovered'])
    nlms_error = calculate_error_metrics(clean_audio, received_signal, nlms_result['recovered'])
    results['NLMS_SNR'] = nlms_snr['improvement_db']
    results['NLMS_MSE'] = nlms_error['error_reduction_percent']
    print(f"  NLMS: {nlms_snr['improvement_db']:.2f}dB SNR | {nlms_error['error_reduction_percent']:.1f}% MSE")
    
    # RLS Stabilized
    rls = StabilizedRLSFilter(filter_order=filter_order, lambda_param=rls_lambda, reg_factor=1e-5)
    rls_result = rls.update(reference_noise, received_signal)
    rls_snr = calculate_snr_improvement(clean_audio, received_signal, rls_result['recovered'])
    rls_error = calculate_error_metrics(clean_audio, received_signal, rls_result['recovered'])
    results['RLS_SNR'] = rls_snr['improvement_db']
    results['RLS_MSE'] = rls_error['error_reduction_percent']
    print(f"  RLS:  {rls_snr['improvement_db']:.2f}dB SNR | {rls_error['error_reduction_percent']:.1f}% MSE")
    
    return results


def main():
    """Run stabilized parameter tests."""
    
    print("\n" + "="*70)
    print("STABILIZED FILTER PARAMETER TUNING (With Divergence Prevention)")
    print("="*70)
    
    all_results = []
    
    # Test 1: Small filter order (better stability)
    all_results.append(run_stabilized_test(
        "TEST 1: Small Order (16)",
        lms_mu=0.01, nlms_mu=0.5, rls_lambda=0.99,
        filter_order=16, input_snr=15
    ))
    
    # Test 2: Medium order with boosted NLMS
    all_results.append(run_stabilized_test(
        "TEST 2: Medium Order (32) + NLMS Boost",
        lms_mu=0.01, nlms_mu=0.3, rls_lambda=0.99,
        filter_order=32, input_snr=15
    ))
    
    # Test 3: RLS with aggressive forgetting
    all_results.append(run_stabilized_test(
        "TEST 3: RLS Aggressive (λ=0.90)",
        lms_mu=0.01, nlms_mu=0.5, rls_lambda=0.90,
        filter_order=32, input_snr=15
    ))
    
    # Test 4: All optimized for this problem
    all_results.append(run_stabilized_test(
        "TEST 4: Fully Optimized (Order=24)",
        lms_mu=0.015, nlms_mu=0.25, rls_lambda=0.85,
        filter_order=24, input_snr=15
    ))
    
    # Test 5: Very small filter (maximum stability)
    all_results.append(run_stabilized_test(
        "TEST 5: Minimal Order (8)",
        lms_mu=0.01, nlms_mu=0.4, rls_lambda=0.95,
        filter_order=8, input_snr=15
    ))
    
    # Test 6: Conservative NLMS, aggressive RLS
    all_results.append(run_stabilized_test(
        "TEST 6: Conservative NLMS (μ=0.1), Aggressive RLS (λ=0.80)",
        lms_mu=0.01, nlms_mu=0.1, rls_lambda=0.80,
        filter_order=32, input_snr=15
    ))
    
    # Create summary
    results_df = pd.DataFrame(all_results)
    
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    results_csv = output_dir / 'stabilized_tuning_results.csv'
    results_df.to_csv(results_csv, index=False)
    
    print(f"\n{'='*70}")
    print("FINAL COMPARISON TABLE")
    print(f"{'='*70}")
    print(results_df.to_string(index=False))
    print(f"\nResults saved to: {results_csv}")
    
    # Recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print(f"{'='*70}")
    best_config = results_df.loc[results_df['RLS_SNR'].idxmax()]
    print(f"✓ Best RLS Performance: {best_config['Config']}")
    print(f"  RLS SNR Improvement: {best_config['RLS_SNR']:.2f}dB")
    
    best_nlms = results_df.loc[results_df['NLMS_SNR'].idxmax()]
    print(f"\n✓ Best NLMS Performance: {best_nlms['Config']}")
    print(f"  NLMS SNR Improvement: {best_nlms['NLMS_SNR']:.2f}dB")


if __name__ == "__main__":
    main()
