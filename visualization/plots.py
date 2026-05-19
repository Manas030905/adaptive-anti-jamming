"""
Visualization and Plotting Module
Generates comprehensive plots for anti-jamming analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


class SignalPlotter:
    """Generate plots for signal processing results."""
    
    def __init__(self, output_dir='outputs'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def plot_signal_comparison(self, clean, jammer, received, recovered_lms, 
                               recovered_nlms, recovered_rls, fs=16000, 
                               time_range=None):
        """
        Plot comprehensive signal comparison.
        
        Args:
            clean: Clean signal
            jammer: Jammer signal
            received: Received corrupted signal
            recovered_lms: LMS recovered signal
            recovered_nlms: NLMS recovered signal
            recovered_rls: RLS recovered signal
            fs: Sampling frequency
            time_range: Optional tuple (start_time, end_time) in seconds
        """
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3)
        
        # Get time vector
        time_vec = np.arange(len(clean)) / fs
        
        # Apply time range if specified
        if time_range:
            start_idx = int(time_range[0] * fs)
            end_idx = int(time_range[1] * fs)
            time_vec = time_vec[start_idx:end_idx]
            clean = clean[start_idx:end_idx]
            jammer = jammer[start_idx:end_idx]
            received = received[start_idx:end_idx]
            recovered_lms = recovered_lms[start_idx:end_idx]
            recovered_nlms = recovered_nlms[start_idx:end_idx]
            recovered_rls = recovered_rls[start_idx:end_idx]
        
        # 1. Clean signal
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(time_vec, clean, 'g-', linewidth=1)
        ax1.set_title('Clean Communication Signal', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        
        # 2. Jammer signal
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(time_vec, jammer, 'r-', linewidth=1)
        ax2.set_title('Combined Jammer Signal', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True, alpha=0.3)
        
        # 3. Received signal
        ax3 = fig.add_subplot(gs[1, :])
        ax3.plot(time_vec, received, 'orange', linewidth=0.8, label='Received (Jammed)')
        ax3.plot(time_vec, clean, 'g--', linewidth=0.8, alpha=0.7, label='Clean (Reference)')
        ax3.set_title('Received Signal (Clean + Jammer + Noise)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Amplitude')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        # 4. LMS Recovery
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(time_vec, recovered_lms, 'b-', linewidth=0.8, label='LMS Recovered')
        ax4.plot(time_vec, clean, 'g--', linewidth=0.8, alpha=0.7, label='Clean')
        ax4.set_title('LMS Recovery', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Amplitude')
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        
        # 5. NLMS Recovery
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.plot(time_vec, recovered_nlms, 'purple', linewidth=0.8, label='NLMS Recovered')
        ax5.plot(time_vec, clean, 'g--', linewidth=0.8, alpha=0.7, label='Clean')
        ax5.set_title('NLMS Recovery', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Amplitude')
        ax5.legend(loc='upper right')
        ax5.grid(True, alpha=0.3)
        
        # 6. RLS Recovery
        ax6 = fig.add_subplot(gs[3, 0])
        ax6.plot(time_vec, recovered_rls, 'brown', linewidth=0.8, label='RLS Recovered')
        ax6.plot(time_vec, clean, 'g--', linewidth=0.8, alpha=0.7, label='Clean')
        ax6.set_title('RLS Recovery', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Amplitude')
        ax6.legend(loc='upper right')
        ax6.grid(True, alpha=0.3)
        
        # 7. Error signals
        ax7 = fig.add_subplot(gs[3, 1])
        error_lms = recovered_lms - clean
        error_nlms = recovered_nlms - clean
        error_rls = recovered_rls - clean
        
        ax7.plot(time_vec, error_lms, 'b-', linewidth=0.8, alpha=0.7, label='LMS Error')
        ax7.plot(time_vec, error_nlms, 'purple', linewidth=0.8, alpha=0.7, label='NLMS Error')
        ax7.plot(time_vec, error_rls, 'brown', linewidth=0.8, alpha=0.7, label='RLS Error')
        ax7.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        ax7.set_title('Recovery Errors', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('Amplitude')
        ax7.legend(loc='upper right')
        ax7.grid(True, alpha=0.3)
        
        plt.suptitle('Adaptive Anti-Jamming Signal Recovery Comparison', 
                     fontsize=14, fontweight='bold', y=0.995)
        
        filepath = self.output_dir / 'signal_comparison.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()
    
    def plot_convergence(self, mse_lms, mse_nlms, mse_rls, fs=16000):
        """
        Plot MSE convergence curves.
        
        Args:
            mse_lms: LMS MSE history
            mse_nlms: NLMS MSE history
            mse_rls: RLS MSE history
            fs: Sampling frequency
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        
        # Sample indices
        samples = np.arange(len(mse_lms))
        time_vec = samples / fs
        
        # Linear scale
        ax1.semilogy(time_vec, mse_lms, 'b-', linewidth=2, label='LMS')
        ax1.semilogy(time_vec, mse_nlms, 'purple', linewidth=2, label='NLMS')
        ax1.semilogy(time_vec, mse_rls, 'brown', linewidth=2, label='RLS')
        ax1.set_xlabel('Time (s)', fontsize=11)
        ax1.set_ylabel('MSE (log scale)', fontsize=11)
        ax1.set_title('Convergence Speed Comparison (Log Scale)', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, which='both')
        
        # Zoom into convergence region (first 30%)
        zoom_end = int(len(mse_lms) * 0.3)
        ax2.plot(time_vec[:zoom_end], mse_lms[:zoom_end], 'b-', linewidth=2, label='LMS')
        ax2.plot(time_vec[:zoom_end], mse_nlms[:zoom_end], 'purple', linewidth=2, label='NLMS')
        ax2.plot(time_vec[:zoom_end], mse_rls[:zoom_end], 'brown', linewidth=2, label='RLS')
        ax2.set_xlabel('Time (s)', fontsize=11)
        ax2.set_ylabel('MSE', fontsize=11)
        ax2.set_title('Convergence Detail (Initial 30% of Signal)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = self.output_dir / 'convergence.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()
    
    def plot_snr_comparison(self, snr_data):
        """
        Plot SNR comparison bar chart.
        
        Args:
            snr_data: Dictionary with algorithm names as keys and SNR data as values
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        algorithms = list(snr_data.keys())
        input_snr = [snr_data[alg]['input_snr_db'] for alg in algorithms]
        output_snr = [snr_data[alg]['output_snr_db'] for alg in algorithms]
        improvement = [snr_data[alg]['improvement_db'] for alg in algorithms]
        
        x = np.arange(len(algorithms))
        width = 0.25
        
        bars1 = ax.bar(x - width, input_snr, width, label='Input SNR', color='orange', alpha=0.8)
        bars2 = ax.bar(x, output_snr, width, label='Output SNR', color='green', alpha=0.8)
        bars3 = ax.bar(x + width, improvement, width, label='Improvement', color='blue', alpha=0.8)
        
        ax.set_xlabel('Algorithm', fontsize=11, fontweight='bold')
        ax.set_ylabel('SNR (dB)', fontsize=11, fontweight='bold')
        ax.set_title('SNR Performance Comparison Across Algorithms', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        filepath = self.output_dir / 'snr_comparison.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()
    
    def plot_frequency_response(self, signal_clean, signal_recovered_lms, 
                                signal_recovered_nlms, signal_recovered_rls, fs=16000):
        """
        Plot frequency domain analysis.
        
        Args:
            signal_clean: Clean signal
            signal_recovered_lms: LMS recovered signal
            signal_recovered_nlms: NLMS recovered signal
            signal_recovered_rls: RLS recovered signal
            fs: Sampling frequency
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Compute FFT
        freq = np.fft.fftfreq(len(signal_clean), 1/fs)
        
        def plot_spectrum(ax, signal, title, color):
            spectrum = np.abs(np.fft.fft(signal))
            ax.semilogy(freq[:len(freq)//2], spectrum[:len(spectrum)//2], 
                       color=color, linewidth=1.5)
            ax.set_xlabel('Frequency (Hz)', fontsize=10)
            ax.set_ylabel('Magnitude (dB)', fontsize=10)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, which='both')
        
        plot_spectrum(axes[0, 0], signal_clean, 'Clean Signal Spectrum', 'green')
        plot_spectrum(axes[0, 1], signal_recovered_lms, 'LMS Recovery Spectrum', 'blue')
        plot_spectrum(axes[1, 0], signal_recovered_nlms, 'NLMS Recovery Spectrum', 'purple')
        plot_spectrum(axes[1, 1], signal_recovered_rls, 'RLS Recovery Spectrum', 'brown')
        
        plt.suptitle('Frequency Domain Analysis', fontsize=13, fontweight='bold')
        plt.tight_layout()
        
        filepath = self.output_dir / 'frequency_response.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()
    
    def plot_mse_comparison(self, mse_data):
        """
        Plot MSE comparison.
        
        Args:
            mse_data: Dictionary with algorithm names and MSE values
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        algorithms = list(mse_data.keys())
        mse_before = [mse_data[alg]['mse_before'] for alg in algorithms]
        mse_after = [mse_data[alg]['mse_after'] for alg in algorithms]
        
        x = np.arange(len(algorithms))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, mse_before, width, label='Before Filtering', 
                       color='red', alpha=0.8)
        bars2 = ax.bar(x + width/2, mse_after, width, label='After Filtering', 
                       color='green', alpha=0.8)
        
        ax.set_xlabel('Algorithm', fontsize=11, fontweight='bold')
        ax.set_ylabel('MSE', fontsize=11, fontweight='bold')
        ax.set_title('Mean Squared Error Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_yscale('log')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2e}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        filepath = self.output_dir / 'mse_comparison.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()
    
    def plot_execution_time(self, timing_data):
        """
        Plot algorithm execution time comparison.
        
        Args:
            timing_data: Dictionary with algorithm names and execution times
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        algorithms = list(timing_data.keys())
        times = [timing_data[alg] for alg in algorithms]
        colors = ['blue', 'purple', 'brown']
        
        bars = ax.bar(algorithms, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Execution Time (seconds)', fontsize=11, fontweight='bold')
        ax.set_title('Computational Complexity Comparison', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        filepath = self.output_dir / 'execution_time.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()
