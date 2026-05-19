"""
Normalized Least Mean Squares (NLMS) Filter
Improved adaptive filter with adaptive step-size normalization.
"""

import numpy as np
import time


class NLMSFilter:
    """
    Normalized Least Mean Squares Adaptive Filter
    
    NLMS extends LMS by normalizing the step-size based on input signal power,
    providing better convergence and stability across different input conditions.
    
    Update rule: w(n+1) = w(n) + (mu / (||x||^2 + eps)) * e(n) * x(n)
    """
    
    def __init__(self, filter_order=64, mu=0.01, eps=1e-8):
        """
        Initialize NLMS filter.
        
        Args:
            filter_order (int): Length of adaptive filter weights
            mu (float): Step-size parameter (0.1-1.0, typically 0.5)
                        Normalized, so larger values can be used than LMS
            eps (float): Small value for numerical stability
        """
        self.filter_order = filter_order
        self.mu = mu
        self.eps = eps
        
        # Initialize weights
        self.weights = np.zeros(filter_order)
        self.weight_history = []
        self.error_history = []
        self.normalization_factors = []
        
    def update(self, reference_signal, desired_signal):
        """
        Adapt filter weights using NLMS algorithm.
        
        Args:
            reference_signal (ndarray): Reference input signal
            desired_signal (ndarray): Desired signal to match
            
        Returns:
            dict: Filtering results containing:
                - 'estimated_noise': Estimated noise signal
                - 'error': Error signal
                - 'recovered': Recovered signal (desired - estimated_noise)
                - 'convergence': Convergence metrics
        """
        n_samples = len(reference_signal)
        estimated_noise = np.zeros(n_samples)
        error = np.zeros(n_samples)
        
        start_time = time.time()
        
        # Process each sample
        for n in range(n_samples):
            # Get filter input vector (past reference samples)
            x_n = np.flip(reference_signal[max(0, n-self.filter_order+1):n+1])
            
            # Pad with zeros if needed
            if len(x_n) < self.filter_order:
                x_n = np.concatenate([x_n, np.zeros(self.filter_order - len(x_n))])
            
            # Compute filter output
            y_n = np.dot(self.weights, x_n)
            estimated_noise[n] = y_n
            
            # Compute error
            e_n = desired_signal[n] - y_n
            error[n] = e_n
            
            # Compute normalization factor (input signal power)
            norm_factor = np.dot(x_n, x_n) + self.eps
            
            # Adaptive step-size
            mu_n = self.mu / norm_factor
            
            # Update weights (NLMS rule)
            self.weights = self.weights + mu_n * e_n * x_n
            
            # Store history every 100 samples
            if n % 100 == 0:
                self.weight_history.append(self.weights.copy())
                self.error_history.append(e_n)
                self.normalization_factors.append(norm_factor)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Compute recovered signal
        recovered_signal = desired_signal - estimated_noise
        
        # Compute convergence metrics
        mse_history = np.array([e**2 for e in error])
        
        results = {
            'estimated_noise': estimated_noise.astype(np.float32),
            'recovered': recovered_signal.astype(np.float32),
            'error': error.astype(np.float32),
            'convergence': {
                'final_mse': np.mean(mse_history[-1000:]) if len(mse_history) > 1000 else np.mean(mse_history),
                'mse_history': mse_history,
                'execution_time': execution_time,
                'final_weights': self.weights.copy(),
                'avg_norm_factor': np.mean(self.normalization_factors) if self.normalization_factors else 0
            }
        }
        
        return results
    
    def get_parameters(self):
        """Get filter parameters."""
        return {
            'filter_order': self.filter_order,
            'mu': self.mu,
            'weights': self.weights.copy()
        }
    
    def reset(self):
        """Reset filter to initial state."""
        self.weights = np.zeros(self.filter_order)
        self.weight_history = []
        self.error_history = []
        self.normalization_factors = []
