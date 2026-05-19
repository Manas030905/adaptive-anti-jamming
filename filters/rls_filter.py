"""
Recursive Least Squares (RLS) Filter
Advanced adaptive filter with fast convergence.
"""

import numpy as np
import time


class RLSFilter:
    """
    Recursive Least Squares Adaptive Filter
    
    RLS provides faster convergence than LMS/NLMS at the cost of higher
    computational complexity. Uses inverse correlation matrix for optimal
    step-size adaptation.
    
    Update rule incorporates: P(n) = correlation matrix inverse
                              g(n) = gain vector
    """
    
    def __init__(self, filter_order=16, lambda_param=0.95, delta=1.0, eps=1e-8):
        """
        Initialize RLS filter.
        
        Args:
            filter_order (int): Length of adaptive filter weights
            lambda_param (float): Forgetting factor (0.9-1.0)
                                 Lower values emphasize recent samples
            delta (float): Initialization parameter for P matrix (typically 1.0)
            eps (float): Small value for numerical stability
        """
        self.filter_order = filter_order
        self.lambda_param = lambda_param
        self.delta = delta
        self.eps = eps
        
        # Initialize weights
        self.weights = np.zeros(filter_order)
        
        # Initialize inverse correlation matrix P
        self.P = (1.0 / delta) * np.eye(filter_order)
        
        self.weight_history = []
        self.error_history = []
        self.gain_history = []
        
    def update(self, reference_signal, desired_signal):
        """
        Adapt filter weights using RLS algorithm.
        
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
            
            # RLS update
            # Compute gain vector
            numerator = np.dot(self.P, x_n)
            denominator = self.lambda_param + np.dot(x_n, numerator) + self.eps
            g_n = numerator / denominator
            
            # Store gain magnitude
            if n % 100 == 0:
                self.gain_history.append(np.linalg.norm(g_n))
            
            # Update weights
            self.weights = self.weights + g_n * e_n
            
            # Update inverse correlation matrix P
            self.P = (1.0 / self.lambda_param) * (
                self.P - np.outer(g_n, np.dot(x_n, self.P))
            )
            
            # Store history every 100 samples
            if n % 100 == 0:
                self.weight_history.append(self.weights.copy())
                self.error_history.append(e_n)
        
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
                'final_P_condition': np.linalg.cond(self.P) if self.P.size > 0 else 0
            }
        }
        
        return results
    
    def get_parameters(self):
        """Get filter parameters."""
        return {
            'filter_order': self.filter_order,
            'lambda': self.lambda_param,
            'delta': self.delta,
            'weights': self.weights.copy()
        }
    
    def reset(self):
        """Reset filter to initial state."""
        self.weights = np.zeros(self.filter_order)
        self.P = (1.0 / self.delta) * np.eye(self.filter_order)
        self.weight_history = []
        self.error_history = []
        self.gain_history = []
