"""
Execution Time Measurement Module
"""

import time


class TimingContext:
    """Context manager for measuring execution time."""
    
    def __init__(self, name="Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        print(f"{self.name}: {self.elapsed_time:.4f} seconds")
    
    def get_time(self):
        """Get elapsed time in seconds."""
        if self.elapsed_time is not None:
            return self.elapsed_time
        elif self.start_time is not None and self.end_time is None:
            return time.time() - self.start_time
        return None


def measure_execution_time(func, *args, **kwargs):
    """
    Measure execution time of a function.
    
    Args:
        func: Function to measure
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        tuple: (result, execution_time)
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    
    return result, execution_time
