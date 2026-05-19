"""
Configuration File for Adaptive Anti-Jamming System
Modify parameters here for different experiments and scenarios.
"""

import os

# ==========================================
# SIGNAL PROCESSING PARAMETERS
# ==========================================

# Audio configuration
AUDIO_CONFIG = {
    'input_path': 'audio/clean_audio.wav',
    'generate_if_missing': True,      # Auto-generate if no file
    'duration': 3.0,                  # Duration in seconds
    'sampling_frequency': 16000,      # Hz
    'normalize': True,                 # Normalize amplitude
}

# Channel configuration
CHANNEL_CONFIG = {
    'input_snr_db': 15.0,             # SNR of received signal (dB)
    'add_noise': True,                # Add Gaussian noise
    'multipath': False,               # Enable multipath fading
}

# ==========================================
# JAMMER CONFIGURATION
# ==========================================

JAMMER_CONFIG = {
    'primary_type': 'combined',       # 'tone', 'noise', 'sweep', 'burst', 'combined'
    
    # Tone jammer parameters
    'tone_frequency': 180,            # Hz
    'tone_amplitude': 0.2,            # 0-1
    
    # Noise jammer parameters
    'noise_level': 0.15,              # 0-1
    
    # Sweep jammer parameters
    'sweep_freq_start': 50,           # Hz
    'sweep_freq_end': 500,            # Hz
    'sweep_amplitude': 0.2,           # 0-1
    
    # Burst jammer parameters
    'burst_duration': 0.1,            # seconds
    'burst_amplitude': 0.8,           # 0-1
    'burst_interval': 0.5,            # seconds between bursts
    
    # Combined jammer (when primary_type='combined')
    'combined_tone_amp': 0.2,
    'combined_noise_level': 0.15,
    'combined_sweep_amp': 0.2,
    'combined_burst_amp': 0.15,
}

# ==========================================
# ADAPTIVE FILTER PARAMETERS
# ==========================================

FILTER_CONFIG = {
    # Common
    'filter_order': 64,               # Number of filter taps
    
    # LMS parameters
    'lms': {
        'mu': 0.01,                   # Step-size (learning rate)
        'description': 'Least Mean Squares',
    },
    
    # NLMS parameters
    'nlms': {
        'mu': 0.5,                    # Normalized step-size (0.1-1.0)
        'eps': 1e-8,                  # Numerical stability
        'description': 'Normalized LMS',
    },
    
    # RLS parameters
    'rls': {
        'lambda_param': 0.99,         # Forgetting factor (0.9-1.0)
        'delta': 1.0,                 # P matrix initialization
        'eps': 1e-8,                  # Numerical stability
        'description': 'Recursive Least Squares',
    },
}

# ==========================================
# REFERENCE SIGNAL CONFIGURATION
# ==========================================

REFERENCE_CONFIG = {
    'correlation': 0.9,               # Correlation with jammer (0.5-1.0)
    'noise_variance': 0.01,           # Random perturbation level
    'bandlimit': False,               # Apply band-limiting filter
    'cutoff_frequency': 2000,         # Hz
}

# ==========================================
# METRICS & ANALYSIS
# ==========================================

METRICS_CONFIG = {
    'segment_length': 512,            # For segmental SNR analysis
    'pesq_weight': 1.0,               # PESQ importance in quality metric
    'compute_detailed': True,         # Compute all detailed metrics
}

# ==========================================
# VISUALIZATION
# ==========================================

VISUALIZATION_CONFIG = {
    'output_directory': 'outputs',
    'figure_dpi': 150,                # Resolution of PNG files
    'figure_quality': 'tight',        # Bounding box ('tight' or 'loose')
    'colormap': 'viridis',            # Matplotlib colormap
    
    # Plot configuration
    'signal_plot_time_range': None,   # (start_sec, end_sec) or None for full
    'convergence_zoom_percent': 30,   # Show first 30% zoomed
    'show_grid': True,                # Show grid on plots
    'show_legends': True,             # Show legend labels
    
    # Font sizes
    'title_fontsize': 12,
    'label_fontsize': 11,
    'legend_fontsize': 10,
}

# ==========================================
# EXPERIMENTAL DESIGN
# ==========================================

EXPERIMENT_CONFIG = {
    'jammer_types': [
        'tone',
        'noise',
        'sweep',
        'burst',
        'combined'
    ],
    
    'filter_orders': [32, 64, 128],   # Orders to test
    
    'lms_mu_sweep': [0.001, 0.005, 0.01, 0.02],
    
    'run_stability_analysis': True,
    'run_comparative_experiments': True,
    'run_convergence_analysis': True,
}

# ==========================================
# AUDIO OUTPUT
# ==========================================

AUDIO_OUTPUT_CONFIG = {
    'format': 'WAV',                  # Audio format
    'bit_depth': 16,                  # 16, 24, or 32 bits
    'normalize': True,                # Normalize to prevent clipping
    'save_intermediate': False,       # Save intermediate signals (jammed, etc.)
}

# ==========================================
# LOGGING & REPORTING
# ==========================================

LOGGING_CONFIG = {
    'verbose': True,                  # Print detailed progress
    'log_to_file': False,             # Save log to file
    'log_filename': 'results.log',
    'export_csv': True,               # Export results to CSV
    'export_comparison': True,        # Export comparison table
}

# ==========================================
# STABILITY PARAMETERS
# ==========================================

STABILITY_CONFIG = {
    'test_filter_orders': [32, 64, 128, 256],
    'compute_mu_max': True,           # Compute maximum stable step-size
    'compute_eigen_analysis': False,  # Analyze eigenvalues
    'check_divergence': True,         # Monitor for divergence
}

# ==========================================
# ADVANCED OPTIONS
# ==========================================

ADVANCED_CONFIG = {
    # Performance optimization
    'use_numpy_optimization': True,
    'use_scipy_optimization': True,
    
    # Numerical stability
    'epsilon': 1e-8,
    'max_weight_magnitude': 100,
    
    # Debugging
    'save_intermediate_weights': False,
    'save_intermediate_errors': False,
    'debug_mode': False,
}

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_filter_config(algorithm_name):
    """Get configuration for specific algorithm."""
    algorithm_name = algorithm_name.upper()
    if algorithm_name in FILTER_CONFIG:
        return FILTER_CONFIG[algorithm_name]
    return FILTER_CONFIG['lms']


def get_jammer_config(jammer_type):
    """Get configuration for specific jammer type."""
    config = {
        'frequency': JAMMER_CONFIG.get('tone_frequency', 180),
        'amplitude': JAMMER_CONFIG.get('tone_amplitude', 0.2),
        'level': JAMMER_CONFIG.get('noise_level', 0.15),
        'freq_start': JAMMER_CONFIG.get('sweep_freq_start', 50),
        'freq_end': JAMMER_CONFIG.get('sweep_freq_end', 500),
    }
    return config


def validate_config():
    """Validate configuration parameters."""
    errors = []
    
    # Filter order
    if FILTER_CONFIG['filter_order'] < 1 or FILTER_CONFIG['filter_order'] > 1024:
        errors.append("Filter order must be between 1 and 1024")
    
    # LMS step-size
    if FILTER_CONFIG['lms']['mu'] <= 0 or FILTER_CONFIG['lms']['mu'] > 0.1:
        errors.append("LMS mu should be between 0 and 0.1")
    
    # NLMS step-size
    if FILTER_CONFIG['nlms']['mu'] <= 0 or FILTER_CONFIG['nlms']['mu'] > 2:
        errors.append("NLMS mu should be between 0 and 2")
    
    # RLS forgetting factor
    if FILTER_CONFIG['rls']['lambda_param'] <= 0.5 or FILTER_CONFIG['rls']['lambda_param'] >= 1:
        errors.append("RLS lambda should be between 0.5 and 1")
    
    # Reference correlation
    if REFERENCE_CONFIG['correlation'] < 0 or REFERENCE_CONFIG['correlation'] > 1:
        errors.append("Reference correlation must be between 0 and 1")
    
    if errors:
        print("Configuration errors found:")
        for error in errors:
            print(f"  ✗ {error}")
        return False
    return True


# ==========================================
# QUICK PRESETS
# ==========================================

PRESETS = {
    'quick_test': {
        'duration': 1.0,
        'filter_order': 32,
        'run_comparative_experiments': False,
    },
    
    'balanced': {
        'duration': 3.0,
        'filter_order': 64,
        'run_comparative_experiments': True,
    },
    
    'high_precision': {
        'duration': 5.0,
        'filter_order': 256,
        'lms_mu': 0.005,
        'nlms_mu': 0.3,
        'rls_lambda': 0.999,
        'run_comparative_experiments': True,
    },
    
    'real_time': {
        'duration': 2.0,
        'filter_order': 32,
        'lms_mu': 0.02,
        'nlms_mu': 1.0,
        'figure_dpi': 72,
    },
}


def apply_preset(preset_name):
    """Apply a configuration preset."""
    if preset_name not in PRESETS:
        print(f"Unknown preset: {preset_name}")
        return False
    
    preset = PRESETS[preset_name]
    print(f"Applying preset: {preset_name}")
    print(f"  Configuration: {preset}")
    return True


# ==========================================
# EXPORT & IMPORT
# ==========================================

def export_config_to_dict():
    """Export all configuration to dictionary."""
    return {
        'audio': AUDIO_CONFIG,
        'channel': CHANNEL_CONFIG,
        'jammer': JAMMER_CONFIG,
        'filter': FILTER_CONFIG,
        'reference': REFERENCE_CONFIG,
        'metrics': METRICS_CONFIG,
        'visualization': VISUALIZATION_CONFIG,
        'experiment': EXPERIMENT_CONFIG,
        'audio_output': AUDIO_OUTPUT_CONFIG,
        'logging': LOGGING_CONFIG,
        'advanced': ADVANCED_CONFIG,
    }


if __name__ == "__main__":
    print("Configuration validation...")
    if validate_config():
        print("✓ All configuration parameters are valid")
    else:
        print("✗ Configuration has errors - please fix")
    
    print("\nActive Configuration Summary:")
    print(f"  Duration: {AUDIO_CONFIG['duration']} sec")
    print(f"  Fs: {AUDIO_CONFIG['sampling_frequency']} Hz")
    print(f"  Filter Order: {FILTER_CONFIG['filter_order']}")
    print(f"  Jammer Type: {JAMMER_CONFIG['primary_type']}")
    print(f"  Input SNR: {CHANNEL_CONFIG['input_snr_db']} dB")
