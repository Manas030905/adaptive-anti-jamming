# Adaptive Anti-Jamming Signal Recovery

## Overview

A **signal processing system** that recovers clean communication signals corrupted by jamming interference using three adaptive filtering algorithms:
- **LMS** - Fast, simple, moderate performance
- **NLMS** - Balanced, robust, good performance
- **RLS** - Slow, optimal, excellent performance

**Perfect for:** Research, education, anti-jamming system design

---

## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run pipeline
python main.py

# 3. View results in outputs/
```

**No audio needed** - system auto-generates test signal!

---

## 📁 Project Structure

```
adaptive_anti_jamming/
├── main.py                  # Main pipeline
├── config.py                # Configuration
├── requirements.txt         # Dependencies
├── signals/                 # Signal generation
│   ├── audio_loader.py
│   ├── jammer_generator.py
│   ├── channel_model.py
│   └── reference_generator.py
├── filters/                 # 3 adaptive filters
│   ├── lms_filter.py
│   ├── nlms_filter.py
│   └── rls_filter.py
├── metrics/                 # Performance metrics
│   ├── snr.py
│   ├── mse.py
│   └── execution_time.py
├── visualization/           # Plots
│   └── plots.py
├── experiments/             # Comparisons
│   └── compare_algorithms.py
├── audio/                   # Input audio
├── outputs/                 # Generated results
└── README.md
```

---

## 🔬 The Three Algorithms

| Algorithm | Formula | Speed | Convergence | Best For |
|-----------|---------|-------|-------------|----------|
| **LMS** | w = w + μ·e·x | ⚡⚡⚡ | Slow | Real-time |
| **NLMS** | w = w + (μ/\|\|x\|\|²)·e·x | ⚡⚡ | Medium | General use |
| **RLS** | Uses gain vector | ⚡ | Fast ⭐ | Research |

**LMS:** Simple gradient descent, easiest to implement  
**NLMS:** Normalized step-size, more robust  
**RLS:** Optimal in least-squares sense, best performance but slowest

---

## 🎯 Jammer Types

| Type | Characteristics | Frequency |
|------|-----------------|-----------|
| **Tone** | Single frequency interference | 180 Hz |
| **Noise** | Broadband random interference | Full spectrum |
| **Sweep** | Frequency chirp | 50-500 Hz |
| **Burst** | High-energy pulses | Intermittent |
| **Combined** | Realistic mix of all types | Multiple |

---

## � Performance Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| **SNR Improvement** | Output SNR - Input SNR (dB) | Signal quality gain |
| **MSE** | Mean squared error | Recovery accuracy |
| **MSE Reduction** | (MSE_in - MSE_out)/MSE_in % | Error decrease |
| **PESQ Score** | Perceptual quality (0-5) | Audio quality |
| **Execution Time** | Runtime in seconds | Computational cost |

---

## ✨ Features

✅ **Three adaptive filter implementations** (LMS, NLMS, RLS)  
✅ **Five jammer types** with realistic channel modeling  
✅ **Six performance metrics** (SNR, MSE, PESQ, etc.)  
✅ **Rich visualization** (6 plot types automatically generated)  
✅ **Comparative experiments** across algorithms and jammer types  
✅ **CSV export** for further analysis  
✅ **Auto-generates test signal** if no audio provided  
✅ **Configurable parameters** (filter order, step-size, duration, etc.)

---

## 📁 Output Files

**Audio** - `recovered_lms.wav`, `recovered_nlms.wav`, `recovered_rls.wav`  
**Plots** - Signal comparison, convergence, SNR, MSE, frequency, execution time  
**Data** - `metrics.csv`, `comparison_results.csv`

---

## ⚙️ Customization

Edit parameters in `main.py` or `config.py`:
- **FILTER_ORDER** - Adaptive filter length (default: 64)
- **DURATION** - Signal duration in seconds (default: 3.0)
- **FS** - Sampling frequency Hz (default: 16000)
- **LMS_MU** - LMS step-size (default: 0.01)
- **NLMS_MU** - NLMS step-size (default: 0.5)
- **RLS_LAMBDA** - RLS forgetting factor (default: 0.99)

For custom audio: Place file at `audio/clean_audio.wav`

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Import errors | Run `pip install -r requirements.txt` |
| No audio file | System auto-generates test signal |
| Plots not saving | `outputs/` folder auto-created |
| Poor SNR | Increase filter order, adjust step-sizes |
| Slow RLS | Use shorter signal or reduce filter order |

---

## 📚 Resources

See inline docstrings in source code for detailed implementation notes.

For algorithm theory:
- **LMS/NLMS:** Widrow & Stearns, *Adaptive Signal Processing*
- **RLS:** Sayed & Kailath, *A Fast H-infinity Tracking Algorithm*

---

## � About

**Research-oriented signal processing project** - Implements adaptive filtering for anti-jamming systems. Perfect for academic work, research, education, and communication system development.

**Dependencies:** Python 3.8+, numpy, scipy, matplotlib, pandas, soundfile  
**Status:** Production ready  
**License:** Academic/Research use

---

## 📖 Further Development

Consider extending with:
- Blind channel estimation
- Multi-channel processing
- Real-time streaming
- GPU acceleration
- Kalman filter variants
- Additional jammer models
