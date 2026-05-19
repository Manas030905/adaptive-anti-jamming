# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Pipeline
```bash
python main.py
```

### Step 3: Check Results
View the outputs in the `outputs/` folder:
- `signal_comparison.png` - Visual comparison of all signals
- `convergence.png` - Algorithm convergence behavior
- `snr_comparison.png` - Performance comparison
- `metrics.csv` - Detailed performance metrics

---

## 📁 Project Contents

| Folder | Purpose |
|--------|---------|
| `signals/` | Audio loading, jammer generation, channel modeling |
| `filters/` | LMS, NLMS, and RLS adaptive filter implementations |
| `metrics/` | SNR, MSE, and performance calculations |
| `visualization/` | Plotting and result visualization |
| `experiments/` | Comparative algorithm testing |
| `audio/` | (Optional) Place your clean_audio.wav here |
| `outputs/` | Generated results and visualizations |

---

## 🔧 Key Components

### Audio Loader (`signals/audio_loader.py`)
- Loads WAV files
- Converts stereo to mono
- Normalizes amplitude
- Generates test signals if needed

### Jammer Generator (`signals/jammer_generator.py`)
- Tone jammer (180 Hz narrowband)
- White noise jammer (broadband)
- Sweep jammer (50-500 Hz chirp)
- Burst jammer (intermittent pulses)
- Combined jammer (realistic mix)

### Adaptive Filters (`filters/`)
- **LMS**: Fast, simple, slower convergence
- **NLMS**: Balanced, normalized step-size
- **RLS**: Fastest convergence, highest complexity

### Performance Metrics (`metrics/`)
- SNR improvement calculation
- MSE reduction analysis
- Signal quality scoring
- Execution time tracking

---

## 📊 Understanding the Output

### Signal Comparison Plot
8 subplots showing:
1. Clean signal
2. Jammer signal
3. Received (corrupted)
4. LMS recovery
5. NLMS recovery
6. RLS recovery
7. Error signals

### Convergence Plot
- Log-scale MSE curves showing algorithm convergence speed
- Detail zoom on initial convergence phase

### SNR Comparison
Bar chart showing:
- Input SNR (before filtering)
- Output SNR (after filtering)
- SNR improvement

### Metrics CSV
```
Algorithm | Input_SNR | Output_SNR | Improvement | MSE_After | Time
----------|-----------|-----------|------------|-----------|------
LMS       | 12.5      | 26.8      | 14.3       | 0.0012    | 0.045
NLMS      | 12.5      | 28.1      | 15.6       | 0.0008    | 0.052
RLS       | 12.5      | 29.7      | 17.2       | 0.0005    | 0.127
```

---

## 🎯 Typical Results

| Algorithm | Convergence | Performance | Speed | Best For |
|-----------|------------|------------|-------|----------|
| **LMS** | Slow | Moderate | ⚡⚡⚡ Fast | Real-time, embedded systems |
| **NLMS** | Medium | Good | ⚡⚡ Medium | General purpose |
| **RLS** | Fast | Excellent | ⚡ Slow | Offline analysis, research |

---

## 🔗 Core Algorithm Differences

### LMS (Basic Gradient Descent)
```
w(n+1) = w(n) + μ × e(n) × x(n)
```
- Simple, low complexity
- Sensitive to input power variations
- May diverge with wrong μ

### NLMS (Normalized LMS)
```
w(n+1) = w(n) + [μ / (||x||² + ε)] × e(n) × x(n)
```
- Normalizes by input power
- More robust than LMS
- Moderate complexity

### RLS (Optimal Least Squares)
```
Uses inverse correlation matrix update
```
- Exponential convergence
- Optimal performance
- Highest computational cost

---

## ✅ Verification

After running `python main.py`, verify these files exist:

**Audio Output:**
- ✓ `outputs/recovered_lms.wav`
- ✓ `outputs/recovered_nlms.wav`
- ✓ `outputs/recovered_rls.wav`
- ✓ `outputs/jammed_signal.wav`

**Plots:**
- ✓ `outputs/signal_comparison.png`
- ✓ `outputs/convergence.png`
- ✓ `outputs/snr_comparison.png`
- ✓ `outputs/mse_comparison.png`
- ✓ `outputs/frequency_response.png`
- ✓ `outputs/execution_time.png`

**Data:**
- ✓ `outputs/metrics.csv`
- ✓ `outputs/comparison_results.csv`

---

## 🎓 What to Learn

1. **Signal Processing**: Jammer generation, channel modeling, filtering
2. **Adaptive Algorithms**: How LMS, NLMS, RLS differ and when to use each
3. **Performance Analysis**: SNR, MSE, convergence metrics
4. **Experimental Design**: Systematic testing and comparison
5. **Visualization**: Creating publication-quality plots

---

## 🛠️ Customization

Edit `main.py` to change:
```python
FILTER_ORDER = 64          # Filter length (increase for better performance)
DURATION = 3.0             # Signal duration
FS = 16000                 # Sampling rate
```

Edit filter parameters:
```python
# In main.py, modify filter initialization:
lms_filter = LMSFilter(filter_order=64, mu=0.01)      # Adjust mu
nlms_filter = NLMSFilter(filter_order=64, mu=0.5)     # Adjust mu
rls_filter = RLSFilter(filter_order=64, lambda_param=0.99)  # Adjust lambda
```

---

## 🐛 Common Issues

| Problem | Solution |
|---------|----------|
| "ModuleNotFoundError" | Run `pip install -r requirements.txt` |
| "No audio file" | System auto-generates test signal, or add to `audio/` |
| Slow execution | Reduce filter order or duration |
| Poor SNR improvement | Increase filter order, adjust step-sizes |
| No output files | Check `outputs/` directory permissions |

---

## 📚 Next Steps

1. **Run the pipeline** - `python main.py`
2. **Review results** - Check `outputs/` folder
3. **Study metrics** - Open `outputs/metrics.csv`
4. **Analyze plots** - View PNG visualizations
5. **Modify parameters** - Experiment with filter orders and step-sizes
6. **Try different jammers** - See how algorithms respond to different interference types

---

## 🎯 Key Takeaways

- ✅ LMS: Simple but needs careful tuning
- ✅ NLMS: Balanced performance and complexity
- ✅ RLS: Best performance but computationally expensive
- ✅ Filter order: Higher = better performance, but slower
- ✅ Reference signal: Quality affects recovery performance

---

**Ready to get started? Run:** `python main.py`
