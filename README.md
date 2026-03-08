# Human Activity Recognition using Hidden Markov Models

A machine learning project that recognizes human activities (Still, Standing, Walking, Jumping) using accelerometer and gyroscope sensor data from smartphones, implementing Hidden Markov Models (HMM) with Baum-Welch training and Viterbi decoding.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)

##  Overview

This project implements a **Hidden Markov Model (HMM)** from scratch using NumPy to classify human activities based on inertial sensor data. The system processes accelerometer and gyroscope readings from smartphones, extracts time-domain and frequency-domain features, and uses probabilistic temporal modeling to predict activity sequences.

### Recognized Activities
- **Still** - No movement
- **Standing** - Stationary with micro-movements
- **Walking** - Regular gait pattern
- **Jumping** - High-intensity vertical movement

### Sensors Used
- **Accelerometer** (x, y, z axes) - Linear acceleration in m/s²
- **Gyroscope** (x, y, z axes) - Angular velocity in rad/s

## Features

- **Full HMM Implementation**: Baum-Welch (Expectation-Maximization) training and Viterbi decoding
- **9 Engineered Features**: Time-domain (RMS, variance, correlation) and frequency-domain (FFT-based) features
- **Signal Resampling**: Harmonizes different sensor sampling rates to 50 Hz
- **Windowing**: 2-second sliding windows with 50% overlap
- **Visualization**: Comprehensive plots for raw signals, frequency analysis, confusion matrices, and more
- **Evaluation Metrics**: Accuracy, sensitivity, specificity, confusion matrix, classification report

## Requirements

### Python Dependencies
```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
scikit-learn>=0.24.0
```

### Optional (for Google Colab)
```
google-colab
```

## Installation

1. **Clone or download this repository**
   ```bash
   cd ML-techniques-II/formative-2
   ```

2. **Install required packages**
   ```bash
   pip install numpy pandas matplotlib seaborn scipy scikit-learn
   ```

3. **Verify installation**
   ```python
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   print("All packages installed successfully!")
   ```

## Dataset Structure

Organize your data in the following structure:

```
formative-2/
├── dataset/
│   ├── train/
│   │   ├── still_1.csv
│   │   ├── still_2.csv
│   │   ├── standing_1.csv
│   │   ├── standing_2.csv
│   │   ├── walking_1.csv
│   │   ├── walking_2.csv
│   │   ├── jumping_1.csv
│   │   └── jumping_2.csv
│   └── test/
│       ├── still_1.csv
│       ├── standing_1.csv
│       ├── walking_1.csv
│       └── jumping_1.csv
├── outputs/              # Generated automatically for plots
├── hmm_activity_recognition_v2.ipynb
├── file_merging.py
└── README.md
```

### CSV File Format

Each CSV file should contain merged accelerometer and gyroscope data with the following columns:

```
time, seconds_elapsed, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
```

**Example:**
```csv
time,seconds_elapsed,a_x,accel_y,a_z,g_x,g_y,g_z
2024-03-08 10:15:23.123,0.000,-0.123,9.801,0.456,-0.012,0.034,-0.001
2024-03-08 10:15:23.143,0.020,-0.145,9.798,0.478,-0.015,0.031,0.002
...
```

## Data Preparation

### 1. Recording Data

Use the **Sensor Logger** app (Android/iOS) to record sensor data:
- Enable both **Accelerometer** and **Gyroscope**
- Record each activity for at least 10-30 seconds
- Export data as CSV files

### 2. Merging Accelerometer and Gyroscope Files

If you have separate files for each sensor, use the provided `file_merging.py` script:

```python
python file_merging.py
```

This script:
- Finds all accelerometer CSV files
- Matches them with corresponding gyroscope files
- Merges them into single CSV files with proper column naming

**Note:** Update the `folder_path` variable in `file_merging.py` to point to your data folder.

### 3. Naming Convention

Name your files following this pattern:
```
{activity}_{number}.csv
```

Examples:
- `still_1.csv`, `still_2.csv`
- `walking_1.csv`, `walking_2.csv`
- `jumping_1.csv`, `jumping_2.csv`

## Usage

### Running the Notebook

1. **Open Jupyter Notebook or Google Colab**
   ```bash
   jupyter notebook hmm_activity_recognition.ipynb
   ```

2. **For Google Colab:**
   - Uncomment the Google Drive mounting cells
   - Upload your dataset to Google Drive
   - Update paths accordingly

3. **For Local Execution:**
   - The notebook is configured to use relative paths
   - Ensure your `dataset/train` and `dataset/test` folders are in place
   - Run all cells sequentially

### Pipeline Steps

The notebook follows this pipeline:

1. **Data Loading** (Section 2)
   - Loads CSV files from train/test directories
   - Resamples all data to 50 Hz using linear interpolation

2. **Visualization** (Sections 3-4)
   - Raw sensor signal plots
   - FFT frequency analysis per activity

3. **Feature Extraction** (Section 5)
   - Extracts 9 features from 2-second windows with 50% overlap

4. **Normalization** (Section 6)
   - Z-score normalization (StandardScaler)
   - Feature distribution analysis

5. **HMM Training** (Sections 7-8)
   - Baum-Welch algorithm for parameter estimation
   - Convergence monitoring

6. **State Mapping** (Section 9)
   - Aligns HMM states to activity labels

7. **Evaluation** (Sections 10-13)
   - Viterbi decoding
   - Confusion matrices
   - Sensitivity and specificity metrics

8. **Analysis** (Section 15)
   - Interpretation of results
   - Suggestions for improvements

## Methodology

### Signal Processing

**Sampling Rate Harmonization:**
- Target: 50 Hz (20 ms per sample)
- Method: Linear interpolation based on `seconds_elapsed` timestamps
- Ensures consistent feature extraction across different devices

**Windowing:**
- Window size: 2 seconds (100 samples @ 50 Hz)
- Overlap: 50% (50 sample step)
- Rationale: Captures full gait/jump cycles while maintaining temporal resolution

### Feature Engineering

| # | Feature | Domain | Description |
|---|---------|--------|-------------|
| 1 | RMS Accel | Time | Root mean square of acceleration magnitude |
| 2 | Var Accel | Time | Variance of acceleration magnitude |
| 3 | SMA | Time | Signal Magnitude Area (sum of absolute values) |
| 4 | Mean Gyro | Time | Mean gyroscope magnitude |
| 5 | Var Gyro | Time | Variance of gyroscope magnitude |
| 6 | Corr ax-ay | Time | Cross-correlation between x and y acceleration |
| 7 | Dom Freq | Frequency | Dominant frequency from FFT |
| 8 | Spec Energy | Frequency | Spectral energy in 0.5-5 Hz band |
| 9 | Spec Entropy | Frequency | Shannon entropy of power spectrum |

### Hidden Markov Model

**Architecture:**
- States: 4 (one per activity)
- Emissions: Gaussian with diagonal covariance
- Initialization: Activity-aware (using training labels)
- Transition matrix: Self-loop dominant (activities persist)

**Training:**
- Algorithm: Baum-Welch (EM)
- Convergence: ΔLL < 10⁻⁴
- Max iterations: 300

**Decoding:**
- Algorithm: Viterbi (log-space dynamic programming)
- Output: Most likely state sequence

## Project Structure

```
formative-2/
│
├── dataset/
│   ├── train/           # Training data CSVs
│   └── test/            # Test data CSVs
│
├── outputs/             # Generated plots (auto-created)
│   ├── fig1_raw_signals.png
│   ├── fig2_fft_per_activity.png
│   ├── fig3_feature_distributions.png
│   ├── fig4_feature_correlation.png
│   ├── fig5_baumwelch_convergence.png
│   ├── fig6_transition_and_pi.png
│   ├── fig7_emission_params.png
│   ├── fig8_viterbi_train.png
│   ├── fig9_viterbi_test.png
│   ├── fig10_confusion_train.png
│   ├── fig11_metrics_train.png
│   ├── fig12_confusion_test.png
│   └── fig13_metrics_test.png
│
├── hmm_activity_recognition.ipynb  # Main notebook
├── file_merging.py                     # Utility to merge sensor files
└── README.md                           # This file
```

## Results

### Expected Performance

With sufficient training data (≥10 recordings per activity):

| Activity | Distinguishability | Expected Accuracy |
|----------|-------------------|-------------------|
| Still | Easiest | > 95% |
| Jumping | Easy | > 90% |
| Walking | Moderate | > 85% |
| Standing | Hardest | > 80% |

### Key Findings

1. **Still vs. Others**: Easiest separation due to near-zero variance
2. **Walking**: Distinctive periodic pattern (1-2 Hz)
3. **Jumping**: High RMS and dominant frequency (2-3 Hz)
4. **Standing vs. Still**: Most challenging due to overlap in feature space

### Visualization Outputs

All plots are automatically saved to the `outputs/` directory:
- Raw sensor signals
- Frequency spectra
- Feature distributions and correlations
- HMM parameters (transition matrix, emissions)
- Confusion matrices
- Performance metrics

## Troubleshooting

### Common Issues

**1. "Missing columns" error:**
- Ensure CSV files have all required columns: `seconds_elapsed`, `accel_x/y/z`, `gyro_x/y/z`
- Check column naming (case-insensitive, underscores/spaces handled)

**2. "No files loaded":**
- Verify dataset folder structure matches the expected layout
- Check file naming convention: `{activity}_{number}.csv`
- Ensure activity names match exactly: `still`, `standing`, `walking`, `jumping`

**3. Poor accuracy:**
- Collect more training data (aim for ≥10 recordings per activity)
- Ensure proper activity execution (e.g., consistent walking pace)
- Check sensor placement (phone in pocket/hand consistently)

**4. Import errors:**
- Install missing packages: `pip install <package_name>`
- For Colab, most packages are pre-installed

## Future Improvements

- [ ] Add more activities (running, cycling, stairs)
- [ ] Implement mixture-of-Gaussians emissions
- [ ] Add cross-participant validation
- [ ] Include magnetometer data
- [ ] Optimize with Cython/Numba for speed
- [ ] Deploy as mobile app
- [ ] Real-time activity recognition
- [ ] Add transition activity detection

## References

- **HMM Theory**: Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition.
- **Activity Recognition**: Lara, O. D., & Labrador, M. A. (2013). A survey on human activity recognition using wearable sensors.
- **Sensor Logger App**: Available on Google Play Store and Apple App Store

## Contributing

This is an academic project for **ML Techniques II - Formative 2** at African Leadership University (ALU).

**Team Members:**
- Add your names here

**Instructor/Course:**
- ML Techniques II
- African Leadership University

## License

This project is created for educational purposes as part of the ALU curriculum.

## Contact

For questions or issues, please contact the project maintainers through ALU channels.

---

**Last Updated:** March 2026  
**Course:** ML Techniques II - Formative 2  
**Institution:** African Leadership University (ALU)
