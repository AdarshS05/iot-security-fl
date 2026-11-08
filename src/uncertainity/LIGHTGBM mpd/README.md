# LightGBM Mean Probability Difference (MPD) & Bootstrap Uncertainty

This folder contains the complete code, baseline dataset outputs, statistical summaries, and visualization plots for calculating **Mean Probability Difference (MPD)** bootstrap ensemble uncertainty on the **LightGBM** tabular malware classification model.

---

## 📌 Overview

Mean Probability Difference (MPD) measures prediction uncertainty across a bootstrap ensemble of models per Algorithm 2 of *Using Uncertainty as a Defense Against Adversarial Attacks for Tabular Datasets* (Santhosh, Gressel, Darling, 2022).

For a sample $x$ and an ensemble of $B$ Platt-calibrated / probability-estimating model replicas $q_b(x) \in [0, 1]$:
$$U_0(x) = \frac{1}{B} \sum_{b=1}^{B} (q_b(x) - 0)^2$$
$$U_1(x) = \frac{1}{B} \sum_{b=1}^{B} (q_b(x) - 1)^2$$
$$\text{MPD}(x) = \min(U_0(x), U_1(x))$$

Low MPD scores indicate high model certainty (predictions consistently near 0 or near 1), while higher MPD scores indicate uncertainty near decision boundaries.

---

## 📦 Prerequisites & Artifacts to Download / Ensure Exist

Before running the scripts in this directory, ensure the following model and data files exist in the repository root directory:

1. **Baseline Model Package**:
   - `models/lightgbm_model.pkl` — Pre-trained baseline LightGBM model dictionary containing `{"model", "label_encoder", "best_threshold", "feature_names", "n_features", "test_size", "val_size"}`.
2. **Bootstrapped Ensemble Checkpoint**:
   - `results/training/lightgbm_bootstrap/ensemble.pkl` — Saved checkpoint of the 25 bootstrapped LightGBM estimators.
3. **Dataset File**:
   - `data/processed/benign_dataset.csv` — Processed tabular IoT malware dataset containing the required 11 features.

---

## 🛠️ Environment Requirements & Dependencies

To execute the scripts in this folder, install the following Python dependencies:

```bash
pip install lightgbm>=3.3.0 scikit-learn>=1.0.0 pandas>=1.3.0 numpy>=1.21.0 matplotlib>=3.4.0 joblib>=1.0.0
```

---

## 🚀 How to Run Code Files

### 1. Train LightGBM Baseline Model (Optional)
If you wish to re-train the baseline LightGBM model from scratch:
```bash
python train_lightgbm.py --csv ../../data/processed/benign_dataset.csv
```

### 2. Compute Bootstrapped MPD Scores
To evaluate the 25-bootstrap LightGBM ensemble and calculate clean baseline MPD scores:
```bash
python bootstrap_mpd.py --n_bootstrap 25 --stat paper
```

### 3. Validate Clean Baseline MPD & Export Summary Statistics
To perform data integrity checks, compute percentiles (p90, p95, p99), and generate output summary JSON/CSV files:
```bash
python validate_mpd_clean.py --csv_path OUTPUT/clean_mpd_baseline_v2.csv --ensemble_path ../../results/training/lightgbm_bootstrap/ensemble.pkl
```

### 4. Generate Publication-Quality Visualization Plots
To generate updated logarithmic-scale distribution plots and cumulative distribution curves:
```bash
python generate_improved_mpd_plot.py
```

---

## 📁 Directory Structure & Generated Files

```
uncertainity/LIGHTGBM mpd/
├── bootstrap_mpd.py              # Main bootstrap MPD scaffolding script
├── train_lightgbm.py              # Baseline LightGBM training pipeline
├── validate_mpd_clean.py          # Data integrity & statistical validation runner
├── generate_improved_mpd_plot.py  # High-resolution MPD visualization generator
├── README.md                      # Documentation & execution guide
├── OUTPUT/                        # Tabular outputs & statistical summary files
│   ├── clean_mpd_baseline.csv     # Clean MPD score output (v1)
│   ├── clean_mpd_baseline_v2.csv  # Paper-faithful clean MPD dataset (v2)
│   ├── clean_mpd_summary.csv      # Statistical summary CSV (mean, std, p90, p95, p99)
│   ├── clean_mpd_summary.json     # Machine-readable JSON summary metrics
│   ├── mpd_threshold.json         # Calibrated decision thresholds
│   └── metrics_log.csv            # Metric evaluation log
└── OUTPUT_PHOTO/                  # Generated plots and visualization figures
    ├── mpd_clean_baseline.png     # Baseline distribution plot
    ├── mpd_clean_baseline_v2.png  # Paper-faithful MPD baseline plot (v2)
    ├── mpd_clean_baseline_academic.png # Academic publication 2-panel plot
    ├── mpd_clean_baseline_improved.png # Improved log-scale MPD distribution plot
    ├── mpd_precision_recall_curve.png  # Precision-recall curve for MPD thresholding
    ├── mpd_threshold_curve.png    # Calibration and threshold curve
    ├── lightgbm_verdicts.png      # Verdict breakdown graph
    ├── confusion_matrix_mpd.png   # Confusion matrix under MPD filtering
    ├── metrics_dashboard.png      # Comprehensive metrics dashboard
    ├── latency_comparison.png     # Model latency comparison
    └── model_choice_diagram.png   # Architectural rationale flowchart
```
