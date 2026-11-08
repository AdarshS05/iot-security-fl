# Siamese SVM Mean Probability Difference (MPD) & Bootstrap Uncertainty

This folder contains the complete code, baseline dataset outputs, statistical summaries, and visualization plots for calculating **Mean Probability Difference (MPD)** bootstrap ensemble uncertainty on the **Siamese LinearSVM** model using TF-IDF feature representations.

---

## 📌 Overview

The Siamese SVM model evaluates pairs of IoT assembly TF-IDF feature vectors $(x_1, x_2)$ using absolute difference vectors $|x_1 - x_2|$.

To quantify pair classification uncertainty, a 25-replica bootstrap ensemble of Platt-calibrated `LinearSVC` models is trained and evaluated per Algorithm 2 of *Using Uncertainty as a Defense Against Adversarial Attacks for Tabular Datasets* (Santhosh, Gressel, Darling, 2022).

For each sample pair, the probability prediction $q_b(x_1, x_2)$ across the ensemble is used to calculate:
$$U_0 = \frac{1}{B} \sum_{b=1}^{B} (q_b - 0)^2$$
$$U_1 = \frac{1}{B} \sum_{b=1}^{B} (q_b - 1)^2$$
$$\text{MPD} = \min(U_0, U_1)$$

Low MPD scores signify strong similarity/dissimilarity consensus, whereas higher MPD scores signal ambiguous pair boundaries.

---

## 📦 Prerequisites & Artifacts to Download / Ensure Exist

Before running the scripts in this directory, ensure the following model and data files exist in the repository root directory:

1. **Baseline Siamese SVM Package**:
   - `models/siamese_svm1.pkl` — Trained Siamese SVM model package containing `{"model": LinearSVC, "scaler": StandardScaler}`.
2. **Bootstrapped Ensemble Checkpoints**:
   - `results/training/siamese_svm_bootstrap/ensemble_svm.pkl` — Saved ensemble containing the 25 calibrated LinearSVC models.
   - `results/training/siamese_svm_bootstrap/ensemble_svm_partial/` — Folder containing individual bootstrap model checkpoints (`model_00.pkl` through `model_24.pkl`).
3. **TF-IDF Dataset Files**:
   - `data/tfidf/tfidf_matrix.npz` — Sparse TF-IDF feature matrix (shape N x 18223).
   - `data/tfidf/labels.pkl` — Binary label array for malware vs. benign samples.
   - `data/tfidf/filenames.pkl` — Filename index for assembly samples.
   - `data/tfidf/feature_names.json` — Vocabulary list of TF-IDF n-gram feature names.

---

## 🛠️ Environment Requirements & Dependencies

To execute the scripts in this folder, install the following Python dependencies:

```bash
pip install scikit-learn>=1.0.0 scipy>=1.7.0 pandas>=1.3.0 numpy>=1.21.0 matplotlib>=3.4.0 joblib>=1.0.0
```

---

## 🚀 How to Run Code Files

### 1. Train Siamese LinearSVM Model & Generate Training Pairs (Optional)
If you wish to re-train the baseline Siamese SVM model:
```bash
python train_siamese_svm.py
```

### 2. Test Siamese Deterministic Pair Predictions
To verify Siamese pairwise inference across benign and malware sample pairs:
```bash
python test_siamese_pairs.py
```

### 3. Compute Bootstrapped Siamese SVM MPD Scores
To train/evaluate the 25 Platt-calibrated LinearSVC bootstrap ensemble replicas and calculate MPD uncertainty on 3,000 fixed sample pairs:
```bash
python bootstrap_mpd_svm.py --n_bootstrap 25 --max_pairs 3000
```

---

## 📁 Directory Structure & Generated Files

```
uncertainity/SIAMES mpd/
├── bootstrap_mpd_svm.py          # Main bootstrap MPD scaffolding script for Siamese SVM
├── train_siamese_svm.py          # Baseline Siamese SVM training script
├── test_siamese_pairs.py         # Deterministic pair prediction tester
├── generate_pairs.py             # Utility for creating positive/negative pairs
├── load_tfidf_dataset.py         # TF-IDF matrix dataset loader
├── pair_generator.py             # Pair generation helper functions
├── README.md                      # Documentation & execution guide
├── OUTPUT/                        # Tabular outputs & statistical summary files
│   ├── clean_mpd_svm_summary.csv # Evaluated MPD score metrics across test pairs
│   └── clean_mpd_svm_summary.json# Detailed JSON metrics (same-class vs diff-class MPD distributions)
└── OUTPUT_PHOTO/                  # Generated plots and visualization figures
    ├── mpd_svm_clean_baseline.png # Log-scale MPD distribution plot split by pair type
    ├── mpd_svm_calibration_check.png # Platt calibration probability verification plot
    ├── mpd_svm_ecdf_thresholds.png # Empirical CDF & threshold curve plot
    └── siamese_verdicts.png      # Pair verdict breakdown & confusion matrix
```
