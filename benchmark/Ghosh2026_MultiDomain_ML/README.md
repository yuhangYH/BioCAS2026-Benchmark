# Ghosh et al. 2026 — Multi-Domain Feature Engineering + Classical ML

**Paper:** "Multi-domain feature engineering and machine learning for epileptic seizure detection"  
**Venue:** *Discover Applied Sciences*, 2026  
**Script:** [`baseline_ghosh_chbmit.py`](baseline_ghosh_chbmit.py)

---

## Method Overview

A classical machine-learning pipeline that represents each 1-second EEG window with a
rich, handcrafted multi-domain feature vector. Feature selection is performed exclusively
on the training fold to prevent data leakage. Three classifiers are evaluated.

```
EEG window (channels × 256 samples)
        │
        ├──▶ Time domain       (7 features/channel)
        ├──▶ Frequency domain  (6 features/channel via Welch PSD)
        ├──▶ Wavelet domain    (DWT db4 L4 + WPD db4 L3)
        └──▶ Spatial (CSP)     (log-variance of 4 CSP components)
                │
                ▼
        Concatenate  →  Mutual Information top-50
                              │
                              ▼
                    Sequential Forward Selection top-15
                              │
                              ▼
              ┌───────────────┴───────────────┐
              ▼               ▼               ▼
             KNN            SVM          Random Forest
```

---

## Feature Specification

### Time-domain features (per channel)
| Feature | Description |
|---------|-------------|
| Mean | Signal mean |
| Variance | Signal variance |
| Skewness | Distribution skewness |
| Kurtosis | Distribution kurtosis |
| RMS | Root mean square amplitude |
| Line length | Sum of absolute differences |
| Zero crossing rate | Sign change rate |

### Frequency-domain features (per channel, Welch PSD)
| Feature | Description |
|---------|-------------|
| δ power | 0.5–4 Hz band power |
| θ power | 4–8 Hz band power |
| α power | 8–13 Hz band power |
| β power | 13–30 Hz band power |
| γ power | 30–45 Hz band power |
| SEF95 | 95 % spectral edge frequency |

### Wavelet features
- **DWT** (db4, level 4): energy and entropy of approximation + detail coefficients
- **WPD** (db4, level 3): energy of all 8 leaf packets

### Spatial features (CSP)
- 4 CSP components (2 extremal per class) via regularised generalised eigenproblem (λ=0.1)
- Log-variance of each filtered component (4 features total)

---

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Window length | 1 s (non-overlapping) |
| Bandpass filter | 0.5–45 Hz |
| Feature selection | MI top-50 → SFS top-15 |
| KNN neighbours | 5 |
| SVM kernel | RBF, C=10, γ=scale |
| Random Forest trees | 200, max_features=sqrt |
| Evaluation | Patient-independent (leave-one-subject-out) |

---

## Running

```bash
python baseline_ghosh_chbmit.py \
    --data_dir /path/to/CHB-MIT-scalp-eeg-database-1.0.0 \
    --subject all \
    --classifiers rf knn svm \
    --balance_train \
    --output_dir ./outputs/ghosh2026
```

---

## Output Files

| File | Description |
|------|-------------|
| `results_summary.txt` | Per-classifier per-subject metrics |
| `final_metrics.json` | Averaged metrics across subjects |

---

## Dependencies

```bash
pip install numpy scipy scikit-learn pywavelets mne pandas
```

---

## Notes

- CSP filters are **re-estimated on training data only**; test data is projected using the
  training-set filters to prevent information leakage.
- MI ranking and SFS are both computed on the training fold.
- `--balance_train` enables random undersampling to equalise ictal/interictal class counts
  before fitting the classifier, matching the paper's data balancing strategy.
