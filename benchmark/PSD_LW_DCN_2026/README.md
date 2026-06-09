# Gu et al. 2026 — PSD + Lightweight 1D Dilated Convolutional Network

**Paper:** "A lightweight EEG seizure detection approach using power spectral density features"  
**Venue:** *Scientific Reports*, 2026  
**Script:** [`baseline_psd_lw_dcn_chbmit.py`](baseline_psd_lw_dcn_chbmit.py)

---

## Method Overview

EEG windows are converted to compact **multitaper power spectral density (PSD)** features
across five physiological frequency bands. Channel-averaged band powers form a short 1D
feature vector (~5–10 values), which is fed into a **lightweight 1D dilated convolutional
network (DCN)** with only ~61 k parameters for fast inference.

```
EEG window (channels × samples)
        │
        ▼
Multitaper PSD (DPSS tapers, 5 bands per channel)
        │
        ▼
Channel average  →  1D feature vector (5 bands)
        │
        ▼
  ConvModule 1: Conv1d(1, 32, k=3, dilation=1) + BN + ReLU + Pool
        │
  ConvModule 2: Conv1d(32, 64, k=3, dilation=2) + BN + ReLU + Pool
        │
  ConvModule 3: Conv1d(64, 128, k=3, dilation=4) + BN + ReLU + Pool
        │
  ConvModule 4: Conv1d(128, 64, k=3, dilation=1) + BN + ReLU + Pool
        │
        ▼
  Flatten  →  FC(fc_hidden=16)  →  Dropout(0.3)  →  FC(2)
        │
        ▼
  Seizure / Non-seizure
```

---

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Window length | 4 s |
| Step size | 2 s |
| Bandpass filter | 0.5–50 Hz |
| Multitaper tapers | DPSS (scipy.signal.dpss) |
| Frequency bands | δ (0–4 Hz), θ (4–8 Hz), α (8–12 Hz), β (12–30 Hz), γ (30–50 Hz) |
| PSD normalisation | log1p |
| Conv filters | 32 → 64 → 128 → 64 |
| FC hidden | 16 |
| Dropout | 0.3 |
| Parameters | ~61 k |
| Inference latency | ~1.9 ms / window (CPU) |
| Loss | Cross-entropy (class-weighted) |
| Optimizer | Adam, lr=1e-3 |
| Epochs | 50 |
| Batch size | 64 |
| Evaluation | LOOCV / patient-independent / window split |

---

## Running

```bash
# Patient-independent split (recommended for cross-subject evaluation)
python baseline_psd_lw_dcn_chbmit.py \
    --data_dir /path/to/CHB-MIT-scalp-eeg-database-1.0.0 \
    --split_mode patient_independent \
    --precompute_features \
    --output_dir ./outputs/psd_lw_dcn

# Leave-one-subject-out (LOOCV) — test on chb01, train on remaining
python baseline_psd_lw_dcn_chbmit.py \
    --data_dir /path/to/CHB-MIT-scalp-eeg-database-1.0.0 \
    --split_mode loocv \
    --test_subject chb01 \
    --precompute_features \
    --output_dir ./outputs/psd_lw_dcn_chb01
```

---

## Output Files

| File | Description |
|------|-------------|
| `final_metrics.json` | Accuracy, sensitivity, specificity, AUC, false alarms/h |
| `roc_curve_data.csv` | FPR / TPR arrays for ROC plotting |
| `pr_curve_data.csv` | Precision / Recall arrays |
| `model_best.pth` | Best model checkpoint |

---

## False-Alarm Rate

The script computes **false alarms per hour** using the SzCORE protocol: a predicted positive
window is matched to a ground-truth seizure if it falls within ±30 s of the annotated event.
Unmatched positive windows count as false alarms.

---

## Dependencies

```bash
pip install torch mne numpy scipy scikit-learn pandas matplotlib
```

---

## Notes

- `--precompute_features` saves PSD feature arrays to disk on first run, enabling fast
  re-runs without re-processing raw EDF files. Feature cache is stored as `.npy` files
  in a `psd_cache/` subdirectory.
- The dilated convolutions with increasing dilation rates (1 → 2 → 4 → 1) expand the
  receptive field without additional parameters, which is key to the model's efficiency.
