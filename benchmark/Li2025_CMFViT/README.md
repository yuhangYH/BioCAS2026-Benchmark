# Li et al. 2025 — CMFViT: CNN + Vision Transformer Multi-Stream Fusion

**Paper:** "Multi-stream feature fusion with CNN and Vision Transformer for EEG seizure detection"  
**Venue:** *Journal of Translational Medicine*, 2025  
**Script:** [`baseline_li_cmfvit_chbmit.py`](baseline_li_cmfvit_chbmit.py)

---

## Method Overview

CMFViT processes each EEG window as a **time-frequency image** through two parallel streams:
a CNN stream capturing local spatial features and a ViT stream capturing global temporal
dependencies. The two 128-dimensional embeddings are stacked and averaged (Multi-Stream
Feature Fusion, MSFF) before a fully connected classifier.

```
EEG window
     │
     ▼
TQWT-style time-frequency map (C × H × W image)
     │
     ├────────────────────┬────────────────────┐
     ▼                    ▼
CNN Local Stream      ViT Temporal Stream
(3 Conv2D layers)     (Conv1d patch embed + TransformerEncoder)
     │                    │
     ▼                    ▼
  h_cnn (128)          h_vit (128)
     │                    │
     └────────┬───────────┘
              ▼
       MSFF: stack → mean → h_fused (128)
              │
              ▼
       FC → Seizure / Non-seizure
```

---

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Window length | 4 s |
| Step size | 2 s (50 % overlap) |
| Input mode | TQWT-style time-frequency map (Morlet CWT) |
| CNN channels | 16 → 32 → 64, kernel 3×3 |
| ViT patch size | 16 |
| ViT embedding dim | 128 |
| ViT encoder layers | 4 |
| ViT heads | 8 |
| Fusion embedding dim | 128 |
| Loss | Cross-entropy (class-weighted) |
| Optimizer | AdamW, lr=1e-4, weight_decay=1e-4 |
| Epochs | 50 |
| Batch size | 32 |
| Evaluation | Patient-independent split |

---

## TQWT Approximation

The original paper uses the **Tunable Q-factor Wavelet Transform (TQWT)** with Q=2.2, r=3,
J=8 to produce a time-frequency image. TQWT is not available in standard Python libraries,
so this implementation approximates it with a **complex Morlet continuous wavelet transform
(CWT)** via `pywt.cwt()` at 32 logarithmically-spaced scales from 1–128. This is documented
transparently in the script as an engineering approximation and produces a functionally
similar time-frequency image.

---

## Running

```bash
# Patient-independent evaluation (pre-compute time-frequency maps)
python baseline_li_cmfvit_chbmit.py \
    --data_dir /path/to/CHB-MIT-scalp-eeg-database-1.0.0 \
    --subject_id all \
    --split_mode patient_independent \
    --input_mode tqwt \
    --precompute_tqwt \
    --output_dir ./outputs/li2025

# Single subject
python baseline_li_cmfvit_chbmit.py \
    --data_dir /path/to/CHB-MIT-scalp-eeg-database-1.0.0 \
    --subject_id chb01 \
    --input_mode raw \
    --output_dir ./outputs/li2025_chb01
```

---

## Output Files

| File | Description |
|------|-------------|
| `final_metrics.json` | Accuracy, sensitivity, specificity, AUC |
| `roc_curve.png` | ROC plot |
| `pr_curve.png` | Precision-recall plot |
| `model_best.pth` | Best model checkpoint |

---

## Dependencies

```bash
pip install torch torchvision mne numpy scipy pywavelets scikit-learn matplotlib
```

Optional (mixed-precision training):
```bash
# AMP is enabled by default if CUDA is available
```

---

## Notes

- `--precompute_tqwt` caches the CWT maps to disk on first run to avoid redundant
  computation. Maps are stored as `.npy` files in a `tqwt_cache/` subdirectory.
- The threshold for binary classification is selected on the validation fold to maximise
  balanced accuracy (equal weighting of sensitivity and specificity).
