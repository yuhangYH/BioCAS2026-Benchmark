# Xu et al. 2026 — GAT + Transformer Spatiotemporal Fusion

**Paper:** "Spatiotemporal feature interaction and fusion for EEG seizure detection"  
**Venue:** *Frontiers in Neurology*, 2026  
**Script:** [`baseline_xu_gat_transformer_chbmit.py`](baseline_xu_gat_transformer_chbmit.py)

---

## Method Overview

EEG channels are modelled as **graph nodes**. At each time window, Pearson correlations
between channel signals build an adjacency matrix, which is thresholded at |r| > 0.5 to
yield a sparse connectivity graph. A **two-layer, 8-head Graph Attention Network (GAT)**
aggregates spatial information along the resulting edges. The per-node GAT outputs are
flattened and fed to a **4-layer Transformer Encoder** with a CLS token for sequence-level
classification.

```
EEG window (16 channels × T samples)
        │
        ▼
Pearson correlation matrix  ──threshold 0.5──▶  adjacency matrix
        │
        ▼
  GAT Layer 1  (8 heads, hidden_dim=64)
        │
  GAT Layer 2  (8 heads, hidden_dim=64)
        │
        ▼
  Transformer Encoder (L=4, d_model=128, 8 heads, FFN dim=256)
        │
        ▼
  CLS token  →  Linear classifier  →  seizure / non-seizure
```

---

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Window length | 1 s |
| Step size | 0.5 s (50 % overlap) |
| Channels | 16 bipolar CHB-MIT channels |
| Pearson threshold | 0.5 |
| GAT layers | 2 |
| GAT heads | 8 |
| GAT hidden dim | 64 |
| Transformer layers | 4 |
| d_model | 128 |
| FFN dim | 256 |
| Transformer heads | 8 |
| Loss | Focal loss (γ=2, α=0.25) |
| Optimizer | Adam, lr=1e-4 |
| Epochs | 50 |
| Batch size | 32 |
| Evaluation | 10-fold CV per subject |

---

## Running

```bash
# Single subject (chb01)
python baseline_xu_gat_transformer_chbmit.py \
    --data_dir /path/to/CHB-MIT-scalp-eeg-database-1.0.0 \
    --subject chb01 \
    --output_dir ./outputs/xu2026

# All subjects sequentially
python baseline_xu_gat_transformer_chbmit.py \
    --data_dir /path/to/CHB-MIT-scalp-eeg-database-1.0.0 \
    --subject all \
    --output_dir ./outputs/xu2026
```

---

## Output Files

| File | Description |
|------|-------------|
| `results.json` | Per-fold and mean metrics (accuracy, sensitivity, specificity, AUC) |
| `train_history.csv` | Loss and accuracy per epoch |

---

## Dependencies

```bash
pip install torch mne numpy scipy scikit-learn pandas
```

> No `torch_geometric` required — GAT is implemented manually without PyG dependency.

---

## Notes

- The GAT implementation follows the original attention equation (Veličković et al., 2018):
  `e_ij = LeakyReLU(a^T [Wh_i || Wh_j])`, normalised by softmax over neighbours.
- No PyG dependency: the adjacency matrix is passed as a dense tensor for portability.
- Focal Loss with γ=2 and α=0.25 mitigates the severe class imbalance in CHB-MIT
  (typically ~1–5 % ictal windows).
