"""
Complete standalone reproduction of:
Xu et al. (2026) "Epilepsy detection based on spatiotemporal feature
interaction fusion of EEG signals" – Frontiers in Neurology
DOI: 10.3389/fneur.2025.1478718

Dataset: CHB-MIT Scalp EEG Database
Model: GAT (2 layers, 8 heads) + Transformer (4 layers) with Focal Loss

Usage:
    python xu2026_gat_transformer_chbmit.py \
        --data_dir ./dataset/CHB-MIT-scalp-eeg-database-1.0.0 \
        --subject chb01 \
        --output_dir ./outputs/xu2026_chbmit

Run all subjects:
    python xu2026_gat_transformer_chbmit.py --data_dir ... --subject all
"""

import os
import re
import json
import math
import random
import argparse
import warnings
from collections import OrderedDict

import numpy as np
import mne
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Hyper-parameters (from paper Section 4.2)
# ─────────────────────────────────────────────
WINDOW_SEC   = 1.0      # 1-second sliding window
STEP_SEC     = 0.5      # 50% overlap
SFREQ_TARGET = 256      # CHB-MIT native sampling rate
N_CHANNELS   = 16       # 16 common bipolar channels
CORR_THRESH  = 0.5      # Pearson correlation threshold for graph edges

GAT_HIDDEN   = 64       # GAT hidden dim per head
GAT_HEADS    = 8        # attention heads per GAT layer
GAT_LAYERS   = 2        # number of GAT layers
GAT_DROPOUT  = 0.5

TF_DIM       = 256      # Transformer model dimension (d_model)
TF_HEADS     = 8        # Transformer attention heads
TF_LAYERS    = 4        # stacked Transformer encoder layers (L)
TF_FFN_DIM   = 512      # feed-forward hidden dim
TF_DROPOUT   = 0.1

FOCAL_GAMMA  = 2.0      # focal loss γ
FOCAL_ALPHA  = 0.25     # focal loss α

LR           = 1e-3
EPOCHS       = 100
BATCH_SIZE   = 64
N_FOLDS      = 10
SEED         = 42

# 16 bipolar channels selected in the paper (Section 4.1.1)
TARGET_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FZ-CZ",  "CZ-PZ",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "F4-C4",  "C4-P4",
]
assert len(TARGET_CHANNELS) == N_CHANNELS


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ─────────────────────────────────────────────
# CHB-MIT data loading helpers
# ─────────────────────────────────────────────
def normalize_ch(name: str) -> str:
    name = str(name).strip().upper()
    for prefix in ("EEG ", "POL "):
        if name.startswith(prefix):
            name = name[len(prefix):]
    for suffix in ("-REF", "-LE"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    name = name.replace(" ", "")
    name = re.sub(r"[-_]\d+$", "", name)          # T8-P8-0 → T8-P8
    name = re.sub(r"[^A-Z0-9\-+]", "", name)
    return name


def parse_summary(summary_path: str) -> OrderedDict:
    """Parse CHB-MIT *-summary.txt → {edf_filename: [(start_s, end_s), ...]}"""
    result = OrderedDict()
    with open(summary_path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    current = None
    for line in lines:
        low = line.lower()
        if low.startswith("file name:"):
            current = line.split(":", 1)[1].strip()
            result[current] = []
        elif re.match(r"^seizure(?:\s+\d+)?\s+start time:", low) and current:
            t = float(line.split(":", 1)[1].strip().split()[0])
            result[current].append([t, None])
        elif re.match(r"^seizure(?:\s+\d+)?\s+end time:", low) and current and result[current]:
            t = float(line.split(":", 1)[1].strip().split()[0])
            if result[current][-1][1] is None:
                result[current][-1][1] = t
    cleaned = OrderedDict()
    for fn, segs in result.items():
        cleaned[fn] = [(s, e) for s, e in segs if s is not None and e is not None]
    return cleaned


def get_picks(raw, target=TARGET_CHANNELS):
    norm2idx = {}
    for i, ch in enumerate(raw.ch_names):
        n = normalize_ch(ch)
        if n not in norm2idx:
            norm2idx[n] = i
    picks, missing = [], []
    for ch in target:
        if ch in norm2idx:
            picks.append(norm2idx[ch])
        else:
            # relaxed prefix match
            found = next(
                (idx for n, idx in norm2idx.items()
                 if n.startswith(ch) or ch.startswith(n)),
                None,
            )
            if found is not None:
                picks.append(found)
            else:
                missing.append(ch)
    if missing:
        raise ValueError(f"Missing channels: {missing}")
    return picks


def has_overlap(ws, we, intervals):
    return any(ws < e and we > s for s, e in intervals)


def bandpass_filter(data: np.ndarray, sfreq: float,
                    l_freq=0.5, h_freq=50.0) -> np.ndarray:
    """Zero-phase Butterworth bandpass (matches paper: bandpass + normalise)."""
    from scipy.signal import butter, filtfilt
    nyq = sfreq / 2.0
    b, a = butter(4, [l_freq / nyq, h_freq / nyq], btype="band")
    return filtfilt(b, a, data, axis=-1)


# ─────────────────────────────────────────────
# Build sample index (all windows for one subject)
# ─────────────────────────────────────────────
def build_windows(data_dir: str, subject_id: str):
    """
    Returns list of dicts with keys:
        edf_path, start_sample, end_sample, sfreq, label
    """
    subj_dir = os.path.join(data_dir, subject_id)
    summary  = parse_summary(os.path.join(subj_dir, f"{subject_id}-summary.txt"))

    windows = []
    for edf_file, intervals in summary.items():
        edf_path = os.path.join(subj_dir, edf_file)
        if not os.path.exists(edf_path):
            print(f"[WARN] Not found: {edf_path}")
            continue
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
            picks = get_picks(raw)
        except Exception as e:
            print(f"[WARN] Skip {edf_path}: {e}")
            continue

        sfreq = float(raw.info["sfreq"])
        n_sam = raw.n_times
        win   = int(WINDOW_SEC * sfreq)
        step  = int(STEP_SEC   * sfreq)

        for start in range(0, n_sam - win + 1, step):
            end   = start + win
            ws    = start / sfreq
            we    = end   / sfreq
            label = 1 if has_overlap(ws, we, intervals) else 0
            windows.append(dict(
                edf_path=edf_path,
                picks=picks,
                sfreq=sfreq,
                start_sample=start,
                end_sample=end,
                label=label,
            ))
    return windows


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class EEGWindowDataset(Dataset):
    """
    Loads EEG windows on demand. Returns:
        x      : (N_CHANNELS, T)  – normalised bandpass signal
        adj    : (N_CHANNELS, N_CHANNELS) – binary adjacency from Pearson corr
        label  : scalar int
    """
    def __init__(self, windows: list, train_mean=None, train_std=None):
        self.windows    = windows
        self.train_mean = train_mean   # (N_CHANNELS, 1) for normalisation
        self.train_std  = train_std

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        raw = mne.io.read_raw_edf(w["edf_path"], preload=False, verbose=False)
        data, _ = raw[w["picks"], w["start_sample"]:w["end_sample"]]  # (C, T)

        # Resample to SFREQ_TARGET if needed
        if abs(w["sfreq"] - SFREQ_TARGET) > 1:
            from scipy.signal import resample
            T_target = int(round(WINDOW_SEC * SFREQ_TARGET))
            data = resample(data, T_target, axis=-1)

        # Bandpass filter
        data = bandpass_filter(data, SFREQ_TARGET)

        # Normalise per-channel (z-score with training statistics if provided)
        if self.train_mean is not None:
            data = (data - self.train_mean) / (self.train_std + 1e-8)
        else:
            mu  = data.mean(axis=-1, keepdims=True)
            std = data.std(axis=-1, keepdims=True) + 1e-8
            data = (data - mu) / std

        # Build graph: Pearson correlation → threshold
        adj = self._build_adj(data)

        x = torch.FloatTensor(data)          # (C, T)
        a = torch.FloatTensor(adj)           # (C, C)
        y = torch.LongTensor([w["label"]])[0]
        return x, a, y

    @staticmethod
    def _build_adj(data: np.ndarray) -> np.ndarray:
        """
        Pearson correlation between channels; threshold at CORR_THRESH.
        Equation (1)-(2) in paper.
        """
        corr = np.corrcoef(data)             # (C, C)
        adj  = (np.abs(corr) >= CORR_THRESH).astype(np.float32)
        np.fill_diagonal(adj, 0)             # no self-loops (paper uses 1-hop neighbours)
        return adj


def compute_train_stats(windows: list):
    """Compute per-channel mean/std over all training windows (approximate)."""
    sums   = np.zeros((N_CHANNELS, 1))
    sq_sum = np.zeros((N_CHANNELS, 1))
    count  = 0
    for w in windows:
        raw  = mne.io.read_raw_edf(w["edf_path"], preload=False, verbose=False)
        data, _ = raw[w["picks"], w["start_sample"]:w["end_sample"]]
        sums   += data.mean(axis=-1, keepdims=True)
        sq_sum += (data ** 2).mean(axis=-1, keepdims=True)
        count  += 1
    mean = sums / count
    std  = np.sqrt(np.maximum(sq_sum / count - mean ** 2, 0))
    return mean.astype(np.float32), std.astype(np.float32)


# ─────────────────────────────────────────────
# Graph Attention Network (manual, no PyG)
# ─────────────────────────────────────────────
class GATLayer(nn.Module):
    """
    Single-layer multi-head GAT.
    Equations (2)–(6) in paper.
    Input : h  (B, N, F_in)
            adj(B, N, N)
    Output: h' (B, N, F_out * n_heads)  [concat mode]
    """
    def __init__(self, F_in: int, F_out: int, n_heads: int, dropout: float = 0.5,
                 concat: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.F_out   = F_out
        self.concat  = concat

        self.W  = nn.Linear(F_in,  F_out * n_heads, bias=False)
        # attention vector a^T ∈ R^{2F_out} per head
        self.a  = nn.Parameter(torch.empty(n_heads, 2 * F_out))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout    = nn.Dropout(dropout)
        self.act        = nn.ELU()

    def forward(self, h, adj):
        """
        h   : (B, N, F_in)
        adj : (B, N, N)  – binary (1=edge exists)
        """
        B, N, _ = h.shape

        # Linear projection → (B, N, H*F_out) → (B, N, H, F_out)
        Wh = self.W(h).view(B, N, self.n_heads, self.F_out)

        # Attention coefficients e_ij for all pairs
        # Wh_i : (B, N, 1, H, F_out), Wh_j : (B, 1, N, H, F_out)
        Wh_i = Wh.unsqueeze(2).expand(B, N, N, self.n_heads, self.F_out)
        Wh_j = Wh.unsqueeze(1).expand(B, N, N, self.n_heads, self.F_out)

        # concat [Wh_i || Wh_j] → (B, N, N, H, 2*F_out)
        concat = torch.cat([Wh_i, Wh_j], dim=-1)

        # e_ij = LeakyReLU(a^T * concat) → (B, N, N, H)
        # a: (H, 2*F_out) → (1,1,1,H,2*F_out)
        e = self.leaky_relu(
            (concat * self.a.view(1, 1, 1, self.n_heads, 2 * self.F_out)).sum(-1)
        )  # (B, N, N, H)

        # Mask non-edges (set to -inf before softmax)
        mask = (adj.unsqueeze(-1) == 0)           # (B, N, N, 1)
        # also keep self-connections for isolated nodes
        diag_mask = torch.eye(N, device=h.device).bool().unsqueeze(0).unsqueeze(-1)
        mask = mask & ~diag_mask
        e = e.masked_fill(mask, float("-inf"))

        alpha = F.softmax(e, dim=2)               # (B, N, N, H)
        # replace nan (isolated nodes all -inf) with 0
        alpha = torch.nan_to_num(alpha, nan=0.0)
        alpha = self.dropout(alpha)

        # Aggregate: h'_i = Σ_j α_ij * Wh_j  → (B, N, H, F_out)
        h_prime = (alpha.unsqueeze(-1) * Wh_j).sum(dim=2)  # (B, N, H, F_out)

        if self.concat:
            h_prime = h_prime.reshape(B, N, self.n_heads * self.F_out)
        else:
            h_prime = h_prime.mean(dim=2)          # (B, N, F_out)

        return self.act(h_prime)


class GAT(nn.Module):
    """Two-layer GAT (paper Section 3.2)."""
    def __init__(self, F_in: int, hidden: int = GAT_HIDDEN,
                 n_heads: int = GAT_HEADS, dropout: float = GAT_DROPOUT):
        super().__init__()
        self.layer1 = GATLayer(F_in,            hidden, n_heads, dropout, concat=True)
        self.layer2 = GATLayer(hidden * n_heads, hidden, n_heads, dropout, concat=False)

    def forward(self, x, adj):
        """
        x   : (B, N, F_in)
        adj : (B, N, N)
        Returns: (B, N, GAT_HIDDEN)
        """
        h = self.layer1(x, adj)
        h = self.layer2(h, adj)
        return h   # (B, N, GAT_HIDDEN)


# ─────────────────────────────────────────────
# Transformer Encoder (paper Section 3.3)
# ─────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1)])


class TransformerClassifier(nn.Module):
    """
    Linear projection of GAT output → class token → positional encoding
    → L Transformer encoder layers → Softmax.
    Paper: L=4, Equation (7)–(9).
    """
    def __init__(self, gat_dim: int = GAT_HIDDEN, d_model: int = TF_DIM,
                 n_heads: int = TF_HEADS, n_layers: int = TF_LAYERS,
                 ffn_dim: int = TF_FFN_DIM, dropout: float = TF_DROPOUT,
                 n_classes: int = 2):
        super().__init__()
        self.proj     = nn.Linear(gat_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_enc  = PositionalEncoding(d_model, max_len=N_CHANNELS + 1, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=ffn_dim, dropout=dropout,
            batch_first=True, norm_first=True,   # pre-norm = paper Figure 4
        )
        self.encoder  = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head     = nn.Linear(d_model, n_classes)

    def forward(self, gat_out):
        """gat_out: (B, N, GAT_HIDDEN)"""
        B = gat_out.size(0)
        x = self.proj(gat_out)                            # (B, N, d_model)
        cls = self.cls_token.expand(B, -1, -1)            # (B, 1, d_model)
        x   = torch.cat([cls, x], dim=1)                  # (B, N+1, d_model)
        x   = self.pos_enc(x)
        x   = self.encoder(x)                             # (B, N+1, d_model)
        logits = self.head(x[:, 0])                       # CLS token
        return logits


# ─────────────────────────────────────────────
# Full GAT + Transformer model
# ─────────────────────────────────────────────
class XuGATTransformer(nn.Module):
    def __init__(self, T: int = SFREQ_TARGET):
        super().__init__()
        self.gat         = GAT(F_in=T)
        self.transformer = TransformerClassifier(gat_dim=GAT_HIDDEN)

    def forward(self, x, adj):
        """
        x   : (B, N_CHANNELS, T)
        adj : (B, N_CHANNELS, N_CHANNELS)
        """
        h      = self.gat(x, adj)          # (B, N, GAT_HIDDEN)
        logits = self.transformer(h)       # (B, 2)
        return logits


# ─────────────────────────────────────────────
# Focal Loss (paper Section 3.4, Equation 11)
# ─────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = FOCAL_GAMMA, alpha: float = FOCAL_ALPHA):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        """logits: (B, 2), targets: (B,)"""
        probs = F.softmax(logits, dim=-1)
        p_t   = probs[range(len(targets)), targets]                   # (B,)
        alpha_t = torch.where(targets == 1,
                              torch.full_like(p_t, self.alpha),
                              torch.full_like(p_t, 1 - self.alpha))
        loss = -alpha_t * (1 - p_t) ** self.gamma * torch.log(p_t + 1e-8)
        return loss.mean()


# ─────────────────────────────────────────────
# Balance dataset 1:1 (paper Section 4.1.1)
# ─────────────────────────────────────────────
def balance_windows(windows: list, seed: int = SEED) -> list:
    pos = [w for w in windows if w["label"] == 1]
    neg = [w for w in windows if w["label"] == 0]
    rng = random.Random(seed)
    if len(pos) == 0:
        return windows
    if len(neg) > len(pos):
        neg = rng.sample(neg, len(pos))
    else:
        pos = rng.sample(pos, len(neg))
    combined = pos + neg
    rng.shuffle(combined)
    return combined


# ─────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for x, adj, y in loader:
            x, adj, y = x.to(device), adj.to(device), y.to(device)
            logits = model(x, adj)
            probs  = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            preds  = logits.argmax(dim=-1).cpu().numpy()
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    labels = np.array(all_labels)
    preds  = np.array(all_preds)
    probs  = np.array(all_probs)

    cm   = confusion_matrix(labels, preds, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    acc  = (TP + TN) / (TP + TN + FP + FN + 1e-8) * 100
    sens = TP / (TP + FN + 1e-8) * 100
    spec = TN / (TN + FP + 1e-8) * 100
    prec = TP / (TP + FP + 1e-8)
    f1   = 2 * prec * (sens / 100) / (prec + sens / 100 + 1e-8) * 100
    try:
        auc_val = roc_auc_score(labels, probs) * 100
    except Exception:
        auc_val = float("nan")

    return dict(acc=acc, sens=sens, spec=spec, f1=f1, auc=auc_val)


# ─────────────────────────────────────────────
# Training loop (one fold)
# ─────────────────────────────────────────────
def train_fold(train_wins, val_wins, device, fold_idx: int, output_dir: str):
    # Build datasets
    train_ds = EEGWindowDataset(train_wins)
    val_ds   = EEGWindowDataset(val_wins)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=False)

    model  = XuGATTransformer(T=SFREQ_TARGET).to(device)
    optim  = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA)

    best_val_acc  = 0.0
    best_ckpt_path = os.path.join(output_dir, f"fold{fold_idx}_best.pt")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x, adj, y in train_loader:
            x, adj, y = x.to(device), adj.to(device), y.to(device)
            optim.zero_grad()
            logits = model(x, adj)
            loss   = criterion(logits, y)
            loss.backward()
            optim.step()
            total_loss += loss.item()

        if epoch % 10 == 0 or epoch == EPOCHS:
            metrics = evaluate(model, val_loader, device)
            print(f"  Fold {fold_idx} Epoch {epoch:3d}/{EPOCHS} "
                  f"loss={total_loss/len(train_loader):.4f} "
                  f"acc={metrics['acc']:.2f}% sens={metrics['sens']:.2f}% "
                  f"spec={metrics['spec']:.2f}% auc={metrics['auc']:.2f}%")
            if metrics["acc"] > best_val_acc:
                best_val_acc = metrics["acc"]
                torch.save(model.state_dict(), best_ckpt_path)

    # Load best and return final val metrics
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    final_metrics = evaluate(model, val_loader, device)
    return final_metrics


# ─────────────────────────────────────────────
# Main: 10-fold cross-validation per subject
# ─────────────────────────────────────────────
def run_subject(subject_id: str, data_dir: str, output_dir: str, device):
    print(f"\n{'='*60}")
    print(f"Subject: {subject_id}")
    print(f"{'='*60}")

    subj_out = os.path.join(output_dir, subject_id)
    ensure_dir(subj_out)

    windows = build_windows(data_dir, subject_id)
    windows = balance_windows(windows)

    labels  = np.array([w["label"] for w in windows])
    n_pos   = labels.sum()
    n_neg   = len(labels) - n_pos
    print(f"  Windows: {len(windows)} (pos={n_pos}, neg={n_neg})")

    if n_pos == 0:
        print("  [SKIP] No seizure segments found.")
        return None

    skf     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(windows, labels), 1):
        train_wins = [windows[i] for i in train_idx]
        val_wins   = [windows[i] for i in val_idx]
        print(f"\n  Fold {fold_idx}/{N_FOLDS}  "
              f"train={len(train_wins)}  val={len(val_wins)}")
        metrics = train_fold(train_wins, val_wins, device, fold_idx, subj_out)
        print(f"  → Best val: acc={metrics['acc']:.2f}%  "
              f"sens={metrics['sens']:.2f}%  spec={metrics['spec']:.2f}%  "
              f"f1={metrics['f1']:.2f}%  auc={metrics['auc']:.2f}%")
        fold_results.append(metrics)

    keys = ["acc", "sens", "spec", "f1", "auc"]
    summary = {
        k: {
            "mean": float(np.mean([r[k] for r in fold_results])),
            "std":  float(np.std( [r[k] for r in fold_results])),
        }
        for k in keys
    }
    print(f"\n  Subject {subject_id} 10-fold summary:")
    for k, v in summary.items():
        print(f"    {k:4s}: {v['mean']:.2f} ± {v['std']:.2f}")

    save_json({"fold_results": fold_results, "summary": summary},
              os.path.join(subj_out, "results.json"))
    return summary


def main():
    global EPOCHS, BATCH_SIZE, LR

    parser = argparse.ArgumentParser(
        description="Xu 2026 GAT+Transformer reproduction on CHB-MIT"
    )
    parser.add_argument("--data_dir",   default="./dataset/CHB-MIT-scalp-eeg-database-1.0.0")
    parser.add_argument("--subject",    default="chb01",
                        help="Subject ID (e.g. chb01) or 'all'")
    parser.add_argument("--output_dir", default="./outputs/xu2026_gat_transformer_chbmit")
    parser.add_argument("--seed",       type=int, default=SEED)
    parser.add_argument("--epochs",     type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LR)
    parser.add_argument("--device",     default="auto",
                        help="cuda / cpu / auto")
    args = parser.parse_args()

    # Allow CLI overrides of global hyper-params
    EPOCHS     = args.epochs
    BATCH_SIZE = args.batch_size
    LR         = args.lr

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    if args.subject.lower() == "all":
        subjects = sorted(
            d for d in os.listdir(args.data_dir)
            if d.lower().startswith("chb")
            and os.path.isdir(os.path.join(args.data_dir, d))
        )
    else:
        subjects = [args.subject]

    all_summaries = {}
    for subj in subjects:
        result = run_subject(subj, args.data_dir, args.output_dir, device)
        if result:
            all_summaries[subj] = result

    if all_summaries:
        keys = ["acc", "sens", "spec", "f1", "auc"]
        print(f"\n{'='*60}")
        print("Overall results across subjects:")
        overall = {}
        for k in keys:
            vals = [s[k]["mean"] for s in all_summaries.values()]
            overall[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            print(f"  {k:4s}: {overall[k]['mean']:.2f} ± {overall[k]['std']:.2f}")
        save_json({"subjects": all_summaries, "overall": overall},
                  os.path.join(args.output_dir, "overall_results.json"))


if __name__ == "__main__":
    main()
