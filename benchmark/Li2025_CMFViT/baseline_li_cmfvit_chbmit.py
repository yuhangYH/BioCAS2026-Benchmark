"""
Efficient CHB-MIT reproduction script for the Li et al. CMFViT baseline.

The paper proposes a CNN + ViT dual-stream model with multi-stream feature
fusion (MSFF). This implementation keeps the architectural idea faithful:

1. CNN stream: local EEG feature extraction.
2. ViT stream: long-range temporal dependency modeling.
3. MSFF: both 128-dimensional stream outputs are stacked and averaged.
4. Classifier: 128 -> 128 -> 2 with dropout.

For fair benchmarking inside the MV-AFA project, this script can run either on
normalized EEG windows or on cached wavelet time-frequency maps. The latter is
provided as a practical TQWT-style preprocessing mode following the paper's
Q=2.2, r=3, J=8 setting. PyWavelets does not expose the exact tunable-Q
wavelet transform, so this mode uses a complex wavelet scalogram as an
engineering approximation unless a dedicated TQWT implementation is added.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import warnings
from collections import OrderedDict
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


warnings.filterwarnings("ignore")


MV_AFA_18_BIPOLAR_CHANNELS = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FZ-CZ",
    "CZ-PZ",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj: dict, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def normalize_channel_name(name: str) -> str:
    name = str(name).strip().upper()
    if name.startswith("EEG "):
        name = name[4:]
    if name.startswith("POL "):
        name = name[4:]
    for suffix in ("-REF", "-LE"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    name = name.replace(" ", "")
    name = re.sub(r"[-_]\d+$", "", name)
    name = re.sub(r"[^A-Z0-9\\-+]", "", name)
    return name


def parse_chbmit_summary(summary_path: str | Path) -> OrderedDict[str, list[tuple[float, float]]]:
    seizure_dict: OrderedDict[str, list[list[float | None]]] = OrderedDict()
    with Path(summary_path).open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    current_file = None
    for line in lines:
        low = line.lower()
        if low.startswith("file name:"):
            current_file = line.split(":", 1)[1].strip()
            seizure_dict[current_file] = []
        elif re.match(r"^seizure(?:\\s+\\d+)?\\s+start time:", low):
            if current_file is not None:
                start_sec = float(line.split(":", 1)[1].strip().split()[0])
                seizure_dict[current_file].append([start_sec, None])
        elif re.match(r"^seizure(?:\\s+\\d+)?\\s+end time:", low):
            if current_file is not None and seizure_dict[current_file]:
                end_sec = float(line.split(":", 1)[1].strip().split()[0])
                if seizure_dict[current_file][-1][1] is None:
                    seizure_dict[current_file][-1][1] = end_sec

    cleaned: OrderedDict[str, list[tuple[float, float]]] = OrderedDict()
    for edf_file, intervals in seizure_dict.items():
        cleaned[edf_file] = [
            (float(start), float(end))
            for start, end in intervals
            if start is not None and end is not None
        ]
    return cleaned


def resolve_subjects(data_dir: str | Path, subject_id: str) -> list[str]:
    data_dir = Path(data_dir)
    if subject_id.lower() == "all":
        subjects = sorted(
            p.name for p in data_dir.iterdir() if p.is_dir() and p.name.lower().startswith("chb")
        )
        if not subjects:
            raise FileNotFoundError(f"No CHB-MIT subject folders found in {data_dir}")
        return subjects

    subject_path = data_dir / subject_id
    if not subject_path.is_dir():
        raise FileNotFoundError(f"Subject folder not found: {subject_path}")
    return [subject_id]


def get_ordered_picks(raw: mne.io.BaseRaw, target_channels: list[str]) -> list[int]:
    normalized_to_idx: dict[str, int] = {}
    for idx, ch_name in enumerate(raw.ch_names):
        normalized = normalize_channel_name(ch_name)
        normalized_to_idx.setdefault(normalized, idx)

    selected = {}
    used = set()
    missing = []
    for ch in target_channels:
        if ch in normalized_to_idx and normalized_to_idx[ch] not in used:
            idx = normalized_to_idx[ch]
            selected[ch] = idx
            used.add(idx)
        else:
            missing.append(ch)

    for ch in list(missing):
        found_idx = None
        for raw_norm, idx in normalized_to_idx.items():
            if idx in used:
                continue
            if raw_norm.startswith(ch) or ch.startswith(raw_norm):
                found_idx = idx
                break
        if found_idx is not None:
            selected[ch] = found_idx
            used.add(found_idx)
            missing.remove(ch)

    if missing:
        raise ValueError(f"Missing required CHB-MIT channels: {missing}")
    return [selected[ch] for ch in target_channels]


def overlaps_seizure(win_start_sec: float, win_end_sec: float, intervals: list[tuple[float, float]]) -> int:
    return int(any(win_start_sec < end and win_end_sec > start for start, end in intervals))


def build_sample_index(
    data_dir: str | Path,
    subject_id: str,
    out_dir: str | Path,
    window_sec: float,
    step_sec: float,
) -> tuple[pd.DataFrame, OrderedDict[str, list[tuple[float, float]]]]:
    ensure_dir(out_dir)
    rows = []
    seizure_intervals: OrderedDict[str, list[tuple[float, float]]] = OrderedDict()
    subjects = resolve_subjects(data_dir, subject_id)

    for subject in subjects:
        subject_dir = Path(data_dir) / subject
        subject_summary = parse_chbmit_summary(subject_dir / f"{subject}-summary.txt")
        seizure_intervals.update(subject_summary)

        for edf_file, intervals in subject_summary.items():
            edf_path = subject_dir / edf_file
            if not edf_path.exists():
                print(f"[WARN] Missing EDF: {edf_path}")
                continue

            try:
                raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
                picks = get_ordered_picks(raw, MV_AFA_18_BIPOLAR_CHANNELS)
            except Exception as exc:
                print(f"[WARN] Skipping {edf_path}: {exc}")
                continue

            sfreq = float(raw.info["sfreq"])
            win = int(round(window_sec * sfreq))
            step = int(round(step_sec * sfreq))
            n_samples = int(raw.n_times)
            if n_samples < win:
                continue

            for start_sample in range(0, n_samples - win + 1, step):
                end_sample = start_sample + win
                start_sec = start_sample / sfreq
                end_sec = end_sample / sfreq
                label = overlaps_seizure(start_sec, end_sec, intervals)
                rows.append(
                    {
                        "subject_id": subject,
                        "edf_file": edf_file,
                        "edf_path": str(edf_path),
                        "sfreq": sfreq,
                        "start_sample": start_sample,
                        "end_sample": end_sample,
                        "win_start_sec": start_sec,
                        "win_end_sec": end_sec,
                        "label": label,
                        "channel_count": len(picks),
                    }
                )

    meta = pd.DataFrame(rows)
    meta.to_csv(Path(out_dir) / "meta_raw.csv", index=False)
    print(f"[INFO] Indexed {len(meta)} windows. Positives={int(meta['label'].sum()) if len(meta) else 0}")
    return meta, seizure_intervals


def downsample_train_majority(meta: pd.DataFrame, seed: int) -> pd.DataFrame:
    pos = meta[meta["label"] == 1]
    neg = meta[meta["label"] == 0]
    if len(pos) == 0 or len(neg) == 0:
        return meta.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = min(len(pos), len(neg))
    pos_sample = pos.sample(n=n, random_state=seed, replace=len(pos) < n)
    neg_sample = neg.sample(n=n, random_state=seed)
    return pd.concat([pos_sample, neg_sample]).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def window_cache_key(row: pd.Series, args: argparse.Namespace) -> str:
    raw = (
        f"{row['edf_path']}|{int(row['start_sample'])}|{int(row['end_sample'])}|"
        f"{args.input_mode}|q={args.tqwt_q}|r={args.tqwt_r}|j={args.tqwt_levels}|"
        f"bins={args.tqwt_time_bins}|wavelet={args.tqwt_wavelet}"
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def resample_time_axis(x: np.ndarray, target_len: int) -> np.ndarray:
    if x.shape[-1] == target_len:
        return x
    old = np.linspace(0.0, 1.0, x.shape[-1])
    new = np.linspace(0.0, 1.0, target_len)
    flat = x.reshape(-1, x.shape[-1])
    out = np.stack([np.interp(new, old, row) for row in flat], axis=0)
    return out.reshape(*x.shape[:-1], target_len)


def compute_tqwt_style_map(
    x: np.ndarray,
    sfreq: float,
    q_factor: float = 2.2,
    redundancy: float = 3.0,
    levels: int = 8,
    time_bins: int = 128,
    wavelet: str | None = None,
) -> np.ndarray:
    """Build a TQWT-style multi-channel time-frequency map.

    Li et al. use TQWT with Q=2.2, r=3, J=8 before the CNN/ViT model. A
    dedicated TQWT implementation is not included in the standard Python stack,
    so this function uses a complex Morlet CWT as a deterministic proxy. The
    Q/r/J parameters still control bandwidth, redundancy, and number of scales.
    """
    if wavelet is None:
        wavelet = f"cmor{max(float(redundancy), 1.0):.1f}-{max(float(q_factor), 0.5):.1f}"

    n = x.shape[-1]
    max_scale = max(2.0, min(float(n // 2), 2.0 ** (levels + 1)))
    scales = np.geomspace(2.0, max_scale, num=levels).astype(np.float32)

    maps = []
    for ch in range(x.shape[0]):
        coef, _ = pywt.cwt(x[ch], scales, wavelet, sampling_period=1.0 / sfreq)
        power = np.log1p(np.abs(coef).astype(np.float32))
        power = resample_time_axis(power, time_bins)
        maps.append(power)

    tf_map = np.concatenate(maps, axis=0).astype(np.float32)
    tf_map = (tf_map - tf_map.mean()) / (tf_map.std() + 1e-6)
    return tf_map


def split_meta(meta: pd.DataFrame, split_mode: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if split_mode == "patient_independent":
        subjects = np.array(sorted(meta["subject_id"].unique()))
        labels = np.array([int(meta.loc[meta["subject_id"] == s, "label"].max()) for s in subjects])
        train_subjects, test_subjects = train_test_split(
            subjects,
            test_size=0.30,
            random_state=seed,
            stratify=labels if len(np.unique(labels)) > 1 else None,
        )
        train_labels = np.array([int(meta.loc[meta["subject_id"] == s, "label"].max()) for s in train_subjects])
        train_subjects, val_subjects = train_test_split(
            train_subjects,
            test_size=0.15 / 0.70,
            random_state=seed,
            stratify=train_labels if len(np.unique(train_labels)) > 1 else None,
        )
        train = meta[meta["subject_id"].isin(train_subjects)].copy()
        val = meta[meta["subject_id"].isin(val_subjects)].copy()
        test = meta[meta["subject_id"].isin(test_subjects)].copy()
    else:
        idx = np.arange(len(meta))
        train_val_idx, test_idx = train_test_split(
            idx,
            test_size=0.30,
            random_state=seed,
            stratify=meta["label"].to_numpy(),
        )
        train_val = meta.iloc[train_val_idx]
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=0.15 / 0.70,
            random_state=seed,
            stratify=train_val["label"].to_numpy(),
        )
        train, val, test = meta.iloc[train_idx].copy(), meta.iloc[val_idx].copy(), meta.iloc[test_idx].copy()

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


class CHBMITWindowDataset(Dataset):
    def __init__(
        self,
        meta: pd.DataFrame,
        target_channels: list[str],
        args: argparse.Namespace,
        normalize: bool = True,
        raw_cache_files: int = 3,
    ):
        self.meta = meta.reset_index(drop=True)
        self.target_channels = target_channels
        self.args = args
        self.normalize = normalize
        self.raw_cache_files = raw_cache_files
        self.raw_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.feature_cache_dir = Path(args.feature_cache_dir) if args.feature_cache_dir else None
        if self.feature_cache_dir is not None:
            self.feature_cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.meta)

    def _get_raw_array(self, edf_path: str) -> np.ndarray:
        if edf_path in self.raw_cache:
            arr = self.raw_cache.pop(edf_path)
            self.raw_cache[edf_path] = arr
            return arr

        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        picks = get_ordered_picks(raw, self.target_channels)
        arr = raw.get_data(picks=picks).astype(np.float32)
        self.raw_cache[edf_path] = arr
        if len(self.raw_cache) > self.raw_cache_files:
            self.raw_cache.popitem(last=False)
        return arr

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.meta.iloc[idx]
        arr = self._get_raw_array(row["edf_path"])
        start, end = int(row["start_sample"]), int(row["end_sample"])
        x = arr[:, start:end].copy()
        if self.normalize:
            mean = x.mean(axis=1, keepdims=True)
            std = x.std(axis=1, keepdims=True) + 1e-6
            x = (x - mean) / std
        if self.args.input_mode == "tqwt":
            cache_path = None
            if self.feature_cache_dir is not None:
                cache_path = self.feature_cache_dir / f"{window_cache_key(row, self.args)}.npz"
            if cache_path is not None and cache_path.exists():
                with np.load(cache_path) as data:
                    x = data["x"].astype(np.float32)
            else:
                x = compute_tqwt_style_map(
                    x,
                    sfreq=float(row["sfreq"]),
                    q_factor=self.args.tqwt_q,
                    redundancy=self.args.tqwt_r,
                    levels=self.args.tqwt_levels,
                    time_bins=self.args.tqwt_time_bins,
                    wavelet=self.args.tqwt_wavelet,
                )
                if cache_path is not None:
                    np.savez_compressed(cache_path, x=x.astype(np.float32))
        y = int(row["label"])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


class CNNLocalStream(nn.Module):
    def __init__(self, out_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(64, 128, kernel_size=3, stride=(1, 2), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(128, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x.unsqueeze(1)))


class ViTTemporalStream(nn.Module):
    def __init__(
        self,
        in_channels: int,
        window_samples: int,
        patch_size: int = 8,
        embed_dim: int = 128,
        depth: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
        out_dim: int = 128,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        n_patches = max(1, window_samples // patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x).transpose(1, 2)
        tokens = tokens + self.pos_embed[:, : tokens.size(1), :]
        tokens = self.encoder(tokens)
        pooled = self.norm(tokens.mean(dim=1))
        return self.fc(pooled)


class CMFViT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        window_samples: int,
        patch_size: int = 8,
        embed_dim: int = 128,
        depth: int = 2,
        heads: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.cnn = CNNLocalStream(out_dim=128, dropout=0.2)
        self.vit = ViTTemporalStream(
            in_channels=in_channels,
            window_samples=window_samples,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            heads=heads,
            dropout=0.2,
            out_dim=128,
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cnn_feat = self.cnn(x)
        vit_feat = self.vit(x)
        fused = torch.stack([cnn_feat, vit_feat], dim=1).mean(dim=1)
        return self.classifier(fused)


def window_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * sens / max(precision + sens, 1e-12)
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = float("nan")
    try:
        p, r, _ = precision_recall_curve(y_true, y_prob)
        auprc = auc(r, p)
    except ValueError:
        auprc = float("nan")
    return {
        "accuracy": float(acc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "f1": float(f1),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probs, preds, labels = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        prob = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        probs.append(prob)
        labels.append(y.numpy())
    y_true = np.concatenate(labels)
    y_prob = np.concatenate(probs)
    y_pred = (y_prob >= 0.5).astype(int)
    return y_true, y_pred, y_prob


def choose_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    thresholds = np.linspace(0.05, 0.95, 181)
    best_thr, best_score = 0.5, -1.0
    for thr in thresholds:
        pred = (y_prob >= thr).astype(int)
        cm = confusion_matrix(y_true, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        score = 0.5 * (sens + spec)
        if score > best_score:
            best_thr, best_score = float(thr), float(score)
    return best_thr


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item()) * len(y)
    return total_loss / max(len(loader.dataset), 1)


def precompute_feature_cache(dataset: CHBMITWindowDataset, name: str) -> None:
    if dataset.args.input_mode != "tqwt" or dataset.feature_cache_dir is None:
        return
    print(f"[INFO] Precomputing TQWT-style cache for {name}: {len(dataset)} windows")
    for idx in range(len(dataset)):
        _ = dataset[idx]
        if (idx + 1) % 500 == 0 or idx + 1 == len(dataset):
            print(f"[INFO] {name}: cached {idx + 1}/{len(dataset)}")


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    ensure_dir(args.output_dir)
    meta, _ = build_sample_index(args.data_dir, args.subject_id, args.output_dir, args.window_sec, args.step_sec)
    if meta.empty:
        raise RuntimeError("No windows were indexed. Check CHB-MIT path and channel mapping.")

    train_meta, val_meta, test_meta = split_meta(meta, args.split_mode, args.seed)
    train_meta_balanced = downsample_train_majority(train_meta, args.seed)
    train_meta.to_csv(Path(args.output_dir) / "meta_train_raw.csv", index=False)
    train_meta_balanced.to_csv(Path(args.output_dir) / "meta_train_balanced.csv", index=False)
    val_meta.to_csv(Path(args.output_dir) / "meta_val.csv", index=False)
    test_meta.to_csv(Path(args.output_dir) / "meta_test.csv", index=False)

    print(
        "[INFO] Split sizes:",
        f"train_raw={len(train_meta)}",
        f"train_balanced={len(train_meta_balanced)}",
        f"val={len(val_meta)}",
        f"test={len(test_meta)}",
    )

    if args.input_mode == "tqwt" and args.feature_cache_dir is None:
        args.feature_cache_dir = str(Path(args.output_dir) / "tqwt_feature_cache")

    train_ds = CHBMITWindowDataset(
        train_meta_balanced,
        MV_AFA_18_BIPOLAR_CHANNELS,
        args=args,
        raw_cache_files=args.raw_cache_files,
    )
    val_ds = CHBMITWindowDataset(
        val_meta,
        MV_AFA_18_BIPOLAR_CHANNELS,
        args=args,
        raw_cache_files=args.raw_cache_files,
    )
    test_ds = CHBMITWindowDataset(
        test_meta,
        MV_AFA_18_BIPOLAR_CHANNELS,
        args=args,
        raw_cache_files=args.raw_cache_files,
    )

    if args.precompute_tqwt:
        precompute_feature_cache(train_ds, "train")
        precompute_feature_cache(val_ds, "val")
        precompute_feature_cache(test_ds, "test")

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **loader_kwargs)

    first_x, _ = train_ds[0]
    in_channels, window_samples = first_x.shape
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = CMFViT(
        in_channels=in_channels,
        window_samples=window_samples,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    class_counts = train_meta_balanced["label"].value_counts().reindex([0, 1]).fillna(1).to_numpy(dtype=np.float32)
    class_weights = class_counts.sum() / (2.0 * np.maximum(class_counts, 1.0))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))

    best_val_auc = -1.0
    best_path = Path(args.output_dir) / "best_model.pt"
    patience_count = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, args.amp)
        y_val, pred_val, prob_val = predict(model, val_loader, device)
        val_metrics = window_metrics(y_val, pred_val, prob_val)
        val_auc = val_metrics["auroc"] if np.isfinite(val_metrics["auroc"]) else val_metrics["f1"]
        history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}})
        print(
            f"[EPOCH {epoch:03d}] loss={train_loss:.4f} "
            f"val_f1={val_metrics['f1']:.4f} val_auc={val_metrics['auroc']:.4f}"
        )
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_count = 0
            torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"[INFO] Early stopping at epoch {epoch}.")
                break

    pd.DataFrame(history).to_csv(Path(args.output_dir) / "train_history.csv", index=False)
    state = torch.load(best_path, map_location=device)
    model.load_state_dict(state["model"])

    y_val, _, prob_val = predict(model, val_loader, device)
    threshold = choose_threshold(y_val, prob_val)
    y_test, _, prob_test = predict(model, test_loader, device)
    pred_test = (prob_test >= threshold).astype(int)
    metrics = window_metrics(y_test, pred_test, prob_test)
    metrics["threshold"] = threshold

    save_json({"window_metrics_test": metrics, "args": vars(args)}, Path(args.output_dir) / "final_metrics.json")
    pred_df = test_meta.copy()
    pred_df["y_true"] = y_test
    pred_df["y_prob"] = prob_test
    pred_df["y_pred"] = pred_test
    pred_df.to_csv(Path(args.output_dir) / "all_window_predictions.csv", index=False)

    print("[RESULT]", json.dumps(metrics, indent=2))

    try:
        fpr, tpr, _ = roc_curve(y_test, prob_test)
        pr, rc, _ = precision_recall_curve(y_test, prob_test)
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Test ROC")
        plt.tight_layout()
        plt.savefig(Path(args.output_dir) / "test_roc.png", dpi=200)
        plt.close()

        plt.figure()
        plt.plot(rc, pr)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Test PR")
        plt.tight_layout()
        plt.savefig(Path(args.output_dir) / "test_pr.png", dpi=200)
        plt.close()
    except Exception as exc:
        print(f"[WARN] Could not save plots: {exc}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Efficient Li 2025 CMFViT baseline reproduction on CHB-MIT.")
    parser.add_argument("--data_dir", type=str, default="./dataset/CHB-MIT-scalp-eeg-database-1.0.0")
    parser.add_argument("--output_dir", type=str, default="./outputs/baselines/li_cmfvit_chbmit")
    parser.add_argument("--subject_id", type=str, default="all", help="CHB-MIT subject id, e.g., chb01, or all.")
    parser.add_argument("--split_mode", choices=["patient_independent", "window"], default="patient_independent")
    parser.add_argument("--window_sec", type=float, default=2.0)
    parser.add_argument("--step_sec", type=float, default=4.0)
    parser.add_argument(
        "--input_mode",
        choices=["raw", "tqwt"],
        default="raw",
        help="raw uses normalized EEG windows; tqwt uses cached wavelet time-frequency maps.",
    )
    parser.add_argument("--tqwt_q", type=float, default=2.2, help="TQWT paper setting Q=2.2.")
    parser.add_argument("--tqwt_r", type=float, default=3.0, help="TQWT paper setting redundancy r=3.")
    parser.add_argument("--tqwt_levels", type=int, default=8, help="TQWT paper setting decomposition level J=8.")
    parser.add_argument("--tqwt_time_bins", type=int, default=128, help="Time bins for cached wavelet maps.")
    parser.add_argument("--tqwt_wavelet", type=str, default=None, help="Optional PyWavelets CWT wavelet name.")
    parser.add_argument("--feature_cache_dir", type=str, default=None, help="Cache directory for TQWT-style maps.")
    parser.add_argument("--precompute_tqwt", action="store_true", help="Precompute TQWT-style maps before training.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--raw_cache_files", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
