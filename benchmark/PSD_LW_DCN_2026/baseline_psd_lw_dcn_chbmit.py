#!/usr/bin/env python3
"""
PSD-LW-DCN baseline for CHB-MIT seizure detection.

This script implements a practical reproduction of the PSD-LW-DCN design:
4-s EEG windows, multitaper PSD features over five EEG bands, channel-averaged
band power spectra, and a lightweight 1D convolutional classifier.

The original paper reports LOOCV results. This script supports both a
patient-independent split and leave-one-subject-out evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import warnings
from collections import OrderedDict
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal.windows import dpss
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("ERROR")


TARGET_CHANNELS = [
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

BANDS = OrderedDict(
    [
        ("delta", (0.5, 4.5)),
        ("theta", (4.0, 8.0)),
        ("alpha", (8.0, 13.0)),
        ("beta", (13.0, 30.0)),
        ("gamma", (30.0, 50.0)),
    ]
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_channel_name(name: str) -> str:
    name = name.upper().replace("EEG ", "").replace("-REF", "").replace("-LE", "")
    name = name.replace(" ", "")
    return name


def infer_subject_id(path: Path) -> str:
    match = re.search(r"(chb\d+)", str(path).lower())
    return match.group(1) if match else path.parent.name.lower()


def parse_summary_file(summary_path: Path) -> dict[str, list[tuple[float, float]]]:
    events: dict[str, list[tuple[float, float]]] = {}
    if not summary_path.exists():
        return events

    current_file = None
    pending_start = None
    with summary_path.open("r", errors="ignore") as f:
        for line in f:
            if line.startswith("File Name:"):
                current_file = line.split(":", 1)[1].strip()
                events.setdefault(current_file, [])
                pending_start = None
            elif "Seizure Start Time" in line and current_file:
                match = re.search(r"(\d+)\s*seconds", line)
                if match:
                    pending_start = float(match.group(1))
            elif "Seizure End Time" in line and current_file and pending_start is not None:
                match = re.search(r"(\d+)\s*seconds", line)
                if match:
                    events[current_file].append((pending_start, float(match.group(1))))
                    pending_start = None
    return events


def find_edf_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.rglob("*.edf"))


def build_record_index(data_dir: Path, limit_files: int | None = None) -> pd.DataFrame:
    rows = []
    summary_cache: dict[Path, dict[str, list[tuple[float, float]]]] = {}
    for edf_path in find_edf_files(data_dir):
        subject = infer_subject_id(edf_path)
        summary_path = edf_path.parent / f"{subject}-summary.txt"
        if summary_path not in summary_cache:
            summary_cache[summary_path] = parse_summary_file(summary_path)
        rows.append(
            {
                "subject": subject,
                "edf_path": str(edf_path),
                "file_name": edf_path.name,
                "events": summary_cache[summary_path].get(edf_path.name, []),
            }
        )
        if limit_files and len(rows) >= limit_files:
            break
    if not rows:
        raise FileNotFoundError(f"No EDF files found under {data_dir}")
    return pd.DataFrame(rows)


def load_aligned_eeg(edf_path: str, l_freq: float, h_freq: float) -> tuple[np.ndarray, float]:
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.rename_channels({ch: clean_channel_name(ch) for ch in raw.ch_names})

    missing = [ch for ch in TARGET_CHANNELS if ch not in raw.ch_names]
    if missing:
        raise ValueError(f"Missing target channels in {edf_path}: {missing[:5]}")

    raw.pick_channels(TARGET_CHANNELS, ordered=True)
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin", verbose=False)
    data = raw.get_data().astype(np.float32)
    sfreq = float(raw.info["sfreq"])
    return data, sfreq


def label_window(start_s: float, end_s: float, events: list[tuple[float, float]]) -> int:
    for ev_start, ev_end in events:
        if start_s < ev_end and end_s > ev_start:
            return 1
    return 0


def make_window_index(
    records: pd.DataFrame,
    window_sec: float,
    step_sec: float,
    limit_windows_per_file: int | None = None,
) -> pd.DataFrame:
    rows = []
    for _, rec in records.iterrows():
        try:
            raw = mne.io.read_raw_edf(rec.edf_path, preload=False, verbose=False)
            duration = raw.n_times / float(raw.info["sfreq"])
        except Exception as exc:
            print(f"[skip] cannot inspect {rec.edf_path}: {exc}")
            continue

        starts = np.arange(0, max(0, duration - window_sec + 1e-6), step_sec)
        if limit_windows_per_file:
            starts = starts[:limit_windows_per_file]
        for start in starts:
            end = start + window_sec
            rows.append(
                {
                    "subject": rec.subject,
                    "edf_path": rec.edf_path,
                    "start_s": float(start),
                    "end_s": float(end),
                    "label": label_window(float(start), float(end), rec.events),
                }
            )
    if not rows:
        raise RuntimeError("No valid EEG windows were created.")
    return pd.DataFrame(rows)


def split_windows(
    windows: pd.DataFrame,
    split_mode: str,
    seed: int,
    test_subject: str | None,
    val_fraction: float,
    test_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if split_mode == "loocv":
        if not test_subject:
            raise ValueError("--test_subject is required when --split_mode loocv")
        test_mask = windows.subject == test_subject
        if not test_mask.any():
            raise ValueError(f"Test subject {test_subject} not found in window index.")
        test_df = windows[test_mask].reset_index(drop=True)
        train_val_df = windows[~test_mask].reset_index(drop=True)
        subjects = np.array(sorted(train_val_df.subject.unique()))
        if len(subjects) <= 1:
            raise ValueError("LOOCV requires at least two non-test subjects for validation.")
        train_subjects, val_subjects = train_test_split(
            subjects, test_size=max(1, int(round(len(subjects) * val_fraction))), random_state=seed
        )
        train_df = train_val_df[train_val_df.subject.isin(train_subjects)].reset_index(drop=True)
        val_df = train_val_df[train_val_df.subject.isin(val_subjects)].reset_index(drop=True)
        return train_df, val_df, test_df

    if split_mode == "patient_independent":
        subjects = np.array(sorted(windows.subject.unique()))
        train_subjects, test_subjects = train_test_split(
            subjects, test_size=test_fraction, random_state=seed
        )
        train_subjects, val_subjects = train_test_split(
            train_subjects, test_size=val_fraction / (1.0 - test_fraction), random_state=seed
        )
        train_df = windows[windows.subject.isin(train_subjects)].reset_index(drop=True)
        val_df = windows[windows.subject.isin(val_subjects)].reset_index(drop=True)
        test_df = windows[windows.subject.isin(test_subjects)].reset_index(drop=True)
        return train_df, val_df, test_df

    train_df, temp_df = train_test_split(
        windows,
        test_size=val_fraction + test_fraction,
        random_state=seed,
        stratify=windows.label,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_fraction / (val_fraction + test_fraction),
        random_state=seed,
        stratify=temp_df.label,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def downsample_majority(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    pos = df[df.label == 1]
    neg = df[df.label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if len(neg) > len(pos):
        neg = neg.sample(n=len(pos), random_state=seed)
    elif len(pos) > len(neg):
        pos = pos.sample(n=len(neg), random_state=seed)
    return pd.concat([pos, neg], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def resample_vector(values: np.ndarray, out_len: int) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(out_len, dtype=np.float32)
    if len(values) == 1:
        return np.full(out_len, float(values[0]), dtype=np.float32)
    old_x = np.linspace(0.0, 1.0, len(values))
    new_x = np.linspace(0.0, 1.0, out_len)
    return np.interp(new_x, old_x, values).astype(np.float32)


def multitaper_band_psd_feature(
    eeg: np.ndarray,
    sfreq: float,
    bins_per_band: int,
    mt_nw: float,
    mt_kmax: int,
) -> np.ndarray:
    """Return log multitaper PSD feature averaged over channels."""
    eeg = eeg.astype(np.float32)
    eeg = eeg - eeg.mean(axis=1, keepdims=True)
    eeg = eeg / (eeg.std(axis=1, keepdims=True) + 1e-6)

    n_times = eeg.shape[1]
    tapers = dpss(n_times, NW=mt_nw, Kmax=mt_kmax, sym=False).astype(np.float32)
    tapered = eeg[:, None, :] * tapers[None, :, :]
    spec = np.fft.rfft(tapered, axis=-1)
    psd = (np.abs(spec) ** 2).mean(axis=1)
    psd = psd.mean(axis=0)
    freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)

    band_features = []
    for low, high in BANDS.values():
        mask = (freqs >= low) & (freqs <= high)
        band = np.log1p(psd[mask])
        band_features.append(resample_vector(band, bins_per_band))
    return np.concatenate(band_features).astype(np.float32)


def cache_key(row: pd.Series, args: argparse.Namespace) -> str:
    payload = (
        f"{row.edf_path}|{row.start_s:.3f}|{row.end_s:.3f}|{args.window_sec}|"
        f"{args.bins_per_band}|{args.mt_nw}|{args.mt_kmax}|{args.l_freq}|{args.h_freq}"
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


class PSDWindowDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        args: argparse.Namespace,
        cache_dir: Path | None = None,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.args = args
        self.cache_dir = cache_dir
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self._edf_cache: dict[str, tuple[np.ndarray, float]] = {}
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.frame)

    def _load_window(self, row: pd.Series) -> tuple[np.ndarray, float]:
        edf_path = row.edf_path
        if edf_path not in self._edf_cache:
            self._edf_cache[edf_path] = load_aligned_eeg(edf_path, self.args.l_freq, self.args.h_freq)
        data, sfreq = self._edf_cache[edf_path]
        start = int(round(row.start_s * sfreq))
        stop = start + int(round(self.args.window_sec * sfreq))
        window = data[:, start:stop]
        expected = int(round(self.args.window_sec * sfreq))
        if window.shape[1] < expected:
            padded = np.zeros((window.shape[0], expected), dtype=np.float32)
            padded[:, : window.shape[1]] = window
            window = padded
        return window, sfreq

    def _compute_feature(self, row: pd.Series) -> np.ndarray:
        if self.cache_dir:
            path = self.cache_dir / f"{cache_key(row, self.args)}.npy"
            if path.exists():
                return np.load(path).astype(np.float32)
        window, sfreq = self._load_window(row)
        feature = multitaper_band_psd_feature(
            window,
            sfreq,
            bins_per_band=self.args.bins_per_band,
            mt_nw=self.args.mt_nw,
            mt_kmax=self.args.mt_kmax,
        )
        if self.cache_dir:
            np.save(path, feature)
        return feature

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.frame.iloc[index]
        feature = self._compute_feature(row)
        if self.feature_mean is not None and self.feature_std is not None:
            feature = (feature - self.feature_mean) / (self.feature_std + 1e-6)
        label = int(row.label)
        return torch.from_numpy(feature.astype(np.float32)), torch.tensor(label, dtype=torch.long)


def fit_feature_scaler(dataset: PSDWindowDataset, max_samples: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    n = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    features = []
    for idx in range(n):
        row = dataset.frame.iloc[idx]
        features.append(dataset._compute_feature(row))
    arr = np.stack(features, axis=0)
    return arr.mean(axis=0).astype(np.float32), (arr.std(axis=0) + 1e-6).astype(np.float32)


class PSDLWDCN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        filters: int = 32,
        kernel_size: int = 3,
        fc_hidden: int = 16,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.features = nn.Sequential(
            nn.Conv1d(1, filters, kernel_size=kernel_size, stride=1, padding=pad),
            nn.BatchNorm1d(filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.Conv1d(filters, filters, kernel_size=kernel_size, stride=1, padding=pad),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.BatchNorm1d(filters),
            nn.Conv1d(filters, filters, kernel_size=kernel_size, stride=1, padding=pad),
            nn.BatchNorm1d(filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.Conv1d(filters, filters, kernel_size=kernel_size, stride=1, padding=pad),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(filters),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_dim)
            flat_dim = int(np.prod(self.features(dummy).shape[1:]))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(flat_dim, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        return self.classifier(self.features(x))


def compute_class_weights(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(labels.astype(int), minlength=2).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(probs)


def select_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    candidates = np.unique(np.quantile(y_prob, np.linspace(0.01, 0.99, 99)))
    best_thr, best_score = 0.5, -math.inf
    for thr in candidates:
        pred = (y_prob >= thr).astype(int)
        score = balanced_accuracy_score(y_true, pred)
        if score > best_score:
            best_score = score
            best_thr = float(thr)
    return best_thr


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp + 1e-12)
    sens = tp / (tp + fn + 1e-12)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": sens,
        "specificity": spec,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "threshold": threshold,
    }
    try:
        out["auroc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        out["auroc"] = float("nan")
    try:
        out["auprc"] = average_precision_score(y_true, y_prob)
    except ValueError:
        out["auprc"] = float("nan")
    return out


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    labels: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, float]]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(labels, device))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    best_state, best_loss, patience = None, math.inf, 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * len(y)
            total_n += len(y)

        model.eval()
        val_loss, val_n = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += float(loss.item()) * len(y)
                val_n += len(y)
        train_loss = total_loss / max(1, total_n)
        val_loss = val_loss / max(1, val_n)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                print(f"early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def save_roc_pr_arrays(out_dir: Path, y_true: np.ndarray, y_prob: np.ndarray) -> None:
    try:
        fpr, tpr, roc_thr = roc_curve(y_true, y_prob)
        pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thr}).to_csv(out_dir / "roc_curve.csv", index=False)
    except ValueError:
        pass
    try:
        precision, recall, pr_thr = precision_recall_curve(y_true, y_prob)
        pr_thr = np.r_[pr_thr, np.nan]
        pd.DataFrame({"precision": precision, "recall": recall, "threshold": pr_thr}).to_csv(
            out_dir / "pr_curve.csv", index=False
        )
    except ValueError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="PSD-LW-DCN CHB-MIT baseline")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--feature_cache_dir", type=Path, default=None)
    parser.add_argument("--split_mode", choices=["patient_independent", "window", "loocv"], default="patient_independent")
    parser.add_argument("--test_subject", type=str, default=None)
    parser.add_argument("--window_sec", type=float, default=4.0)
    parser.add_argument("--step_sec", type=float, default=4.0)
    parser.add_argument("--l_freq", type=float, default=0.5)
    parser.add_argument("--h_freq", type=float, default=50.0)
    parser.add_argument("--bins_per_band", type=int, default=24)
    parser.add_argument("--mt_nw", type=float, default=3.0)
    parser.add_argument("--mt_kmax", type=int, default=5)
    parser.add_argument("--filters", type=int, default=32)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--fc_hidden", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--limit_files", type=int, default=None)
    parser.add_argument("--limit_windows_per_file", type=int, default=None)
    parser.add_argument("--precompute_features", action="store_true")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.feature_cache_dir or (args.output_dir / "feature_cache")

    records = build_record_index(args.data_dir, limit_files=args.limit_files)
    windows = make_window_index(records, args.window_sec, args.step_sec, args.limit_windows_per_file)
    train_df, val_df, test_df = split_windows(
        windows,
        split_mode=args.split_mode,
        seed=args.seed,
        test_subject=args.test_subject,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )
    train_balanced = downsample_majority(train_df, args.seed)

    for name, frame in [("train_original", train_df), ("train_balanced", train_balanced), ("val", val_df), ("test", test_df)]:
        counts = frame.label.value_counts().to_dict()
        print(f"{name}: n={len(frame)}, labels={counts}")
        frame.to_csv(args.output_dir / f"{name}_windows.csv", index=False)

    train_raw_ds = PSDWindowDataset(train_balanced, args, cache_dir=cache_dir)
    val_raw_ds = PSDWindowDataset(val_df, args, cache_dir=cache_dir)
    test_raw_ds = PSDWindowDataset(test_df, args, cache_dir=cache_dir)

    if args.precompute_features:
        print("precomputing PSD features...")
        for ds_name, ds in [("train", train_raw_ds), ("val", val_raw_ds), ("test", test_raw_ds)]:
            for idx in range(len(ds)):
                _ = ds._compute_feature(ds.frame.iloc[idx])
            print(f"  {ds_name}: done")

    mean, std = fit_feature_scaler(train_raw_ds)
    np.save(args.output_dir / "feature_mean.npy", mean)
    np.save(args.output_dir / "feature_std.npy", std)

    train_ds = PSDWindowDataset(train_balanced, args, cache_dir=cache_dir, feature_mean=mean, feature_std=std)
    val_ds = PSDWindowDataset(val_df, args, cache_dir=cache_dir, feature_mean=mean, feature_std=std)
    test_ds = PSDWindowDataset(test_df, args, cache_dir=cache_dir, feature_mean=mean, feature_std=std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    input_dim = len(BANDS) * args.bins_per_band
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PSDLWDCN(
        input_dim=input_dim,
        filters=args.filters,
        kernel_size=args.kernel_size,
        fc_hidden=args.fc_hidden,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model parameters: {n_params:,}")

    model, history = train_model(
        model,
        train_loader,
        val_loader,
        labels=train_balanced.label.to_numpy(),
        args=args,
        device=device,
    )
    pd.DataFrame(history).to_csv(args.output_dir / "train_history.csv", index=False)

    val_y, val_prob = predict(model, val_loader, device)
    threshold = select_threshold(val_y, val_prob)
    test_y, test_prob = predict(model, test_loader, device)
    metrics = compute_metrics(test_y, test_prob, threshold)
    metrics["parameters"] = n_params
    metrics["input_dim"] = input_dim
    metrics["split_mode"] = args.split_mode
    metrics["test_subject"] = args.test_subject or ""

    print(json.dumps(metrics, indent=2))
    with (args.output_dir / "final_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame({"label": test_y, "probability": test_prob, "prediction": (test_prob >= threshold).astype(int)}).to_csv(
        args.output_dir / "test_predictions.csv", index=False
    )
    save_roc_pr_arrays(args.output_dir, test_y, test_prob)
    torch.save(model.state_dict(), args.output_dir / "psd_lw_dcn.pt")


if __name__ == "__main__":
    main()
