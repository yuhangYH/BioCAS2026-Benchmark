"""
seizure_baseline_common.py
Shared utilities for all seizure-detection baselines (CHB-MIT / Siena).

Ghosh 2026 multi-domain feature baseline is fully implemented here via:
    add_ghosh_args()   – parser additions
    run_ghosh_baseline() – end-to-end pipeline
"""

import argparse
import json
import os
import random
import time
import warnings
from pathlib import Path

import numpy as np
import scipy.signal as sig
from scipy.stats import skew, kurtosis
import pywt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector, mutual_info_classif
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Argument helpers
# ---------------------------------------------------------------------------

def add_common_args(
    parser: argparse.ArgumentParser,
    dataset_name: str,
    method_name: str,
    default_data_dir: str,
    default_output_dir: str,
    default_subject: str,
):
    parser.add_argument("--data_dir", default=default_data_dir)
    parser.add_argument("--output_dir", default=default_output_dir)
    parser.add_argument("--subject", default=default_subject)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default=dataset_name)
    parser.add_argument("--method", default=method_name)


def add_ghosh_args(parser: argparse.ArgumentParser):
    parser.add_argument("--fs", type=float, default=256.0, help="Sampling frequency (Hz)")
    parser.add_argument("--window_sec", type=float, default=1.0, help="EEG window length (s)")
    parser.add_argument("--n_channels", type=int, default=23, help="Number of EEG channels")
    parser.add_argument("--n_csp", type=int, default=4, help="Number of CSP components (top+bottom)")
    parser.add_argument("--csp_lambda", type=float, default=0.1, help="CSP shrinkage regularisation λ")
    parser.add_argument("--n_mi_top", type=int, default=50, help="MI pre-filter: keep top-k features")
    parser.add_argument("--n_features_final", type=int, default=15, help="Final SFS feature count")
    parser.add_argument("--n_folds", type=int, default=5, help="Inner CV folds for SFS/HP tuning")
    parser.add_argument("--test_size", type=float, default=0.30, help="Patient-level test fraction")
    parser.add_argument("--classifiers", nargs="+", default=["knn", "svm", "rf"],
                        choices=["knn", "svm", "rf"])


def add_deep_args(parser: argparse.ArgumentParser):
    """Placeholder for deep-learning baselines."""
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--window_sec", type=float, default=4.0)
    parser.add_argument("--step_sec", type=float, default=4.0)


def run_deep_baseline(args, method: str):
    raise NotImplementedError(f"Deep baseline '{method}' not implemented in this module.")


# ===========================================================================
# DATA LOADING  (CHB-MIT)
# ===========================================================================

def _load_chbmit_subject(data_dir: str, subject: str, fs: float, window_sec: float, n_channels: int):
    """
    Load all .edf files for a given CHB-MIT subject, apply preprocessing,
    segment into non-overlapping 1-s windows, return (X, y, patient_ids).

    Returns
    -------
    X : ndarray, shape (N, C, T)
    y : ndarray, shape (N,)  0=non-seizure 1=seizure
    """
    import glob, re
    try:
        import mne
    except ImportError:
        raise ImportError("MNE-Python is required: pip install mne")

    subj_dir = Path(data_dir) / subject
    if not subj_dir.exists():
        raise FileNotFoundError(f"Subject directory not found: {subj_dir}")

    # Parse summary file for seizure annotations
    summary_file = subj_dir / f"{subject}-summary.txt"
    seizure_map = _parse_chbmit_summary(summary_file)

    window_samples = int(fs * window_sec)
    segments, labels = [], []

    edf_files = sorted(glob.glob(str(subj_dir / "*.edf")))
    for edf_path in edf_files:
        fname = Path(edf_path).name
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        except Exception:
            continue

        # Resample to target fs if needed
        if abs(raw.info["sfreq"] - fs) > 1:
            raw.resample(fs, npad="auto", verbose=False)

        # Select up to n_channels EEG channels
        picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
        if len(picks) == 0:
            picks = list(range(min(n_channels, len(raw.ch_names))))
        picks = picks[:n_channels]
        data = raw.get_data(picks=picks)  # (C, T_total)

        # Pad or truncate to n_channels
        if data.shape[0] < n_channels:
            pad = np.zeros((n_channels - data.shape[0], data.shape[1]))
            data = np.vstack([data, pad])

        data = _preprocess_eeg(data, fs)

        # Seizure intervals for this file
        seizure_intervals = seizure_map.get(fname, [])

        n_windows = data.shape[1] // window_samples
        for w in range(n_windows):
            start = w * window_samples
            end = start + window_samples
            seg = data[:, start:end]

            # Skip malformed segments
            if seg.shape != (n_channels, window_samples):
                continue
            if not np.isfinite(seg).all():
                continue

            # Label: seizure if any part overlaps a seizure interval
            t_start_s = start / fs
            t_end_s = end / fs
            label = 0
            for (sz_start, sz_end) in seizure_intervals:
                if t_end_s > sz_start and t_start_s < sz_end:
                    label = 1
                    break

            segments.append(seg)
            labels.append(label)

    if len(segments) == 0:
        raise RuntimeError(f"No valid EEG segments loaded for subject {subject}.")

    X = np.stack(segments, axis=0)   # (N, C, T)
    y = np.array(labels, dtype=np.int32)
    return X, y


def _parse_chbmit_summary(summary_path: Path):
    """Return dict: {filename: [(start_s, end_s), ...]}"""
    import re
    seizure_map = {}
    if not summary_path.exists():
        return seizure_map

    with open(summary_path, "r", errors="replace") as f:
        text = f.read()

    # Split by file block
    blocks = re.split(r"File Name:", text)[1:]
    for block in blocks:
        lines = block.strip().splitlines()
        fname = lines[0].strip()
        intervals = []
        n_seizures = 0
        pending_start = None
        for line in lines:
            m = re.search(r"Number of Seizures in File:\s*(\d+)", line)
            if m:
                n_seizures = int(m.group(1))
            ms = re.search(r"Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+)\s+seconds", line)
            me = re.search(r"Seizure(?:\s+\d+)?\s+End Time:\s*(\d+)\s+seconds", line)
            if ms:
                pending_start = int(ms.group(1))
            if me and pending_start is not None:
                intervals.append((pending_start, int(me.group(1))))
                pending_start = None
        seizure_map[fname] = intervals
    return seizure_map


def _preprocess_eeg(data: np.ndarray, fs: float) -> np.ndarray:
    """
    Bandpass 0.5-45 Hz + baseline drift correction (demean per channel).
    data: (C, T)
    """
    b, a = sig.butter(4, [0.5, 45.0], btype="bandpass", fs=fs)
    out = np.zeros_like(data)
    for c in range(data.shape[0]):
        ch = data[c] - np.mean(data[c])          # baseline drift
        ch = sig.filtfilt(b, a, ch)               # zero-phase filter
        out[c] = ch
    return out


# ===========================================================================
# FEATURE EXTRACTION
# ===========================================================================

EEG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def extract_time_features(channel: np.ndarray) -> np.ndarray:
    """7 time-domain features for one channel."""
    x = channel
    mean_val   = np.mean(x)
    var_val    = np.var(x, ddof=1)
    skew_val   = float(skew(x))
    kurt_val   = float(kurtosis(x))
    rms_val    = np.sqrt(np.mean(x ** 2))
    ll_val     = np.sum(np.abs(np.diff(x)))          # line length
    zcr_val    = np.sum(np.diff(np.sign(x)) != 0) / (len(x) - 1)  # ZCR
    return np.array([mean_val, var_val, skew_val, kurt_val, rms_val, ll_val, zcr_val])


def extract_freq_features(channel: np.ndarray, fs: float) -> np.ndarray:
    """6 frequency-domain features (5 band powers + SEF95)."""
    freqs, psd = sig.welch(channel, fs=fs, window="hann", nperseg=min(len(channel), 256))
    df = freqs[1] - freqs[0]
    feats = []
    for bname, (f1, f2) in EEG_BANDS.items():
        mask = (freqs >= f1) & (freqs <= f2)
        power = np.trapz(psd[mask], freqs[mask]) if mask.any() else 0.0
        feats.append(power)
    # SEF95: smallest f where cumulative power >= 95% of total
    total_power = np.trapz(psd, freqs) + 1e-12
    cumulative = np.cumsum(psd) * df
    sef95_idx = np.searchsorted(cumulative, 0.95 * total_power)
    sef95 = freqs[min(sef95_idx, len(freqs) - 1)]
    feats.append(sef95)
    return np.array(feats)


def extract_wavelet_features(channel: np.ndarray) -> np.ndarray:
    """
    DWT db4 level-4: log-variance, relative energy, std per subband.
    WPD db4 level-3: energy, variance, CV per node packet.
    """
    feats = []

    # --- DWT (db4, level 4) ---
    coeffs = pywt.wavedec(channel, "db4", level=4)
    total_energy = sum(np.sum(c ** 2) for c in coeffs) + 1e-12
    for c in coeffs:
        var_c = np.var(c, ddof=1) if len(c) > 1 else 0.0
        log_var = np.log(var_c + 1e-12)
        rel_energy = np.sum(c ** 2) / total_energy
        std_c = np.std(c, ddof=1) if len(c) > 1 else 0.0
        feats.extend([log_var, rel_energy, std_c])

    # --- WPD (db4, level 3) ---
    wp = pywt.WaveletPacket(data=channel, wavelet="db4", mode="symmetric", maxlevel=3)
    nodes = [node.path for node in wp.get_level(3, "natural")]
    for path in nodes:
        c = wp[path].data
        energy = np.sum(c ** 2)
        var_c  = np.var(c, ddof=1) if len(c) > 1 else 0.0
        mean_c = np.mean(np.abs(c))
        cv     = np.std(c, ddof=1) / (mean_c + 1e-12)
        feats.extend([energy, var_c, cv])

    return np.array(feats)


def compute_csp_filters(X_seizure: np.ndarray, X_nonsz: np.ndarray,
                         n_components: int, lam: float) -> np.ndarray:
    """
    Regularised CSP (shrinkage λ).
    X_seizure, X_nonsz: (N, C, T)
    Returns W: (C, n_components)  – top and bottom n_components//2 eigenvectors
    """
    def _cov(X):
        # Mean covariance across trials
        covs = []
        for trial in X:
            c = trial @ trial.T / trial.shape[1]
            covs.append(c)
        return np.mean(covs, axis=0)

    C1 = _cov(X_seizure)
    C2 = _cov(X_nonsz)

    def _shrink(C, lam):
        I = np.eye(C.shape[0])
        return (1 - lam) * C + lam * np.trace(C) / C.shape[0] * I

    C1r = _shrink(C1, lam)
    C2r = _shrink(C2, lam)
    Cc  = C1r + C2r

    # Whitening
    eigvals, eigvecs = np.linalg.eigh(Cc)
    eigvals = np.clip(eigvals, 1e-10, None)
    P = eigvecs @ np.diag(eigvals ** -0.5) @ eigvecs.T

    # Generalised eigenproblem in whitened space
    S1w = P @ C1r @ P.T
    eigvals2, eigvecs2 = np.linalg.eigh(S1w)

    # Sort descending, take top and bottom n_components//2
    idx = np.argsort(eigvals2)[::-1]
    half = n_components // 2
    selected = np.concatenate([idx[:half], idx[-half:]])
    W = (P.T @ eigvecs2)[:, selected]   # (C, n_components)
    return W


def extract_csp_features(X: np.ndarray, W: np.ndarray, fs: float) -> np.ndarray:
    """
    Project X onto CSP filters; compute log-variance + band powers (beta, gamma).
    X: (N, C, T)
    Returns: (N, n_components * 3)
    """
    N, C, T = X.shape
    n_comp = W.shape[1]
    all_feats = []
    for trial in X:
        proj = W.T @ trial   # (n_comp, T)
        row = []
        for k in range(n_comp):
            ch = proj[k]
            log_var = np.log(np.var(ch, ddof=1) + 1e-12)
            # beta power (13-30 Hz)
            f, p = sig.welch(ch, fs=fs, nperseg=min(T, 256))
            beta_p  = np.trapz(p[(f >= 13) & (f <= 30)],  f[(f >= 13) & (f <= 30)]) if np.any((f >= 13) & (f <= 30)) else 0.0
            gamma_p = np.trapz(p[(f >= 30) & (f <= 45)],  f[(f >= 30) & (f <= 45)]) if np.any((f >= 30) & (f <= 45)) else 0.0
            row.extend([log_var, beta_p, gamma_p])
        all_feats.append(row)
    return np.array(all_feats)


def extract_all_features(X: np.ndarray, W_csp: np.ndarray, fs: float,
                          selected_indices: np.ndarray = None) -> np.ndarray:
    """
    Full feature extraction for a batch X: (N, C, T).
    Returns F: (N, n_features_total)  or  (N, len(selected_indices)) if given.
    """
    N, C, T = X.shape
    all_feats = []
    for i in range(N):
        row = []
        for c in range(C):
            ch = X[i, c]
            row.extend(extract_time_features(ch))
            row.extend(extract_freq_features(ch, fs))
            row.extend(extract_wavelet_features(ch))
        all_feats.append(row)
    F_channel = np.array(all_feats)           # (N, C * per_channel_feats)

    # CSP spatial features
    F_csp = extract_csp_features(X, W_csp, fs)   # (N, n_csp*3)

    F_all = np.hstack([F_channel, F_csp])

    if selected_indices is not None:
        F_all = F_all[:, selected_indices]
    return F_all


# ===========================================================================
# FEATURE SELECTION (training only)
# ===========================================================================

def select_features_mi_sfs(X_train: np.ndarray, y_train: np.ndarray,
                             n_mi_top: int, n_final: int, n_folds: int,
                             random_state: int = 42):
    """
    Step 1: MI ranking → top n_mi_top features.
    Step 2: SFS (5-fold CV, RF scorer) → final n_final features.
    Returns selected_indices (in original feature space).
    """
    # Step 1 – MI
    mi = mutual_info_classif(X_train, y_train, random_state=random_state)
    mi_top_idx = np.argsort(mi)[::-1][:n_mi_top]
    X_mi = X_train[:, mi_top_idx]

    # Step 2 – SFS
    base_clf = RandomForestClassifier(n_estimators=50, max_depth=10,
                                      random_state=random_state, n_jobs=-1)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    sfs = SequentialFeatureSelector(
        base_clf,
        n_features_to_select=n_final,
        direction="forward",
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
    )
    sfs.fit(X_mi, y_train)
    sfs_mask = sfs.get_support()
    selected_in_mi = mi_top_idx[sfs_mask]
    return selected_in_mi


# ===========================================================================
# CLASSIFIERS
# ===========================================================================

def build_classifiers():
    return {
        "knn": KNeighborsClassifier(
            n_neighbors=7, weights="distance", metric="manhattan"
        ),
        "svm": SVC(
            kernel="rbf", C=1.0, probability=True, random_state=42
        ),
        "rf": RandomForestClassifier(
            n_estimators=100, max_depth=15,
            min_samples_split=2, min_samples_leaf=1,
            random_state=42, n_jobs=-1,
        ),
    }


# ===========================================================================
# PATIENT-INDEPENDENT SPLIT (CHB-MIT)
# ===========================================================================

def _patient_independent_split(X: np.ndarray, y: np.ndarray,
                                 patient_ids: np.ndarray, test_size: float,
                                 random_state: int = 42):
    """
    Split by patient so no patient appears in both train and test.
    """
    rng = np.random.default_rng(random_state)
    patients = np.unique(patient_ids)
    rng.shuffle(patients)
    n_test = max(1, int(len(patients) * test_size))
    test_patients  = patients[:n_test]
    train_patients = patients[n_test:]
    train_mask = np.isin(patient_ids, train_patients)
    test_mask  = np.isin(patient_ids, test_patients)
    return (X[train_mask], y[train_mask],
            X[test_mask],  y[test_mask])


def _resolve_chbmit_subjects(data_dir: str, subject: str):
    data_dir = Path(data_dir)
    if subject.lower() == "all":
        subjects = sorted(
            p.name for p in data_dir.iterdir()
            if p.is_dir() and p.name.lower().startswith("chb")
        )
        if not subjects:
            raise FileNotFoundError(f"No CHB-MIT subject folders found in {data_dir}")
        return subjects
    return [subject]


def _load_chbmit_dataset(data_dir: str, subject: str, fs: float, window_sec: float, n_channels: int):
    """Load one subject or all CHB-MIT subjects and keep subject ids for clean splitting."""
    X_parts, y_parts, pid_parts = [], [], []
    for subj in _resolve_chbmit_subjects(data_dir, subject):
        try:
            X_subj, y_subj = _load_chbmit_subject(data_dir, subj, fs, window_sec, n_channels)
        except Exception as exc:
            print(f"   [WARN] skipping {subj}: {exc}")
            continue
        X_parts.append(X_subj)
        y_parts.append(y_subj)
        pid_parts.append(np.array([subj] * len(y_subj), dtype=object))
        print(f"   {subj}: windows={len(y_subj)} seizure={int(y_subj.sum())}")

    if not X_parts:
        raise RuntimeError("No valid CHB-MIT subjects were loaded.")
    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts), np.concatenate(pid_parts)


def _downsample_training_majority(X: np.ndarray, y: np.ndarray, seed: int):
    """Balance only the training set by downsampling the majority class."""
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return X, y
    if len(neg_idx) > len(pos_idx):
        neg_idx = rng.choice(neg_idx, size=len(pos_idx), replace=False)
    elif len(pos_idx) > len(neg_idx):
        pos_idx = rng.choice(pos_idx, size=len(neg_idx), replace=False)
    keep = np.concatenate([pos_idx, neg_idx])
    rng.shuffle(keep)
    return X[keep], y[keep]


# ===========================================================================
# MAIN GHOSH BASELINE RUNNER
# ===========================================================================

def run_ghosh_baseline(args):
    """
    Full Ghosh 2026 multi-domain feature engineering pipeline.
    Works with either CHB-MIT or Siena (adjust data loading for Siena).
    """
    os.makedirs(args.output_dir, exist_ok=True)
    fs            = args.fs
    window_sec    = args.window_sec
    n_channels    = args.n_channels
    n_csp         = args.n_csp
    csp_lambda    = args.csp_lambda
    n_mi_top      = args.n_mi_top
    n_final       = args.n_features_final
    n_folds       = args.n_folds

    print(f"\n{'='*60}")
    print(f"Ghosh 2026 Multi-Domain Feature Baseline")
    print(f"Dataset : {args.dataset}  |  Subject: {args.subject}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("[1/6] Loading EEG data ...")
    t0 = time.time()

    if args.dataset.lower() in ("chbmit", "chb-mit"):
        X, y, patient_ids = _load_chbmit_dataset(args.data_dir, args.subject, fs, window_sec, n_channels)
    else:
        raise NotImplementedError(f"Dataset '{args.dataset}' loading not implemented. "
                                   "Add a loader for your dataset.")

    n_seizure = int(y.sum())
    n_total   = len(y)
    print(f"   Loaded {n_total} windows  |  seizure={n_seizure}  non-seizure={n_total-n_seizure}")
    print(f"   X shape: {X.shape}   ({time.time()-t0:.1f}s)\n")

    # ------------------------------------------------------------------
    # 2. Train / test split (patient-independent)
    # ------------------------------------------------------------------
    unique_patients = np.unique(patient_ids)
    if len(unique_patients) > 1:
        print("[2/6] Train/test split (patient-independent by subject) ...")
        X_train, y_train, X_test, y_test = _patient_independent_split(
            X, y, patient_ids, args.test_size, args.seed
        )
    else:
        print("[2/6] Train/test split (chronological within one subject) ...")
        n_train = int(n_total * (1 - args.test_size))
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]

    if getattr(args, "balance_train", False):
        before = dict(pos=int(y_train.sum()), neg=int(len(y_train) - y_train.sum()))
        X_train, y_train = _downsample_training_majority(X_train, y_train, args.seed)
        after = dict(pos=int(y_train.sum()), neg=int(len(y_train) - y_train.sum()))
        print(f"   Training downsampling: {before} -> {after}")
    print(f"   Train: {len(y_train)} windows  |  Test: {len(y_test)} windows\n")

    # ------------------------------------------------------------------
    # 3. Train CSP spatial filters (on training set only)
    # ------------------------------------------------------------------
    print("[3/6] Training CSP spatial filters ...")
    t0 = time.time()
    X_sz  = X_train[y_train == 1]
    X_nsz = X_train[y_train == 0]
    if len(X_sz) == 0 or len(X_nsz) == 0:
        print("   WARNING: one class has 0 samples; using identity CSP.")
        W_csp = np.eye(n_channels)[:, :n_csp]
    else:
        W_csp = compute_csp_filters(X_sz, X_nsz, n_csp, csp_lambda)
    print(f"   CSP filters shape: {W_csp.shape}   ({time.time()-t0:.1f}s)\n")

    # ------------------------------------------------------------------
    # 4. Feature extraction (training set)
    # ------------------------------------------------------------------
    print("[4/6] Extracting multi-domain features from training set ...")
    t0 = time.time()
    F_train_full = extract_all_features(X_train, W_csp, fs)
    print(f"   Feature matrix shape (train, all): {F_train_full.shape}   ({time.time()-t0:.1f}s)")

    # Remove near-zero-variance features
    var_mask = np.var(F_train_full, axis=0) > 1e-10
    F_train_full = F_train_full[:, var_mask]
    var_indices  = np.where(var_mask)[0]
    print(f"   After variance filter: {F_train_full.shape[1]} features\n")

    # ------------------------------------------------------------------
    # 5. Feature selection (MI → SFS, training set only)
    # ------------------------------------------------------------------
    print(f"[5/6] Feature selection: MI top-{n_mi_top} → SFS top-{n_final} ...")
    t0 = time.time()
    n_mi_top_clipped = min(n_mi_top, F_train_full.shape[1])
    selected_local = select_features_mi_sfs(
        F_train_full, y_train,
        n_mi_top=n_mi_top_clipped, n_final=n_final,
        n_folds=n_folds, random_state=args.seed,
    )
    # Map back to global feature indices (before variance filter)
    selected_global = var_indices[selected_local]
    print(f"   Selected {len(selected_local)} features   ({time.time()-t0:.1f}s)\n")

    F_train = F_train_full[:, selected_local]

    # Extract test features using the same CSP + same feature indices
    print("   Extracting features from test set ...")
    t0 = time.time()
    F_test_full = extract_all_features(X_test, W_csp, fs)
    F_test_full = F_test_full[:, var_mask]
    F_test      = F_test_full[:, selected_local]
    print(f"   Test feature matrix: {F_test.shape}   ({time.time()-t0:.1f}s)\n")

    # ------------------------------------------------------------------
    # 6. Classification + evaluation
    # ------------------------------------------------------------------
    print("[6/6] Classification & evaluation ...")
    classifiers = build_classifiers()
    results = {}

    for clf_name in args.classifiers:
        clf = classifiers[clf_name]
        print(f"\n  --- {clf_name.upper()} ---")

        # Scale for SVM / KNN
        if clf_name in ("svm", "knn"):
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(F_train)
            X_te_s = scaler.transform(F_test)
        else:
            X_tr_s, X_te_s = F_train, F_test

        # 5-fold CV on training set
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=args.seed)
        cv_scores = cross_val_score(clf, X_tr_s, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        print(f"  Train CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Fit on full training set, evaluate on test
        clf.fit(X_tr_s, y_train)
        y_pred = clf.predict(X_te_s)
        y_prob = clf.predict_proba(X_te_s)[:, 1] if hasattr(clf, "predict_proba") else y_pred

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = float("nan")

        print(f"  Test  Accuracy : {acc:.4f}")
        print(f"  Test  Precision: {prec:.4f}")
        print(f"  Test  Recall   : {rec:.4f}")
        print(f"  Test  F1-Score : {f1:.4f}")
        print(f"  Test  ROC-AUC  : {auc:.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['non-seizure','seizure'])}")
        print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

        results[clf_name] = dict(acc=acc, prec=prec, rec=rec, f1=f1, auc=auc,
                                  cv_mean=cv_scores.mean(), cv_std=cv_scores.std())

    # ------------------------------------------------------------------
    # Save results summary
    # ------------------------------------------------------------------
    out_file = Path(args.output_dir) / "results_summary.txt"
    with open(out_file, "w") as f:
        f.write("Ghosh 2026 Multi-Domain Feature Baseline Results\n")
        f.write(f"Subject: {args.subject}  |  Dataset: {args.dataset}\n")
        f.write(f"Selected features: {n_final}\n\n")
        for clf_name, m in results.items():
            f.write(f"{clf_name.upper()}: acc={m['acc']:.4f}  prec={m['prec']:.4f}  "
                    f"rec={m['rec']:.4f}  f1={m['f1']:.4f}  auc={m['auc']:.4f}  "
                    f"cv={m['cv_mean']:.4f}±{m['cv_std']:.4f}\n")
    print(f"\nResults saved to: {out_file}")

    json_file = Path(args.output_dir) / "final_metrics.json"
    with open(json_file, "w") as f:
        json.dump(
            {
                "method": "Ghosh_2026_multi_domain_feature_engineering",
                "dataset": args.dataset,
                "subject": args.subject,
                "window_sec": window_sec,
                "selected_features": int(n_final),
                "classifiers": results,
            },
            f,
            indent=2,
        )
    print(f"JSON metrics saved to: {json_file}")

    return results


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Ghosh 2026 multi-domain feature engineering baseline on CHB-MIT."
    )
    add_common_args(
        parser,
        dataset_name="chbmit",
        method_name="ghosh2026_multidomain_rf",
        default_data_dir="./dataset/CHB-MIT-scalp-eeg-database-1.0.0",
        default_output_dir="./outputs/baselines/ghosh2026_chbmit",
        default_subject="all",
    )
    add_ghosh_args(parser)
    parser.add_argument(
        "--balance_train",
        action="store_true",
        help="Downsample the majority class in the training set only.",
    )
    return parser


def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)
    run_ghosh_baseline(args)


if __name__ == "__main__":
    main()
