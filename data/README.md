# Datasets

Three publicly available EEG datasets are used across the MV-AFA paper and its benchmarks.
None of the raw data is included in this repository — download instructions are provided below.

---

## 1. CHB-MIT Scalp EEG Database

**Used by:** All four benchmark methods + MV-AFA

### Description

The CHB-MIT Scalp EEG Database was collected at Boston Children's Hospital and is jointly
maintained by MIT. It contains long-term, continuous scalp EEG recordings from **24 paediatric
patients** (ages 1.5–22 years) with intractable epilepsy. Recordings span multiple sessions
per patient and cover both ictal (seizure) and interictal (non-seizure) periods.

| Property | Value |
|----------|-------|
| Subjects | 24 (paediatric) |
| Total recordings | 664 EDF files |
| Total duration | ~979 hours |
| Channels | 18–23 (international 10–20 system) |
| Sampling rate | 256 Hz |
| Seizure events | 198 annotated |
| Format | European Data Format (EDF) |

### Channel subset used in benchmarks

Most baselines use the following **18 bipolar channels** for consistency across subjects:

```
FP1-F7, F7-T7, T7-P7, P7-O1,
FP1-F3, F3-C3, C3-P3, P3-O1,
FP2-F4, F4-C4, C4-P4, P4-O2,
FP2-F8, F8-T8, T8-P8, P8-O2,
FZ-CZ,  CZ-PZ
```

### Download

**PhysioNet (free, registration required):**
> https://physionet.org/content/chbmit/1.0.0/

```bash
# Using PhysioNet wget script
wget -r -N -c -np https://physionet.org/files/chbmit/1.0.0/

# Or using the PhysioNet client
pip install wfdb
python -c "import wfdb; wfdb.dl_database('chbmit', './data/CHB-MIT-scalp-eeg-database-1.0.0')"
```

### Citation

```
Shoeb AH (2009). Application of Machine Learning to Epileptic Seizure Onset Detection
and Treatment. PhD Thesis, MIT.
Goldberger AL et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.
Circulation 101(23):e215-e220.
```

---

## 2. Siena Scalp EEG Database

**Used by:** Li 2025 (CMFViT cross-subject evaluation) + MV-AFA

### Description

The Siena Scalp EEG Database was collected at the Unit of Neurology and Neurophysiology,
University of Siena, Italy. It contains EEG recordings from **14 adult patients** with
epilepsy, covering a total of 47 seizures. All recordings were performed during standard
video-EEG monitoring sessions; ictal and interictal segments are annotated by clinical
neurophysiologists.

| Property | Value |
|----------|-------|
| Subjects | 14 (adults) |
| Total seizures | 47 annotated |
| Channels | 19–29 (international 10–20 system) |
| Sampling rate | 512 Hz |
| Total duration | ~128 hours |
| Seizure types | Focal, generalised |
| Format | European Data Format (EDF) |

### Download

**PhysioNet (free, registration required):**
> https://physionet.org/content/siena-scalp-eeg/1.0.0/

```bash
# Using PhysioNet wget script
wget -r -N -c -np https://physionet.org/files/siena-scalp-eeg/1.0.0/

# Or using the PhysioNet client
pip install wfdb
python -c "import wfdb; wfdb.dl_database('siena-scalp-eeg', './data/siena-scalp-eeg-1.0.0')"
```

### Citation

```
Detti P et al. (2020). Siena Scalp EEG Database (version 1.0.0).
PhysioNet. https://doi.org/10.13026/5d4a-j060

Detti P et al. (2020). EEG synchronization analysis for seizure prediction:
A study on data of noninvasive recordings.
Processes 8(7):846. doi: 10.3390/pr8070846
```

---

## 3. Temple University Hospital EEG Seizure Corpus (TUSZ)

**Used by:** Xu 2026 (TUH), PSD-LW-DCN 2026 (TUSZ) + MV-AFA

### Description

The Temple University Hospital EEG (TUH EEG) corpus is the **largest publicly available
clinical EEG dataset**, collected from routine EEG recordings at Temple University Hospital.
The TUSZ (TUH Seizure) subset provides expert-annotated seizure events with detailed
seizure type labels.

| Property | Value |
|----------|-------|
| Total patients | 675 |
| Total sessions | 1,643 |
| Total recordings | 3,971 seizure events |
| Total seizure duration | ~1,474 hours |
| Channels | 20–128 (varies per recording) |
| Sampling rate | 250 Hz (16-bit) |
| Seizure types | Multiple (FNSZ, GNSZ, ABSZ, etc.) |
| Format | EDF + TSV annotations |

### Download

**Requires free registration with Temple University:**
> https://isip.piconepress.com/projects/tuh_eeg/

```bash
# After registration, use rsync (credentials provided by TUH)
rsync -auxvL nedc_tuh_eeg@www.isip.piconepress.com:data/tuh_eeg_seizure/ ./data/tusz/
```

> **Note:** The TUSZ dataset requires a data use agreement. Registration is free for
> academic research. Approval typically takes 1–3 business days.

### Version used in benchmarks

Baselines in this repo reference **TUSZ v1.5.1** (or a controlled 20-subject subset for
reproducible experiments).

### Citation

```
Obeid I and Picone J (2016). The Temple University Hospital EEG Data Corpus.
Frontiers in Neuroscience 10:196. doi: 10.3389/fnins.2016.00196

Shah V et al. (2018). The Temple University Hospital Seizure Detection Corpus.
Frontiers in Neuroinformatics 12:83. doi: 10.3389/fninf.2018.00083
```

---

## Data Directory Layout (after download)

```
data/
├── README.md                              ← This file
├── CHB-MIT-scalp-eeg-database-1.0.0/     ← CHB-MIT raw EDF files
│   ├── chb01/
│   │   ├── chb01_01.edf
│   │   ├── ...
│   │   └── chb01-summary.txt
│   ├── chb02/
│   └── ...
├── siena-scalp-eeg-1.0.0/                ← Siena EDF files
│   ├── PN00/
│   ├── PN01/
│   └── ...
└── tusz/                                  ← TUSZ EDF + annotation files
    ├── train/
    ├── dev/
    └── ...
```

---

## Notes on Data Preprocessing

All preprocessing (bandpass filtering, segmentation, normalisation) is performed **on-the-fly**
inside each baseline script to ensure reproducibility and avoid storing processed derivatives.
See the `--l_freq`, `--h_freq`, `--window_sec`, and `--step_sec` arguments in each script.
