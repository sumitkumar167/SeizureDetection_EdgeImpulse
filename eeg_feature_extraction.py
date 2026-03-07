"""
eeg_feature_extraction.py
─────────────────────────────────────────────────────────────────────────────
CHB-MIT Scalp EEG → Edge Impulse TinyML Feature Pipeline
─────────────────────────────────────────────────────────────────────────────

Reads .edf EEG recordings and paired *-summary.txt seizure annotations,
segments the signals into overlapping windows, extracts a fixed set of
time-domain, Hjorth, frequency-domain and spectral-entropy features per
window, and writes a flat CSV file ready to upload to Edge Impulse.

Configuration
-------------
Edit the CONFIG block below, then run:
    python eeg_feature_extraction.py

Required libraries
------------------
    pip install mne numpy scipy pandas tqdm
"""

# ═════════════════════════════════════════════════════════════════════════════
# ✏️  USER CONFIGURATION — edit these values before running
# ═════════════════════════════════════════════════════════════════════════════
CONFIG = {
    # Path to the folder that contains the .edf files and *-summary.txt
    "data_dir": "/Users/sumitkumar/Downloads/Lectures/Spring26/Individual Instruction/EEG_Classification/MIT_Scalp_EEG_Dataset/physionet.org/files/chbmit/1.0.0",

    # Output CSV filename
    "output": "seizure_features.csv",

    # Window length in seconds
    "window_sec": 2.0,

    # Fractional overlap between consecutive windows (0.0 – <1.0)
    "overlap": 0.5,

    # ── OPT 2: Channel subset ────────────────────────────────────────────────
    # 4 channels instead of ~23 → ~5× fewer FFT/Hjorth calls per window.
    # Set to None to use ALL channels (much slower).
    "channels": ["F3-C3", "C3-P3", "F4-C4", "C4-P4"],

    # ── OPT 1: Segment trimming ──────────────────────────────────────────────
    # Only process data within this many seconds of each seizure event.
    # Eliminates up to 95 % of samples in long recordings.
    "context_margin_sec": 600,    # ± 10 min around each seizure

    # For files with NO seizures, keep only this many seconds from the start.
    "max_background_sec": 300,    # 5 min of background context

    # ── OPT 6: Parallel processing ───────────────────────────────────────────
    # Number of EDF files to process simultaneously. Set to 1 to disable.
    "n_workers": 4,
}
# ═════════════════════════════════════════════════════════════════════════════

import glob
import multiprocessing
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np
import pandas as pd
from tqdm import tqdm
# scipy.signal.welch removed — replaced by numpy.fft.rfft (_fft_psd helper)

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("ERROR")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

FREQ_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 40.0),
}

FEATURE_COLUMNS = [
    "mean",
    "variance",
    "rms",
    "hjorth_activity",
    "hjorth_mobility",
    "hjorth_complexity",
    "delta_power",
    "theta_power",
    "alpha_power",
    "beta_power",
    "gamma_power",
    "spectral_entropy",
    "label",
]


# ─────────────────────────────────────────────────────────────────────────────
# FFT Helper — computed ONCE per channel per window, shared by all spectral
# features (OPT 4 & 5: eliminates 6 redundant Welch calls per channel)
# ─────────────────────────────────────────────────────────────────────────────

def _fft_psd(
    signal: np.ndarray,
    sfreq: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a one-sided power spectrum via numpy.fft.rfft.

    OPT 4: Replaces 6 independent scipy.signal.welch calls (5 band-power
    queries + 1 entropy query) with a single FFT.  All spectral features
    in extract_features() share this (freqs, power) result.

    Returns
    -------
    freqs : ndarray — frequency axis in Hz
    power : ndarray — power at each frequency bin (|FFT|² / N)
    """
    n      = len(signal)
    fft_v  = np.fft.rfft(signal, n=n)
    power  = (np.abs(fft_v) ** 2) / n           # one-sided power spectrum
    freqs  = np.fft.rfftfreq(n, d=1.0 / sfreq)  # Hz axis
    return freqs, power


# ─────────────────────────────────────────────────────────────────────────────
# 1. EDF Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_edf(
    edf_path: str,
    channel_subset: Optional[List[str]] = None,
) -> Tuple[np.ndarray, float, List[str]]:
    """
    Load an EDF file using MNE.

    Parameters
    ----------
    edf_path : str
        Absolute or relative path to the .edf file.
    channel_subset : list of str or None
        Channel names to keep.  Pass None or an empty list to use all channels.

    Returns
    -------
    data : ndarray, shape (n_channels, n_samples)
        Raw EEG data in µV (converted from MNE's default Volts).
    sfreq : float
        Sampling frequency in Hz.
    ch_names : list of str
        Channel names actually loaded.
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    sfreq = raw.info["sfreq"]

    # Optional channel selection
    if channel_subset:
        available = [ch.upper() for ch in raw.ch_names]
        keep = [ch for ch in raw.ch_names
                if ch.upper() in [c.upper() for c in channel_subset]]
        if not keep:
            print(
                f"  [WARN] None of the requested channels found in {edf_path}. "
                "Using all channels."
            )
            keep = raw.ch_names
        raw.pick_channels(keep)

    data = raw.get_data() * 1e6  # Volts → µV
    ch_names = raw.ch_names
    return data, sfreq, ch_names


# ─────────────────────────────────────────────────────────────────────────────
# 2. Seizure-Interval Parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_seizures(
    summary_path: str,
    edf_filename: str,
) -> List[Tuple[float, float]]:
    """
    Parse seizure start/end times (in seconds) from a CHB-MIT *-summary.txt.

    The function locates the section that belongs to `edf_filename` and
    extracts every Seizure Start/End Time pair within that section.

    Parameters
    ----------
    summary_path : str
        Path to the *-summary.txt file.
    edf_filename : str
        Bare filename of the EDF (e.g. ``"chb01_03.edf"``).

    Returns
    -------
    list of (start_sec, end_sec) tuples, possibly empty.
    """
    seizure_intervals: List[Tuple[float, float]] = []

    with open(summary_path, "r", errors="replace") as fh:
        text = fh.read()

    # Split into per-file sections based on "File Name:" headers
    sections = re.split(r"(?=File Name:)", text, flags=re.IGNORECASE)

    target_section = None
    for section in sections:
        # Match the bare filename, case-insensitively
        if re.search(re.escape(edf_filename), section, re.IGNORECASE):
            target_section = section
            break

    if target_section is None:
        return seizure_intervals  # no information for this file

    starts = [float(m) for m in re.findall(
        r"Seizure(?:\s+\d+)?\s+Start\s+Time\s*:\s*([\d.]+)\s*second",
        target_section, re.IGNORECASE)]
    ends = [float(m) for m in re.findall(
        r"Seizure(?:\s+\d+)?\s+End\s+Time\s*:\s*([\d.]+)\s*second",
        target_section, re.IGNORECASE)]

    if len(starts) != len(ends):
        print(
            f"  [WARN] Mismatched seizure start/end counts in {summary_path} "
            f"for {edf_filename}. Skipping partial entries."
        )
        n = min(len(starts), len(ends))
        starts, ends = starts[:n], ends[:n]

    seizure_intervals = list(zip(starts, ends))
    return seizure_intervals


# ─────────────────────────────────────────────────────────────────────────────
# 3. Signal Windowing
# ─────────────────────────────────────────────────────────────────────────────

def window_signal(
    n_samples: int,
    sfreq: float,
    window_sec: float = 2.0,
    overlap: float = 0.5,
) -> List[Tuple[int, int]]:
    """
    Generate (start, end) sample-index pairs for all windows.

    OPT 3: Replaced Python while-loop with numpy.arange — all start
    indices are generated in one vectorised operation, no Python-level
    iteration required.

    Returns
    -------
    list of (window_start, window_end) tuples (end is exclusive).
    """
    window_size = int(window_sec * sfreq)
    step_size   = max(1, int(window_size * (1.0 - overlap)))

    # Vectorised: generate all start indices at once
    starts = np.arange(0, n_samples - window_size + 1, step_size)
    return list(zip(starts.tolist(), (starts + window_size).tolist()))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Window Labeling
# ─────────────────────────────────────────────────────────────────────────────

def label_window(
    win_start: int,
    win_end: int,
    seizure_intervals: List[Tuple[float, float]],
    sfreq: float,
) -> int:
    """
    Return 1 if the window overlaps any seizure interval, else 0.

    Parameters
    ----------
    win_start, win_end : int
        Window boundaries in sample indices (end is exclusive).
    seizure_intervals : list of (start_sec, end_sec)
    sfreq : float
    """
    for (sz_start_sec, sz_end_sec) in seizure_intervals:
        sz_start = sz_start_sec * sfreq
        sz_end   = sz_end_sec   * sfreq
        # Overlap condition: window ends after seizure starts
        #                    AND window starts before seizure ends
        if win_end >= sz_start and win_start <= sz_end:
            return 1
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# 5a. Hjorth Parameters
# ─────────────────────────────────────────────────────────────────────────────

def compute_hjorth(
    signal: np.ndarray,
    activity: float,
) -> Tuple[float, float, float]:
    """
    Compute Hjorth Activity, Mobility, and Complexity.

    OPT 5: `activity` (variance of the signal) is passed in rather than
    re-computed here.  extract_features() already computes variance for all
    channels at once; passing it in avoids a redundant np.var() call.

    Parameters
    ----------
    signal   : 1-D ndarray
    activity : float — pre-computed variance of `signal`

    Returns
    -------
    (activity, mobility, complexity)
    """
    diff1     = np.diff(signal)
    diff2     = np.diff(diff1)
    var_diff1 = float(np.var(diff1))
    var_diff2 = float(np.var(diff2))

    mobility   = np.sqrt(var_diff1 / activity) if activity > 0 else 0.0
    mobility_d = np.sqrt(var_diff2 / var_diff1) if var_diff1 > 0 else 0.0
    complexity = (mobility_d / mobility) if mobility > 0 else 0.0

    return activity, float(mobility), float(complexity)


# ─────────────────────────────────────────────────────────────────────────────
# 5b. Band Power (FFT-based)
# ─────────────────────────────────────────────────────────────────────────────

def compute_bandpower(
    fft_result: Tuple[np.ndarray, np.ndarray],
    band: Tuple[float, float],
) -> float:
    """
    Compute power in a frequency band from a pre-computed FFT result.

    OPT 4: Accepts `(freqs, power)` from `_fft_psd` instead of running
    its own Welch PSD — eliminates 5 redundant FFT calls per channel per
    window (one per frequency band).

    Parameters
    ----------
    fft_result : (freqs, power) tuple from _fft_psd()
    band       : (low_hz, high_hz)

    Returns
    -------
    Band power (float).
    """
    freqs, power = fft_result
    low, high    = band
    idx_band     = np.logical_and(freqs >= low, freqs <= high)
    if not idx_band.any():
        return 0.0
    return float(np.trapezoid(power[idx_band], freqs[idx_band]))


# ─────────────────────────────────────────────────────────────────────────────
# 5c. Spectral Entropy
# ─────────────────────────────────────────────────────────────────────────────

def compute_entropy(
    fft_result: Tuple[np.ndarray, np.ndarray],
) -> float:
    """
    Compute normalised spectral (Shannon) entropy from a pre-computed FFT.

    OPT 4: Accepts `(freqs, power)` from `_fft_psd` — reuses the same FFT
    already computed for band-power features, eliminating the 6th redundant
    spectral computation per channel per window.

    Steps:
      1. Normalise power to a probability distribution.
      2. Apply Shannon entropy: −Σ p·ln(p).

    Returns
    -------
    Spectral entropy (float, ≥ 0).
    """
    _, power = fft_result
    psd_sum  = power.sum()
    if psd_sum == 0:
        return 0.0
    psd_norm = np.clip(power / psd_sum, 1e-12, None)
    return float(-np.sum(psd_norm * np.log(psd_norm)))


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Master Feature Extractor (per window)
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(
    window_data: np.ndarray,
    sfreq: float,
) -> np.ndarray:
    """
    Extract the feature vector for one EEG window.

    OPT 3 — Vectorised time-domain:
        mean, variance, and RMS are computed for ALL channels simultaneously
        using NumPy axis operations instead of per-channel scalar calls.

    OPT 4 & 5 — Shared FFT, no duplicate variance:
        _fft_psd() is called once per channel; its result is passed into
        compute_bandpower() (×5) and compute_entropy() (×1).  The variance
        already computed for time-domain features is passed directly into
        compute_hjorth() as `activity`, removing a redundant np.var() call.

    Parameters
    ----------
    window_data : ndarray, shape (n_channels, window_samples)
    sfreq : float

    Returns
    -------
    features : 1-D ndarray of length 12
        [mean, variance, rms,
         hjorth_activity, hjorth_mobility, hjorth_complexity,
         delta_power, theta_power, alpha_power, beta_power, gamma_power,
         spectral_entropy]
    """
    data64 = window_data.astype(np.float64)     # (n_ch, n_samples)

    # ── Vectorised time-domain across ALL channels at once ───────────────────
    means = data64.mean(axis=1)                 # (n_ch,)
    vars_ = data64.var(axis=1)                  # (n_ch,)  == hjorth activity
    rms   = np.sqrt((data64 ** 2).mean(axis=1)) # (n_ch,)

    n_channels = data64.shape[0]
    channel_features = []

    for ch in range(n_channels):
        sig     = data64[ch]
        var_val = float(vars_[ch])

        # Hjorth — reuse pre-computed variance as `activity` (OPT 5)
        _, mobility, complexity = compute_hjorth(sig, var_val)

        # Single FFT for all spectral features (OPT 4)
        fft_result = _fft_psd(sig, sfreq)

        band_powers = [
            compute_bandpower(fft_result, FREQ_BANDS["delta"]),
            compute_bandpower(fft_result, FREQ_BANDS["theta"]),
            compute_bandpower(fft_result, FREQ_BANDS["alpha"]),
            compute_bandpower(fft_result, FREQ_BANDS["beta"]),
            compute_bandpower(fft_result, FREQ_BANDS["gamma"]),
        ]

        sp_entropy = compute_entropy(fft_result)

        channel_features.append(
            [float(means[ch]), var_val, float(rms[ch]),
             var_val, mobility, complexity,
             *band_powers,
             sp_entropy]
        )

    # Average across channels → shape (12,)
    return np.mean(channel_features, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# OPT 1 Helper — Trim recording to seizure context window
# ─────────────────────────────────────────────────────────────────────────────

def _clip_to_context(
    data: np.ndarray,
    sfreq: float,
    seizure_intervals: List[Tuple[float, float]],
    margin_sec: float,
    max_background_sec: float,
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """
    Trim the recording to the region of interest.

    OPT 1: Skips long stretches of pure background EEG that carry no new
    information for the classifier, reducing windows by up to 95 %.

    - Seizure files  → keep [earliest_start − margin, latest_end + margin].
    - No-seizure files → keep only first `max_background_sec` seconds.

    Returns adjusted data array and seizure times relative to new t = 0.
    """
    n_samples    = data.shape[1]
    duration_sec = n_samples / sfreq

    if seizure_intervals:
        clip_start_sec = max(0.0, min(s for s, _ in seizure_intervals) - margin_sec)
        clip_end_sec   = min(duration_sec, max(e for _, e in seizure_intervals) + margin_sec)
    else:
        clip_start_sec = 0.0
        clip_end_sec   = min(duration_sec, max_background_sec)

    s = int(clip_start_sec * sfreq)
    e = int(clip_end_sec   * sfreq)
    data_clipped = data[:, s:e]

    # Shift seizure times to be relative to new start
    adjusted = [
        (max(0.0, sz_s - clip_start_sec), max(0.0, sz_e - clip_start_sec))
        for (sz_s, sz_e) in seizure_intervals
    ]
    return data_clipped, adjusted


# ─────────────────────────────────────────────────────────────────────────────
# OPT 6 Helper — Top-level worker (must be picklable for multiprocessing)
# ─────────────────────────────────────────────────────────────────────────────

def _process_one_edf(args: tuple) -> dict:
    """
    Process a single EDF file end-to-end and return results as a plain dict.

    OPT 6: Defined at module level so it is picklable by multiprocessing.
    Called by generate_dataset() via Pool.imap_unordered().
    """
    (edf_path, summary_path, window_sec, overlap,
     channel_subset, margin_sec, max_background_sec) = args

    edf_name = os.path.basename(edf_path)
    t0 = time.perf_counter()

    try:
        data, sfreq, ch_names = load_edf(edf_path, channel_subset)
    except Exception as exc:
        return {"rows": [], "total": 0, "seizure": 0,
                "name": edf_name, "error": str(exc), "elapsed": 0.0,
                "duration_orig": 0.0, "duration_clipped": 0.0,
                "channels": 0, "sfreq": 0.0}

    n_channels, n_samples = data.shape
    duration_orig = n_samples / sfreq

    # Parse seizure annotations
    seizure_intervals: List[Tuple[float, float]] = []
    if summary_path:
        seizure_intervals = parse_seizures(summary_path, edf_name)

    # OPT 1: Trim to context window around seizures
    data, seizure_intervals = _clip_to_context(
        data, sfreq, seizure_intervals, margin_sec, max_background_sec
    )
    _, n_clipped = data.shape

    # Window and extract
    windows  = window_signal(n_clipped, sfreq, window_sec, overlap)
    rows     = []
    n_seizure = 0

    for (win_start, win_end) in windows:
        label = label_window(win_start, win_end, seizure_intervals, sfreq)
        if label == 1:
            n_seizure += 1
        features = extract_features(data[:, win_start:win_end], sfreq)
        rows.append([*features.tolist(), label])

    return {
        "rows":             rows,
        "total":            len(windows),
        "seizure":          n_seizure,
        "name":             edf_name,
        "channels":         n_channels,
        "sfreq":            sfreq,
        "duration_orig":    duration_orig,
        "duration_clipped": n_clipped / sfreq,
        "error":            None,
        "elapsed":          time.perf_counter() - t0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Full Dataset Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(
    data_dir: str,
    output_csv: str,
    window_sec: float = 2.0,
    overlap: float = 0.5,
    channel_subset: Optional[List[str]] = None,
    context_margin_sec: float = 600.0,
    max_background_sec: float = 300.0,
    n_workers: int = 4,
) -> pd.DataFrame:
    """
    Walk `data_dir` for EDF files, extract windowed features, and save CSV.

    OPT 6: Uses multiprocessing.Pool to process EDF files in parallel.
    OPT 7: Prints per-file elapsed time and running totals.
    """
    data_dir = str(Path(data_dir).resolve())

    # ── Locate EDF files ────────────────────────────────────────────────────
    edf_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.edf"),
                                 recursive=True))
    if not edf_files:
        raise FileNotFoundError(f"No .edf files found under: {data_dir}")

    # ── Locate the closest summary file per patient subfolder ───────────────
    # Build a map: edf_path → summary_path (search same dir, then parent subtree)
    all_summaries = sorted(glob.glob(
        os.path.join(data_dir, "**", "*summary*.txt"), recursive=True))

    def _find_summary(edf_path: str) -> Optional[str]:
        edf_dir = os.path.dirname(edf_path)
        for s in all_summaries:
            if os.path.dirname(s) == edf_dir:
                return s
        return all_summaries[0] if all_summaries else None

    if not all_summaries:
        print("[WARN] No *-summary.txt found. All windows will be labeled 0.")

    print(f"\n{'─'*60}")
    print(f"  EEG Feature Extraction Pipeline  [OPTIMIZED]")
    print(f"{'─'*60}")
    print(f"  Data directory   : {data_dir}")
    print(f"  EDF files found  : {len(edf_files)}")
    print(f"  Window length    : {window_sec} s   Overlap: {overlap*100:.0f}%")
    print(f"  Channels         : {'all' if not channel_subset else channel_subset}")
    print(f"  Context margin   : ±{context_margin_sec/60:.0f} min  (OPT 1)")
    print(f"  Max background   : {max_background_sec/60:.0f} min per no-seizure file")
    print(f"  Parallel workers : {n_workers}  (OPT 6)")
    print(f"{'─'*60}\n")

    # ── Build argument list for workers ─────────────────────────────────────
    worker_args = [
        (edf_path, _find_summary(edf_path),
         window_sec, overlap, channel_subset,
         context_margin_sec, max_background_sec)
        for edf_path in edf_files
    ]

    all_rows:       List[List] = []
    total_windows  = 0
    seizure_windows = 0
    n_total        = len(edf_files)
    n_done         = 0
    wall_start     = time.perf_counter()

    # ── OPT 6: Process files in parallel ────────────────────────────────────
    pool_ctx = (multiprocessing.Pool(n_workers)
                if n_workers > 1 else None)
    try:
        iterator = (
            pool_ctx.imap_unordered(_process_one_edf, worker_args)
            if pool_ctx else
            map(_process_one_edf, worker_args)
        )

        for result in iterator:
            n_done += 1
            name    = result["name"]

            # OPT 7: Progress reporting
            if result["error"]:
                print(f"  [{n_done:>3}/{n_total}]  ✗ {name}  ERROR: {result['error']}")
                continue

            kept_pct = (result["duration_clipped"] / result["duration_orig"] * 100
                        if result["duration_orig"] > 0 else 0)
            print(
                f"  [{n_done:>3}/{n_total}]  ▶ {name}  "
                f"ch:{result['channels']}  "
                f"orig:{result['duration_orig']:.0f}s → "
                f"clipped:{result['duration_clipped']:.0f}s ({kept_pct:.0f}%)  "
                f"wins:{result['total']}  "
                f"sz:{result['seizure']}  "
                f"elapsed:{result['elapsed']:.1f}s"
            )

            all_rows.extend(result["rows"])
            total_windows   += result["total"]
            seizure_windows += result["seizure"]

    finally:
        if pool_ctx:
            pool_ctx.close()
            pool_ctx.join()

    # ── Assemble DataFrame ───────────────────────────────────────────────────
    if not all_rows:
        raise RuntimeError("No windows were extracted. Check data_dir and files.")

    df = pd.DataFrame(all_rows, columns=FEATURE_COLUMNS)
    df.to_csv(output_csv, index=False)

    # ── Diagnostics ──────────────────────────────────────────────────────────
    nonseizure_windows = total_windows - seizure_windows
    ratio = (seizure_windows / nonseizure_windows
             if nonseizure_windows > 0 else float("inf"))
    total_elapsed = time.perf_counter() - wall_start

    print(f"\n{'═'*60}")
    print(f"  Dataset Statistics")
    print(f"{'─'*60}")
    print(f"  Total windows      : {total_windows:>10,}")
    print(f"  Seizure windows    : {seizure_windows:>10,}  (label = 1)")
    print(f"  Non-seizure windows: {nonseizure_windows:>10,}  (label = 0)")
    if ratio > 0 and ratio != float("inf"):
        print(f"  Class imbalance    : 1 : {1/ratio:.1f}  (seizure : non-seizure)")
    else:
        print(f"  Class imbalance    : no seizure windows detected")
    print(f"  Total elapsed time : {total_elapsed:.1f} s  ({total_elapsed/60:.1f} min)")
    print(f"  Output CSV         : {os.path.abspath(output_csv)}")
    print(f"{'═'*60}\n")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    generate_dataset(
        data_dir=CONFIG["data_dir"],
        output_csv=CONFIG["output"],
        window_sec=CONFIG["window_sec"],
        overlap=CONFIG["overlap"],
        channel_subset=CONFIG["channels"],
        context_margin_sec=CONFIG["context_margin_sec"],
        max_background_sec=CONFIG["max_background_sec"],
        n_workers=CONFIG["n_workers"],
    )


if __name__ == "__main__":
    main()
