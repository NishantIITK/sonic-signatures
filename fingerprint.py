"""
fingerprint.py
================
Core Shazam-style audio fingerprinting engine for EE200 Q3A / Q3B.

This single module is the one source of truth for the algorithm: the Kaggle
notebook (Q3A) imports it for the research/analysis, and the Streamlit app
(Q3B) imports the exact same functions so behaviour never drifts between the
report and the deployed app.

Pipeline
--------
audio -> spectrogram -> constellation (peaks) -> pair hashes (f1, f2, dt)
      -> database {hash: [(song, anchor_time), ...]}
      -> query matching via offset-histogram voting
"""

import os
import glob
import pickle
from collections import defaultdict

import numpy as np
import librosa
from scipy.ndimage import maximum_filter

# ----------------------------------------------------------------------------
# Default configuration (balanced for music fingerprinting, see report)
# ----------------------------------------------------------------------------
SR        = 22050   # uniform resample rate (Hz)
N_FFT     = 2048     # STFT window length
HOP_LEN   = 512      # STFT hop length
FAN_VALUE = 15       # max forward neighbours per anchor peak
MIN_DT    = 1        # min Δt between anchor & target (frames)
MAX_DT    = 200      # max Δt
NBHD      = (20, 20) # (freq_bins, time_frames) neighbourhood for peak-picking
THRESH_DB = -40      # dB threshold relative to spectrogram max

AUDIO_EXTS = ("*.mp3", "*.wav", "*.flac", "*.m4a", "*.ogg")


# ----------------------------------------------------------------------------
# Dataset discovery (works the same on Kaggle, Streamlit Cloud, or locally)
# ----------------------------------------------------------------------------
def list_audio_files(song_dir):
    """Return a sorted list of audio files directly inside or nested under song_dir."""
    files = []
    for ext in AUDIO_EXTS:
        files += glob.glob(os.path.join(song_dir, ext))
        files += glob.glob(os.path.join(song_dir, "**", ext), recursive=True)
    return sorted(set(files))


def find_song_dir(extra_candidates=None):
    """
    Auto-detect a folder containing the song library.

    Search order: any extra_candidates given -> every /kaggle/input/* dataset
    folder -> ./songs -> current directory. Raises if nothing is found.
    """
    candidates = list(extra_candidates) if extra_candidates else []
    candidates += sorted(glob.glob("/kaggle/input/*"))
    candidates += ["./songs", "songs", "."]
    for c in candidates:
        if os.path.isdir(c) and list_audio_files(c):
            return c
    raise FileNotFoundError(
        "Could not find a folder with audio files (.mp3/.wav/.flac/.m4a/.ogg). "
        "On Kaggle: Add Data -> attach the song-library dataset. "
        "Locally: place songs in a 'songs/' folder."
    )


# ----------------------------------------------------------------------------
# Spectrogram + constellation
# ----------------------------------------------------------------------------
def compute_spectrogram(y, sr=SR, n_fft=N_FFT, hop_length=HOP_LEN):
    """Log-magnitude STFT spectrogram. Shape: (freq_bins, time_frames)."""
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    return librosa.amplitude_to_db(S, ref=np.max)


def extract_peaks(S_db, neighborhood=NBHD, threshold_db=THRESH_DB):
    """
    Local-maxima ("constellation") peaks of a log-magnitude spectrogram.
    Returns a list of (time_frame_idx, freq_bin_idx), sorted by time.
    """
    loc_max = maximum_filter(S_db, size=neighborhood)
    is_peak = (S_db == loc_max) & (S_db > S_db.max() + threshold_db)
    f_idx, t_idx = np.where(is_peak)
    return sorted(zip(t_idx.tolist(), f_idx.tolist()), key=lambda p: p[0])


# ----------------------------------------------------------------------------
# Hashing
# ----------------------------------------------------------------------------
def generate_pair_hashes(peaks, fan_value=FAN_VALUE, min_dt=MIN_DT, max_dt=MAX_DT):
    """Pair each anchor peak with up to fan_value forward peaks -> (f1,f2,dt)."""
    hashes, n = [], len(peaks)
    for i in range(n):
        t1, f1 = peaks[i]
        cnt = 0
        for j in range(i + 1, n):
            if cnt >= fan_value:
                break
            t2, f2 = peaks[j]
            dt = t2 - t1
            if dt < min_dt:
                continue
            if dt > max_dt:
                break
            hashes.append(((int(f1), int(f2), int(dt)), t1))
            cnt += 1
    return hashes


def generate_single_hashes(peaks):
    """Single-peak fingerprints h=(f,) — used as the weaker baseline."""
    return [((int(f),), t) for t, f in peaks]


def generate_ratio_hashes(peaks, fan_value=FAN_VALUE, min_dt=MIN_DT, max_dt=MAX_DT, decimals=2):
    """
    Pitch-invariant fingerprint: h = (round(f2/f1, decimals), dt).
    Under a uniform pitch shift f -> alpha*f, the ratio f2/f1 is unchanged,
    so this hash survives pitch shifts that break the plain (f1,f2,dt) hash.
    """
    hashes, n = [], len(peaks)
    for i in range(n):
        t1, f1 = peaks[i]
        if f1 == 0:
            continue
        cnt = 0
        for j in range(i + 1, n):
            if cnt >= fan_value:
                break
            t2, f2 = peaks[j]
            dt = t2 - t1
            if dt < min_dt:
                continue
            if dt > max_dt:
                break
            ratio = round(f2 / f1, decimals)
            hashes.append(((ratio, int(dt)), t1))
            cnt += 1
    return hashes


def fingerprint_audio(y, sr=SR, n_fft=N_FFT, hop_length=HOP_LEN, nbhd=NBHD,
                       thresh=THRESH_DB, fan=FAN_VALUE, min_dt=MIN_DT, max_dt=MAX_DT):
    """Full pipeline for one clip: spectrogram -> peaks -> pair/single hashes."""
    S_db = compute_spectrogram(y, sr, n_fft, hop_length)
    peaks = extract_peaks(S_db, nbhd, thresh)
    pair_hashes = generate_pair_hashes(peaks, fan, min_dt, max_dt)
    single_hashes = generate_single_hashes(peaks)
    return S_db, peaks, pair_hashes, single_hashes


# ----------------------------------------------------------------------------
# Database build / persist
# ----------------------------------------------------------------------------
def build_database(song_dir=None, song_files=None, sr=SR, verbose=True, **kw):
    """
    Fingerprint every song and populate two lookup tables:
      db_pair[hash]   -> [(song_name, anchor_time_frame), ...]
      db_single[hash] -> [(song_name, anchor_time_frame), ...]
    """
    if song_files is None:
        song_dir = song_dir or find_song_dir()
        song_files = list_audio_files(song_dir)
    if not song_files:
        raise FileNotFoundError("No song files supplied to build_database().")

    db_p, db_s, stats = defaultdict(list), defaultdict(list), {}
    for path in song_files:
        name = os.path.splitext(os.path.basename(path))[0]
        if verbose:
            print(f"  indexing {name!r} ...", end=" ", flush=True)
        y, _ = librosa.load(path, sr=sr, mono=True)
        _, peaks, ph, sh = fingerprint_audio(y, sr, **kw)
        for h, t in ph:
            db_p[h].append((name, t))
        for h, t in sh:
            db_s[h].append((name, t))
        stats[name] = {
            "peaks": len(peaks),
            "pair_hashes": len(ph),
            "duration_s": round(len(y) / sr, 1),
        }
        if verbose:
            print(f"{len(peaks):,} peaks, {len(ph):,} pair-hashes")
    return dict(db_p), dict(db_s), stats


def save_database(path, db_p, db_s, stats, song_files):
    with open(path, "wb") as f:
        pickle.dump(
            {"db_p": db_p, "db_s": db_s, "stats": stats, "song_files": song_files}, f
        )


def load_database(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["db_p"], d["db_s"], d["stats"], d["song_files"]


# ----------------------------------------------------------------------------
# Query matching (offset-histogram voting)
# ----------------------------------------------------------------------------
def match_clip(y_clip, sr=SR, db_p=None, db_s=None, **kw):
    """
    Fingerprint a query clip and vote against both databases.

    Returns a dict with the winning song for pair-hash and single-hash
    methods, their score dicts, the full offset histograms (for plotting),
    the query's peaks, and its spectrogram.
    """
    S_q, peaks_q, ph_q, sh_q = fingerprint_audio(y_clip, sr, **kw)

    off_p = defaultdict(lambda: defaultdict(int))
    off_s = defaultdict(lambda: defaultdict(int))

    for h, tq in ph_q:
        for sname, tdb in (db_p or {}).get(h, []):
            off_p[sname][tdb - tq] += 1
    for h, tq in sh_q:
        for sname, tdb in (db_s or {}).get(h, []):
            off_s[sname][tdb - tq] += 1

    def best(od):
        sc = {s: max(d.values()) for s, d in od.items() if d}
        return (max(sc, key=sc.get) if sc else None), sc

    mp, sp = best(off_p)
    ms, ss = best(off_s)

    return {
        "pair_match": mp, "pair_scores": sp, "pair_offsets": dict(off_p),
        "single_match": ms, "single_scores": ss, "single_offsets": dict(off_s),
        "peaks": peaks_q, "S_db": S_q,
    }


def safe_pitch_shift(y, sr, n_steps):
    """librosa's pitch_shift signature changed across versions; handle both."""
    try:
        return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)
    except TypeError:
        return librosa.effects.pitch_shift(y, sr, n_steps)


def safe_time_stretch(y, rate):
    """librosa's time_stretch signature changed across versions; handle both."""
    try:
        return librosa.effects.time_stretch(y=y, rate=rate)
    except TypeError:
        return librosa.effects.time_stretch(y, rate)


def add_awgn(y, snr_db, rng=None):
    """Add white Gaussian noise to y at the given SNR (dB)."""
    rng = rng or np.random
    sig_p = np.mean(y ** 2)
    noise_p = sig_p / (10 ** (snr_db / 10))
    noise = rng.standard_normal(len(y)) * np.sqrt(noise_p)
    return np.clip(y + noise, -1.0, 1.0).astype(np.float32)
