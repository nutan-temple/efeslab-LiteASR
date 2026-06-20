#!/usr/bin/env python3
"""
imu_enhance.py - improved IMU-accelerometer -> audio enhancement, designed for the FIXED
3333 Hz hardware constraint (Nyquist ~= 1666 Hz, usable speech band ~80-1600 Hz).

This is a drop-in successor to imu_denoise.py. It keeps the same CSV front-end and CLI shape,
but fixes the two failure modes seen on NORMAL-volume speakers:

  1. Hallucinations on noise.  MMSE/Wiener/specsub shape broadband noise into speech-like
     spectra, then norm_rms() amplifies noise-only clips to full loudness -> the STT invents
     text (the "Hindi/Russian gibberish" you saw on bp_mmse). Fixes here:
       * OM-LSA + MCRA denoiser with an explicit Speech-Presence-Probability (SPP) gain. Noise
         bins are pushed to a floor instead of being sculpted into harmonics.
       * VAD gate to TRUE silence: non-speech frames become hard zeros before normalization.
       * SNR-conditional normalization: quiet/noise-only clips are emitted as silence, never
         boosted to target loudness.
       * An SNR estimate is returned so a caller can refuse to send junk to the STT (see
         stt_gate.py).

  2. Weak signal on normal speakers.  Within the 1.6 kHz ceiling we squeeze out every dB:
       * Band-pass EACH axis BEFORE combining (the caveat from the old combine_axes_pca).
       * SNR-WEIGHTED multi-axis combine (the axis best-coupled to bone vibration dominates),
         instead of plain PCA which maximizes gross-motion variance.

What this CANNOT do: it cannot synthesize the >1.6 kHz fricative/consonant band that the
hardware never captured. For that you need the supervised model in train_bcse.py (it learns
the missing band from your paired mic data). Classical DSP is bounded by Nyquist.

Deps: numpy, scipy  (pandas only for the CSV front-end).

CLI examples:
  python imu_enhance.py raw.csv --axis pca --denoise omlsa
  python imu_enhance.py raw.csv --axis snr --denoise omlsa --vad-gate --snr-norm
  python imu_enhance.py raw.csv --denoise omlsa --vad-gate --snr-norm --min-snr-db 3 -o out.wav

Library use:
  t, x3 = load_axes("raw.csv")              # N x 3 (X,Y,Z)
  sig, sr, snr = reconstruct_multi(t, x3, 3333, denoise="omlsa", vad_gate=True, snr_norm=True)
"""
from __future__ import annotations
import argparse
import os
import wave
import numpy as np


# ============================================================ front-end (CSV -> samples)
def load_axis(csv, axis="Accel Z", ts_col="timestamp"):
    """Return (timestamps_ms, x). `axis` is a column name, 'mag', 'pca', or 'snr'
    ('pca'/'snr' combine Accel X/Y/Z; see reconstruct_multi for the better path)."""
    import pandas as pd
    df = pd.read_csv(csv)
    if ts_col not in df.columns:
        raise SystemExit(f"timestamp column {ts_col!r} not found; columns: {list(df.columns)}")
    if axis in ("pca", "mag", "snr"):
        need = ["Accel X", "Accel Y", "Accel Z"]
        miss = [c for c in need if c not in df.columns]
        if miss:
            raise SystemExit(f"--axis {axis} needs {need}; missing {miss}")
        sub = df[[ts_col] + need].dropna(subset=need).sort_values(ts_col)
        t = sub[ts_col].to_numpy(np.float64)
        xyz = sub[need].to_numpy(np.float64)
        return t, xyz                                  # caller combines (axis decides how)
    if axis not in df.columns:
        raise SystemExit(f"column {axis!r} not found; columns: {list(df.columns)}")
    sub = df[[ts_col, axis]].dropna(subset=[axis]).sort_values(ts_col)
    t = sub[ts_col].to_numpy(np.float64)
    x = sub[axis].to_numpy(np.float64)
    if len(x) < 2:
        raise SystemExit(f"only {len(x)} samples for axis {axis!r}")
    return t, x


def load_axes(csv, ts_col="timestamp", cols=("Accel X", "Accel Y", "Accel Z")):
    """Return (timestamps_ms, N x C array) for the multi-axis path."""
    import pandas as pd
    df = pd.read_csv(csv)
    miss = [c for c in (ts_col,) + tuple(cols) if c not in df.columns]
    if miss:
        raise SystemExit(f"missing columns {miss}; have {list(df.columns)}")
    sub = df[[ts_col] + list(cols)].dropna(subset=list(cols)).sort_values(ts_col)
    return sub[ts_col].to_numpy(np.float64), sub[list(cols)].to_numpy(np.float64)


def infer_rate(t_ms):
    d = np.diff(t_ms)
    d = d[d > 0]
    return max(1, int(round(1000.0 / np.median(d)))) if len(d) else 3333


# ============================================================ gap handling / resample
def largest_contiguous_chunk(t_ms, x, gap_mult=3.0):
    """Largest run of samples with no timestamp gap > median_dt * gap_mult."""
    t = np.asarray(t_ms, float)
    x = np.asarray(x)
    if len(t) < 2:
        return t, x
    d = np.diff(t)
    pos = d[d > 0]
    if not len(pos):
        return t, x
    thr = float(np.median(pos)) * gap_mult
    bounds, s = [], 0
    for b in np.where(d > thr)[0]:
        bounds.append((s, int(b) + 1))
        s = int(b) + 1
    bounds.append((s, len(t)))
    lo, hi = max(bounds, key=lambda r: r[1] - r[0])
    return t[lo:hi], x[lo:hi]


def resample_uniform(t_ms, x, sr):
    """Interp irregularly-timed samples (ms timestamps) onto a uniform grid at sr Hz.
    Works on 1-D x or N x C x (resamples each column)."""
    t_ms = np.asarray(t_ms, float)
    x = np.asarray(x, float)
    if len(t_ms) < 2:
        return x.astype(np.float32)
    keep = np.concatenate([[True], np.diff(t_ms) > 0])   # drop non-increasing timestamps
    t_ms, x = t_ms[keep], x[keep]
    t = (t_ms - t_ms[0]) / 1000.0
    n = max(1, int(round(t[-1] * sr)))
    grid = np.arange(n) / sr
    if x.ndim == 1:
        return np.interp(grid, t, x).astype(np.float32)
    return np.stack([np.interp(grid, t, x[:, c]) for c in range(x.shape[1])], axis=1).astype(np.float32)


def resample_poly_to(x, sr_in, sr_out):
    """Anti-aliased polyphase resample (FIR under the hood)."""
    from math import gcd
    from scipy.signal import resample_poly
    x = np.asarray(x, float)
    if sr_in == sr_out or len(x) < 2:
        return x.astype(np.float32)
    g = gcd(int(sr_out), int(sr_in))
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


# ============================================================ filters
def apply_bandpass(x, sr, low_hz, high_hz, order=4):
    """IIR Butterworth band-pass, zero-phase (filtfilt)."""
    from scipy.signal import butter, filtfilt
    x = np.asarray(x, float)
    if len(x) <= order * 3:
        return x.astype(np.float32)
    nyq = sr / 2.0
    hi = min(float(high_hz), nyq * 0.999)
    lo = max(float(low_hz), 0.001)
    if lo >= hi:
        b, a = butter(order, lo / nyq, btype="high")
    else:
        b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, x).astype(np.float32)


# ============================================================ multi-axis combine
def _bandpower(x, sr, low, high):
    """Crude in-band power via FFT (used for SNR weighting)."""
    x = np.asarray(x, float)
    X = np.abs(np.fft.rfft(x - x.mean())) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    band = (f >= low) & (f <= high)
    return float(X[band].sum() + 1e-12)


def combine_axes_pca(axes, sr=None, low=80.0, high=1600.0, prefilter=True):
    """First principal component, but (by default) band-pass EACH axis first so PCA tracks the
    speech vibration, not gross body motion. `sr` required when prefilter=True."""
    A = np.asarray(axes, dtype=np.float64)
    if A.ndim == 1:
        return A.astype(np.float32)
    if prefilter:
        if sr is None:
            raise ValueError("prefilter=True needs sr")
        A = np.stack([apply_bandpass(A[:, c], sr, low, high) for c in range(A.shape[1])], axis=1).astype(np.float64)
    A = A - A.mean(axis=0)
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    return (A @ Vt[0]).astype(np.float32)


def combine_axes_snr(axes, sr, low=80.0, high=1600.0):
    """SNR-WEIGHTED combine: band-pass each axis, weight each by its in-band/out-of-band power
    ratio (proxy for speech-vibration coupling), then sum. Phase-aligned to the strongest axis to
    avoid destructive cancellation. Beats PCA when one axis is best-coupled to bone."""
    A = np.asarray(axes, dtype=np.float64)
    if A.ndim == 1:
        return apply_bandpass(A, sr, low, high)
    C = A.shape[1]
    bp = np.stack([apply_bandpass(A[:, c], sr, low, high) for c in range(C)], axis=1)
    weights = np.zeros(C)
    for c in range(C):
        in_band = _bandpower(bp[:, c], sr, low, high)
        full = _bandpower(A[:, c] - A[:, c].mean(), sr, 0.1, sr / 2.0)
        weights[c] = in_band / full                    # fraction of energy that is in-band
    if weights.sum() <= 0:
        weights = np.ones(C)
    weights = weights / weights.sum()
    ref = int(np.argmax(weights))                      # align signs to strongest axis
    signs = np.array([np.sign(np.corrcoef(bp[:, ref], bp[:, c])[0, 1] or 1.0) for c in range(C)])
    signs[signs == 0] = 1.0
    return (bp * (weights * signs)).sum(axis=1).astype(np.float32)


# ============================================================ denoiser: OM-LSA + MCRA
def _mcra_noise_psd(P, ad=0.95, ap=0.2, delta=5.0, win=40, beta=0.8):
    """Minima-Controlled Recursive Averaging (Cohen & Berdugo 2001) noise-PSD tracker.
    Returns (noise_psd, spp) where spp is per-bin speech-presence probability. This is what lets
    the gain DISTINGUISH speech bins from noise bins, instead of shaping everything."""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    nb, nf = P.shape
    S = uniform_filter1d(P, size=3, axis=0)            # light frequency smoothing
    Sf = np.zeros_like(S)
    Sf[:, 0] = S[:, 0]
    for l in range(1, nf):                             # temporal smoothing
        Sf[:, l] = beta * Sf[:, l - 1] + (1 - beta) * S[:, l]
    Smin = minimum_filter1d(Sf, size=win, axis=1)      # running spectral minimum
    Sr = Sf / (Smin + 1e-12)                           # ratio -> speech indicator
    I = (Sr > delta).astype(float)
    p = np.zeros_like(P)                               # speech-presence probability
    for l in range(1, nf):
        p[:, l] = ap * p[:, l - 1] + (1 - ap) * I[:, l]
    ad_tilde = ad + (1 - ad) * p                       # adaptive smoothing
    N = np.zeros_like(P)
    N[:, 0] = P[:, 0]
    for l in range(1, nf):
        N[:, l] = ad_tilde[:, l] * N[:, l - 1] + (1 - ad_tilde[:, l]) * P[:, l]
    return N, p


def denoise_omlsa(signal, fs, alpha=0.92, nperseg=256, noverlap=192, gain_floor=0.08, q=0.3):
    """Optimally-Modified Log-Spectral-Amplitude (Cohen 2003): LSA gain weighted by SPP.
    Replaces MMSE-LSA/Wiener. Noise-only bins collapse to gain_floor instead of being shaped into
    fake harmonics -> far fewer STT hallucinations on normal-volume / low-SNR clips.

    q = a priori prob of speech ABSENCE (higher q -> more aggressive noise suppression)."""
    from scipy.signal import stft, istft
    from scipy.special import exp1
    signal = np.asarray(signal, float)
    if len(signal) < nperseg:
        return signal.astype(np.float32)
    _, _, Z = stft(signal, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap,
                   boundary="zeros", padded=True)
    P = np.abs(Z) ** 2
    noise, _ = _mcra_noise_psd(P)
    nb, nf = Z.shape
    out = np.zeros_like(Z)
    Gmin = gain_floor
    gp = np.ones(nb)
    Yp = Z[:, 0]
    for l in range(nf):
        Y = Z[:, l]
        nd = noise[:, l] + 1e-12
        gamma = np.abs(Y) ** 2 / nd                                   # a posteriori SNR
        xi = (np.maximum(gamma - 1, 0) if l == 0
              else alpha * np.abs(gp * Yp) ** 2 / nd + (1 - alpha) * np.maximum(gamma - 1, 0))
        xi = np.maximum(xi, 1e-3)                                     # a priori SNR (decision-directed)
        v = np.maximum(xi * gamma / (1 + xi), 1e-10)
        GH1 = xi / (1 + xi) * np.exp(0.5 * exp1(v))                   # LSA gain (speech present)
        ratio = q / (1.0 - q)
        spp = 1.0 / (1.0 + ratio * (1 + xi) * np.exp(-v))             # speech-presence probability
        G = np.clip(GH1 ** spp * Gmin ** (1.0 - spp), Gmin, 1.0)      # OM-LSA gain
        out[:, l] = G * Y
        gp, Yp = G, Y
    _, y = istft(out, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, boundary=True)
    n = len(signal)
    return (y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))).astype(np.float32)


# --- legacy denoisers kept for A/B comparison -------------------------------------------------
def _min_stats_noise_psd(P, frames=50, alpha=0.8, bias=1.66):
    from scipy.ndimage import minimum_filter1d
    sm = np.zeros_like(P)
    sm[:, 0] = P[:, 0]
    for l in range(1, P.shape[1]):
        sm[:, l] = alpha * sm[:, l - 1] + (1.0 - alpha) * P[:, l]
    return bias * minimum_filter1d(sm, size=frames, axis=1)


def denoise_wiener(signal, fs, alpha=0.98, nperseg=256, noverlap=192, gain_floor=0.1):
    from scipy.signal import stft, istft
    signal = np.asarray(signal, float)
    if len(signal) < nperseg:
        return signal.astype(np.float32)
    _, _, Z = stft(signal, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap,
                   boundary="zeros", padded=True)
    nb, nf = Z.shape
    noise = _min_stats_noise_psd(np.abs(Z) ** 2)
    out = np.zeros_like(Z)
    gp = np.ones(nb)
    Yp = Z[:, 0]
    for l in range(nf):
        Y = Z[:, l]
        nd = noise[:, l] + 1e-12
        gamma = np.abs(Y) ** 2 / nd
        xi = (np.maximum(gamma - 1, 0) if l == 0
              else alpha * np.abs(gp * Yp) ** 2 / nd + (1 - alpha) * np.maximum(gamma - 1, 0))
        xi = np.maximum(xi, 1e-4)
        G = np.clip(xi / (1 + xi), gain_floor, 1.0)
        out[:, l] = G * Y
        gp, Yp = G, Y
    _, y = istft(out, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, boundary=True)
    n = len(signal)
    return (y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))).astype(np.float32)


DENOISERS = {
    "none": lambda s, fs: np.asarray(s, np.float32),
    "omlsa": denoise_omlsa,
    "wiener": denoise_wiener,
}


# ============================================================ VAD gate + SNR-aware normalize
def frame_rms(x, sr, frame_ms=30, hop_ms=10):
    nf = max(1, int(sr * frame_ms / 1000))
    nh = max(1, int(sr * hop_ms / 1000))
    if len(x) <= nf:
        return np.array([np.sqrt((np.asarray(x, float) ** 2).mean() + 1e-12)]), np.array([0]), nf, nh
    starts = np.arange(0, len(x) - nf + 1, nh)
    rms = np.sqrt(np.array([(x[s:s + nf] ** 2).mean() for s in starts]) + 1e-12)
    return rms, starts, nf, nh


def estimate_snr_db(x, sr, frame_ms=30, hop_ms=10, noise_pct=20):
    """Estimate speech-band SNR (dB): active-frame RMS vs. noise-floor RMS (low-energy frames).
    Used both to gate normalization and to decide whether to send a clip to the STT."""
    rms, _, _, _ = frame_rms(x, sr, frame_ms, hop_ms)
    if rms.size < 3:
        return 0.0
    noise = np.percentile(rms, noise_pct) + 1e-9
    active = np.percentile(rms, 90)
    return float(20.0 * np.log10(active / noise))


def vad_gate(x, sr, frame_ms=30, hop_ms=10, thresh_ratio=2.0, hangover=5, attn_db=-40.0):
    """Zero out (attenuate) frames whose RMS is below thresh_ratio * noise-floor, with hangover so
    word tails are not clipped. This is THE fix for STT inventing words on shaped noise: the
    recognizer receives real silence on non-speech frames instead of sculpted noise.

    Returns the gated signal (overlap-add of a smooth per-sample mask)."""
    x = np.asarray(x, float)
    rms, starts, nf, nh = frame_rms(x, sr, frame_ms, hop_ms)
    if rms.size < 3:
        return x.astype(np.float32)
    noise = np.percentile(rms, 20) + 1e-9
    active = rms > (thresh_ratio * noise)
    # hangover: keep `hangover` frames after each active frame
    held = active.copy()
    count = 0
    for i in range(len(active)):
        if active[i]:
            count = hangover
        elif count > 0:
            held[i] = True
            count -= 1
    floor = 10.0 ** (attn_db / 20.0)                   # residual gain for gated frames
    mask = np.full(len(x), floor, dtype=float)
    wsum = np.zeros(len(x), dtype=float)
    win = np.hanning(nf) + 1e-6
    gsig = np.zeros(len(x), dtype=float)
    for k, s in enumerate(starts):
        g = 1.0 if held[k] else floor
        seg = slice(s, s + nf)
        gsig[seg] += x[seg] * g * win
        wsum[seg] += win
    wsum[wsum == 0] = 1.0
    out = gsig / wsum
    # samples past the last frame: gate by last decision
    tail = starts[-1] + nf
    if tail < len(x):
        out[tail:] = x[tail:] * (1.0 if held[-1] else floor)
    return out.astype(np.float32)


def norm_rms(x, target=0.1):
    x = np.asarray(x, float)
    r = np.sqrt(np.mean(x ** 2)) + 1e-9
    return (x * (target / r)).astype(np.float32)


def norm_rms_active(x, sr, target=0.1, frame_ms=30, hop_ms=10):
    """RMS-normalize using ONLY speech-active frames, so a long silence does not inflate the gain
    and blow up the (now-gated) noise floor."""
    x = np.asarray(x, float)
    rms, starts, nf, _ = frame_rms(x, sr, frame_ms, hop_ms)
    if rms.size < 3:
        return norm_rms(x, target)
    thr = np.percentile(rms, 60)
    act = rms >= thr
    if not act.any():
        return norm_rms(x, target)
    seg_rms = rms[act]
    ref = np.sqrt(np.mean(seg_rms ** 2)) + 1e-9
    return (x * (target / ref)).astype(np.float32)


def norm_peak(x):
    x = np.asarray(x, float)
    p = np.max(np.abs(x))
    return (x / p).astype(np.float32) if p > 0 else x.astype(np.float32)


def trim_silence(x, sr, frame_ms=30, hop_ms=10, energy_ratio=0.1, pad_ms=150):
    x = np.asarray(x, np.float32)
    nf, nh = max(1, int(sr * frame_ms / 1000)), max(1, int(sr * hop_ms / 1000))
    if len(x) <= nf:
        return x
    starts = np.arange(0, len(x) - nf + 1, nh)
    rms = np.sqrt(np.array([(x[s:s + nf] ** 2).mean() for s in starts]) + 1e-12)
    if rms.max() <= 0:
        return x
    act = np.where(rms > rms.max() * energy_ratio)[0]
    if not act.size:
        return x
    pad = int(sr * pad_ms / 1000)
    return x[max(0, int(starts[act[0]]) - pad): min(len(x), int(starts[act[-1]]) + nf + pad)]


def write_wav(path, x, sr):
    x = np.asarray(x, np.float32)
    p = np.max(np.abs(x)) if x.size else 0.0
    if p > 1.0:
        x = x / p
    pcm = np.clip(x * 32767, -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())


# ============================================================ end-to-end pipeline
def reconstruct_multi(t_ms, axes, in_rate, gap_mode="uniform", combine="snr",
                      low=80.0, high=1600.0, denoise="omlsa", norm="rms",
                      vad=True, snr_norm=True, min_snr_db=2.0, out_rate=8000,
                      trim=True, gap_mult=3.0):
    """Full multi-axis IMU -> enhanced audio. Returns (signal float32, out_rate, snr_db).

    combine: 'snr' (SNR-weighted) | 'pca' (band-passed PCA) | 'mag' | int column index.
    If snr_norm=True and the estimated SNR < min_snr_db, the clip is returned as SILENCE
    (so it is never amplified into hallucination fodder)."""
    t = np.asarray(t_ms, float)
    A = np.asarray(axes, float)

    if gap_mode == "chunk":
        t, A = largest_contiguous_chunk(t, A, gap_mult)
        g = resample_uniform(t, A, in_rate)
    elif gap_mode == "uniform":
        g = A.astype(np.float32)
    else:                                              # "all"
        g = resample_uniform(t, A, in_rate)

    # combine axes (band-pass happens INSIDE the combiner, per axis)
    if g.ndim == 1:
        sig = apply_bandpass(g - np.mean(g), in_rate, low, high)
    elif combine == "snr":
        sig = combine_axes_snr(g, in_rate, low, high)
    elif combine == "pca":
        sig = combine_axes_pca(g, in_rate, low, high, prefilter=True)
    elif combine == "mag":
        m = np.sqrt((g ** 2).sum(axis=1))
        sig = apply_bandpass(m - m.mean(), in_rate, low, high)
    elif isinstance(combine, int):
        sig = apply_bandpass(g[:, combine] - g[:, combine].mean(), in_rate, low, high)
    else:
        raise SystemExit(f"unknown combine {combine!r}")

    sig = DENOISERS[denoise](sig, in_rate)             # denoise

    # Estimate SNR BEFORE gating: VAD attenuates non-speech frames, which would otherwise
    # collapse the noise-floor estimate and make every clip look high-SNR.
    snr_db = estimate_snr_db(sig, in_rate)

    if vad:
        sig = vad_gate(sig, in_rate)                   # gate non-speech to silence

    if snr_norm and snr_db < min_snr_db:               # refuse to amplify junk
        sig = np.zeros_like(sig)
    elif norm == "rms":
        sig = norm_rms_active(sig, in_rate) if snr_norm else norm_rms(sig)
    elif norm == "peak":
        sig = norm_peak(sig)

    if trim and np.any(sig):
        sig = trim_silence(sig, in_rate)
    if out_rate and out_rate != in_rate:
        sig = resample_poly_to(sig, in_rate, out_rate)
    return sig, (out_rate or in_rate), snr_db


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("-o", "--out", help="output WAV path (default: auto-named)")
    ap.add_argument("--axis", default="snr",
                    help='multi-axis combine: snr | pca | mag, OR a single column name (e.g. "Accel Z")')
    ap.add_argument("--ts-col", default="timestamp")
    ap.add_argument("--rate", type=int, default=0, help="working rate Hz; 0 = infer from timestamps")
    ap.add_argument("--gap-mode", choices=["chunk", "uniform", "all"], default="uniform")
    ap.add_argument("--low", type=float, default=80.0)
    ap.add_argument("--high", type=float, default=1600.0, help="capped at Nyquist (~1666 Hz @ 3333)")
    ap.add_argument("--denoise", choices=list(DENOISERS), default="omlsa")
    ap.add_argument("--norm", choices=["none", "peak", "rms"], default="rms")
    ap.add_argument("--vad-gate", dest="vad", action="store_true", default=True,
                    help="gate non-speech frames to silence (default ON)")
    ap.add_argument("--no-vad-gate", dest="vad", action="store_false")
    ap.add_argument("--snr-norm", dest="snr_norm", action="store_true", default=True,
                    help="emit silence when SNR < --min-snr-db, normalize on active frames (default ON)")
    ap.add_argument("--no-snr-norm", dest="snr_norm", action="store_false")
    ap.add_argument("--min-snr-db", type=float, default=2.0)
    ap.add_argument("--no-trim", action="store_true")
    ap.add_argument("--out-rate", type=int, default=8000, help="0 = keep working rate")
    a = ap.parse_args()

    # load: combine modes need all 3 axes; a column name needs just that column
    if a.axis in ("snr", "pca", "mag"):
        t_ms, axes = load_axes(a.csv, a.ts_col)
        combine = a.axis
    else:
        t_ms, axes = load_axis(a.csv, a.axis, a.ts_col)
        combine = 0 if axes.ndim == 1 else a.axis

    in_rate = a.rate or infer_rate(t_ms)
    if axes.ndim == 1:
        axes = axes.reshape(-1)                        # keep 1-D path

    sig, sr, snr = reconstruct_multi(
        t_ms, axes, in_rate, a.gap_mode, combine, a.low, a.high, a.denoise,
        a.norm, a.vad, a.snr_norm, a.min_snr_db, a.out_rate, not a.no_trim)

    stem = os.path.splitext(os.path.basename(a.csv))[0]
    out = a.out or f"{stem}__{a.axis}_{a.gap_mode}_{a.denoise}_vad{int(a.vad)}@{sr}.wav"
    write_wav(out, sig, sr)
    state = "SILENCED (low SNR)" if not np.any(sig) else "ok"
    print(f"axis/combine={a.axis} in_rate={in_rate}Hz gap={a.gap_mode} "
          f"band={a.low:.0f}-{a.high:.0f} denoise={a.denoise} vad={a.vad} snr_norm={a.snr_norm} "
          f"est_SNR={snr:.1f}dB -> {out}  ({len(sig) / sr:.2f}s @ {sr}Hz) [{state}]")


if __name__ == "__main__":
    main()
