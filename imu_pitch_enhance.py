#!/usr/bin/env python3
"""
Pitch Enhancement for Bone Conduction IMU Signals
==================================================
Implements multiple pitch enhancement / speaker volume enhancement techniques
for bone conduction IMU accelerometer data captured at 3.3 kHz.

Techniques:
  1. PSOLA (Pitch-Synchronous Overlap-Add) - shift pitch up
  2. Harmonic Regeneration - synthesize missing harmonics H2-H12
  3. Cepstral Pitch Sharpening - boost harmonic structure
  4. Pitch-Adaptive Comb Filter - suppress inter-harmonic noise
  5. Excitation Regeneration (LPC-based) - clean pitch pulse excitation
  6. WORLD Vocoder-style Resynthesis - full control resynthesis
  7. Combined Pipeline - best combination of above

Usage:
  python imu_pitch_enhance.py recording.csv
  python imu_pitch_enhance.py --input recording.csv --pitch-shift 1.2
  python imu_pitch_enhance.py recording.csv --f0-range 60 300 --output-dir results/
"""

import argparse
import os
import sys
import warnings
from math import gcd

import numpy as np
from scipy.signal import butter, sosfilt, resample_poly, lfilter, firwin
from scipy.special import exp1

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ORIGINAL_SR = 3300
TARGET_SR = 16000
BANDPASS_LOW = 50
BANDPASS_HIGH = 1500
BUTTER_ORDER = 4


# ---------------------------------------------------------------------------
# Embedded sample data generation for standalone testing
# ---------------------------------------------------------------------------

def generate_sample_csv(path):
    """
    Generate a synthetic IMU CSV file simulating bone conduction speech.

    Creates ~1 second of data at 3.3 kHz with a 120 Hz fundamental
    (male voice) plus harmonics, mimicking bone conduction characteristics.
    """
    sr = 3300
    duration = 1.0  # seconds
    n_samples = int(sr * duration)
    t = np.arange(n_samples) / sr
    timestamps_ms = t * 1000.0

    # Simulate bone conduction: F0=120Hz + harmonics (attenuated above 800Hz)
    f0 = 120.0
    signal_z = np.zeros(n_samples)
    for h in range(1, 8):
        freq = f0 * h
        # Bone conduction attenuates higher frequencies
        amp = 0.3 / (h ** 1.2)
        phase = np.random.uniform(0, 2 * np.pi)
        signal_z += amp * np.sin(2 * np.pi * freq * t + phase)

    # Add some noise (typical IMU noise floor)
    noise = np.random.randn(n_samples) * 0.02
    signal_z += noise

    # Amplitude modulation (speech envelope)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
    signal_z *= envelope

    # X and Y axes (weaker coupling)
    signal_x = signal_z * 0.1 + np.random.randn(n_samples) * 0.01
    signal_y = signal_z * 0.05 + np.random.randn(n_samples) * 0.01

    with open(path, "w") as f:
        f.write("timestamp,Accel X,Accel Y,Accel Z\n")
        for i in range(n_samples):
            f.write(f"{timestamps_ms[i]:.3f},{signal_x[i]:.6f},"
                    f"{signal_y[i]:.6f},{signal_z[i]:.6f}\n")

    return path


# ===========================================================================
# SECTION 1: CSV Loading & Gap-Aware Resampling
# ===========================================================================

def load_imu_csv(path, axis="z"):
    """Load IMU CSV: timestamp, Accel X, Accel Y, Accel Z."""
    import csv
    timestamps = []
    values = []
    axis_idx = {"x": 1, "y": 2, "z": 3}.get(axis.lower(), 3)

    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        header_lower = [h.strip().lower() for h in header]
        for i, h in enumerate(header_lower):
            if f"accel {axis.lower()}" in h or f"accel_{axis.lower()}" in h:
                axis_idx = i
                break

        for row in reader:
            if len(row) > axis_idx:
                try:
                    timestamps.append(float(row[0].strip()))
                    values.append(float(row[axis_idx].strip()))
                except (ValueError, IndexError):
                    continue

    return np.array(timestamps, dtype=np.float64), np.array(values, dtype=np.float64)


def gap_aware_resample(timestamps, signal, mode="uniform"):
    """Handle non-uniform/burst IMU timestamps."""
    diffs = np.diff(timestamps)
    if len(diffs) == 0:
        return signal, ORIGINAL_SR

    median_dt = np.median(diffs)
    inferred_sr = 1000.0 / median_dt

    if mode == "all":
        return signal, inferred_sr
    elif mode == "chunk":
        gap_threshold = 2.0 * median_dt
        gap_indices = np.where(diffs > gap_threshold)[0]
        boundaries = np.concatenate([[0], gap_indices + 1, [len(signal)]])
        chunk_lengths = np.diff(boundaries)
        largest_idx = np.argmax(chunk_lengths)
        start = boundaries[largest_idx]
        end = boundaries[largest_idx + 1]
        chunk_signal = signal[start:end]
        chunk_ts = timestamps[start:end]
        if len(chunk_ts) > 1:
            chunk_dt = np.median(np.diff(chunk_ts))
            inferred_sr = 1000.0 / chunk_dt
        duration_ms = chunk_ts[-1] - chunk_ts[0]
        n_samples = int(np.round(duration_ms * inferred_sr / 1000.0))
        if n_samples < 2:
            return chunk_signal, inferred_sr
        uniform_ts = np.linspace(chunk_ts[0], chunk_ts[-1], n_samples)
        resampled = np.interp(uniform_ts, chunk_ts, chunk_signal)
        return resampled, inferred_sr
    else:  # uniform
        duration_ms = timestamps[-1] - timestamps[0]
        n_samples = int(np.round(duration_ms * inferred_sr / 1000.0))
        if n_samples < 2:
            return signal, inferred_sr
        uniform_ts = np.linspace(timestamps[0], timestamps[-1], n_samples)
        resampled = np.interp(uniform_ts, timestamps, signal)
        return resampled, inferred_sr


# ===========================================================================
# SECTION 2: Preprocessing utilities (bandpass, AGC, MMSE-LSA)
# ===========================================================================

def apply_bandpass(signal, fs, low=BANDPASS_LOW, high=BANDPASS_HIGH, order=BUTTER_ORDER):
    """Butterworth bandpass filter."""
    nyquist = fs / 2.0
    high_norm = min(high, nyquist * 0.99)
    low_norm = max(low, 1.0)
    if low_norm >= high_norm:
        return signal
    sos = butter(order, [low_norm / nyquist, high_norm / nyquist], btype="band", output="sos")
    return sosfilt(sos, signal)


def agc(signal, fs, target_rms=0.1, frame_ms=50, max_gain_db=40):
    """Adaptive Gain Control - segment-by-segment energy-based gain."""
    frame_len = max(int(frame_ms * fs / 1000.0), 1)
    max_gain = 10.0 ** (max_gain_db / 20.0)
    output = np.zeros_like(signal)
    n = len(signal)

    for start in range(0, n, frame_len):
        end = min(start + frame_len, n)
        frame = signal[start:end]
        rms = np.sqrt(np.mean(frame ** 2) + 1e-10)
        gain = min(target_rms / rms, max_gain)
        output[start:end] = frame * gain

    return output


def mmse_lsa_denoise(signal, fs, frame_ms=25, hop_ms=10,
                     noise_frames=10, alpha_dd=0.98, floor_db=-30):
    """MMSE Log-Spectral Amplitude estimator (Ephraim & Malah, 1985)."""
    frame_len = int(np.round(frame_ms * fs / 1000.0))
    hop_len = int(np.round(hop_ms * fs / 1000.0))
    if frame_len < 4:
        frame_len = 4
    if hop_len < 1:
        hop_len = 1

    win = np.hanning(frame_len)
    win_sq = win ** 2
    n_orig = len(signal)
    n_frames = max(1, 1 + (n_orig - frame_len) // hop_len)
    pad_len = max(n_orig, (n_frames - 1) * hop_len + frame_len)
    n_frames = max(1, 1 + (pad_len - frame_len) // hop_len)
    x = np.zeros(pad_len)
    x[:n_orig] = signal

    nfft = frame_len
    freq_bins = nfft // 2 + 1

    X = np.zeros((n_frames, freq_bins), dtype=np.complex128)
    for i in range(n_frames):
        start = i * hop_len
        frame = x[start:start + frame_len] * win
        X[i, :] = np.fft.rfft(frame, n=nfft)

    P = np.abs(X) ** 2
    n_noise = min(noise_frames, n_frames)
    noise_psd = np.mean(P[:n_noise, :], axis=0) + 1e-10
    G_floor = 10.0 ** (floor_db / 20.0)

    enhanced_X = np.zeros_like(X)
    prev_gain = np.ones(freq_bins)
    prev_P = P[0, :] if n_frames > 0 else np.ones(freq_bins)

    for i in range(n_frames):
        gamma = P[i, :] / (noise_psd + 1e-10)
        gamma = np.maximum(gamma, 1e-3)
        if i == 0:
            xi = np.maximum(gamma - 1.0, 0.0)
        else:
            xi_ml = np.maximum(gamma - 1.0, 0.0)
            xi_dd = (prev_gain ** 2) * prev_P / (noise_psd + 1e-10)
            xi = alpha_dd * xi_dd + (1.0 - alpha_dd) * xi_ml
        xi = np.maximum(xi, 1e-3)
        v = xi * gamma / (1.0 + xi)
        v = np.maximum(v, 1e-10)
        e1_v = exp1(v)
        gain = (xi / (1.0 + xi)) * np.exp(0.5 * e1_v)
        gain = np.real(gain)
        gain = np.maximum(gain, G_floor)
        gain = np.minimum(gain, 1.0)
        enhanced_X[i, :] = gain * X[i, :]
        prev_gain = gain
        prev_P = P[i, :]

    output = np.zeros(pad_len)
    win_sum = np.zeros(pad_len)
    for i in range(n_frames):
        start = i * hop_len
        frame = np.fft.irfft(enhanced_X[i, :], n=nfft)
        output[start:start + frame_len] += frame * win
        win_sum[start:start + frame_len] += win_sq
    win_sum = np.maximum(win_sum, 1e-8)
    output = output / win_sum
    return output[:n_orig]


def upsample_to_target(signal, fs_in, fs_out=TARGET_SR):
    """Upsample using polyphase resampling."""
    fs_in_int = int(np.round(fs_in))
    g = gcd(fs_out, fs_in_int)
    up = fs_out // g
    down = fs_in_int // g
    return resample_poly(signal, up, down)


def normalize_signal(signal):
    """Normalize to [-1, 1]."""
    peak = np.max(np.abs(signal))
    if peak > 0:
        return signal / peak
    return signal


# ===========================================================================
# SECTION 3: F0 Detection (Autocorrelation-based)
# ===========================================================================

def detect_f0_autocorr(signal, fs, f0_min=60, f0_max=300, frame_ms=30, hop_ms=10):
    """
    Detect F0 frame-by-frame using autocorrelation.

    Returns
    -------
    f0_contour : ndarray
        F0 for each frame (0 if unvoiced).
    hop_samples : int
        Hop size in samples.
    """
    frame_len = int(frame_ms * fs / 1000.0)
    hop_len = int(hop_ms * fs / 1000.0)
    if frame_len < 4:
        frame_len = 4
    if hop_len < 1:
        hop_len = 1

    n_frames = max(1, 1 + (len(signal) - frame_len) // hop_len)

    lag_min = int(fs / f0_max)
    lag_max = int(fs / f0_min)
    lag_max = min(lag_max, frame_len - 1)

    f0_contour = np.zeros(n_frames)

    for i in range(n_frames):
        start = i * hop_len
        end = min(start + frame_len, len(signal))
        frame = signal[start:end]
        if len(frame) < 4:
            continue
        frame = frame - np.mean(frame)

        # Autocorrelation
        if np.max(np.abs(frame)) < 1e-10:
            continue

        # Normalized autocorrelation
        autocorr = np.correlate(frame, frame, mode='full')
        mid = len(frame) - 1
        autocorr = autocorr[mid:]  # positive lags only
        if len(autocorr) == 0 or autocorr[0] <= 0:
            continue
        autocorr = autocorr / autocorr[0]

        # Search for peak in valid lag range
        if lag_min >= lag_max or lag_max >= len(autocorr):
            continue

        search_region = autocorr[lag_min:lag_max + 1]
        if len(search_region) == 0:
            continue

        peak_idx = np.argmax(search_region)
        peak_val = search_region[peak_idx]

        # Voicing threshold
        if peak_val > 0.3:
            lag = lag_min + peak_idx
            if lag > 0:
                f0_contour[i] = fs / lag

    return f0_contour, hop_len


def get_median_f0(f0_contour):
    """Get median F0 from voiced frames."""
    voiced = f0_contour[f0_contour > 0]
    if len(voiced) > 0:
        return np.median(voiced)
    return 100.0  # default fallback


# ===========================================================================
# SECTION 4: Technique 1 - PSOLA (Pitch-Synchronous Overlap-Add)
# ===========================================================================

def find_pitch_marks(signal, fs, f0, f0_min=60, f0_max=300):
    """
    Find pitch marks (glottal closure instants) using autocorrelation.
    Places marks at pitch-period intervals aligned with signal peaks.
    """
    if f0 <= 0:
        f0 = 100.0
    period = int(np.round(fs / f0))
    marks = []

    # Place initial mark at first significant peak
    search_len = min(2 * period, len(signal))
    if search_len > 0:
        first_mark = np.argmax(np.abs(signal[:search_len]))
    else:
        first_mark = 0
    marks.append(first_mark)

    # Place subsequent marks at period intervals, refining with local peak
    pos = first_mark
    while pos + period < len(signal):
        nominal_next = pos + period
        # Search window around nominal position
        search_start = max(0, nominal_next - period // 4)
        search_end = min(len(signal), nominal_next + period // 4)
        if search_start >= search_end:
            break
        local_peak = search_start + np.argmax(np.abs(signal[search_start:search_end]))
        marks.append(local_peak)
        pos = local_peak

    return np.array(marks, dtype=int)


def psola_pitch_shift(signal, fs, f0, shift_factor=1.2):
    """
    TD-PSOLA pitch shifting.

    Shifts pitch UP by shift_factor (>1 = higher pitch).
    Works by shortening pitch periods in the overlap-add synthesis.

    Parameters
    ----------
    signal : ndarray
    fs : float
    f0 : float
        Detected fundamental frequency.
    shift_factor : float
        Factor to shift pitch (1.2 = 20% higher, 1.5 = 50% higher).

    Returns
    -------
    output : ndarray
        Pitch-shifted signal.
    """
    if f0 <= 0:
        f0 = 100.0

    period = int(np.round(fs / f0))
    new_period = int(np.round(period / shift_factor))

    if new_period < 2:
        new_period = 2

    # Find pitch marks
    marks = find_pitch_marks(signal, fs, f0)
    if len(marks) < 3:
        return signal.copy()

    # Estimate output length
    n_periods = len(marks) - 1
    out_len = int(n_periods * new_period) + 2 * period
    output = np.zeros(out_len)
    out_pos = 0

    for i in range(len(marks)):
        mark = marks[i]

        # Extract grain (one pitch period centered on mark)
        half_period = period // 2
        grain_start = max(0, mark - half_period)
        grain_end = min(len(signal), mark + half_period)
        grain = signal[grain_start:grain_end]

        # Apply Hanning window
        win = np.hanning(len(grain))
        grain_windowed = grain * win

        # Place grain in output at new period spacing
        place_start = out_pos - len(grain) // 2
        place_end = place_start + len(grain)

        if place_start < 0:
            grain_windowed = grain_windowed[-place_start:]
            place_start = 0
        if place_end > out_len:
            grain_windowed = grain_windowed[:out_len - place_start]
            place_end = out_len

        actual_len = min(len(grain_windowed), place_end - place_start)
        if actual_len > 0:
            output[place_start:place_start + actual_len] += grain_windowed[:actual_len]

        out_pos += new_period

    # Trim to valid length
    output = output[:out_pos + period]
    if len(output) == 0:
        return signal.copy()

    return output


# ===========================================================================
# SECTION 5: Technique 2 - Harmonic Regeneration
# ===========================================================================

def estimate_spectral_envelope(signal, fs, order=20):
    """Estimate spectral envelope using LPC (cepstral smoothing)."""
    # Compute LPC coefficients using autocorrelation method (Levinson-Durbin)
    n = len(signal)
    if n < order + 1:
        return np.ones(n // 2 + 1)

    # Autocorrelation
    r = np.correlate(signal, signal, mode='full')
    r = r[n - 1:n + order]

    # Levinson-Durbin
    a = np.zeros(order + 1)
    a[0] = 1.0
    err = r[0] + 1e-10

    for i in range(1, order + 1):
        # Compute reflection coefficient
        lam = 0.0
        for j in range(1, i):
            lam += a[j] * r[i - j]
        lam = -(r[i] + lam) / (err + 1e-10)

        # Update coefficients
        a_new = a.copy()
        for j in range(1, i):
            a_new[j] = a[j] + lam * a[i - j]
        a_new[i] = lam
        a = a_new

        err = err * (1.0 - lam * lam)
        if err <= 0:
            err = 1e-10
            break

    # Compute envelope from LPC spectrum
    nfft = max(512, n)
    freq_response = np.fft.rfft(a, n=nfft)
    envelope = 1.0 / (np.abs(freq_response) + 1e-10)

    return envelope


def harmonic_regeneration(signal, fs, f0, n_harmonics=12, mix_ratio=0.5):
    """
    Regenerate harmonics H2 through H12.

    Detects F0, synthesizes harmonics shaped by the spectral envelope,
    and mixes back with original.

    Parameters
    ----------
    signal : ndarray
    fs : float
    f0 : float
        Fundamental frequency.
    n_harmonics : int
        Number of harmonics to synthesize (2 through n_harmonics).
    mix_ratio : float
        How much synthesized harmonics to mix in (0-1).

    Returns
    -------
    output : ndarray
    """
    if f0 <= 0:
        f0 = 100.0

    n = len(signal)
    t = np.arange(n) / fs

    # Estimate spectral envelope for amplitude shaping
    envelope = estimate_spectral_envelope(signal, fs, order=20)
    nfft_env = (len(envelope) - 1) * 2

    # Fundamental amplitude (RMS of signal)
    fund_amp = np.sqrt(np.mean(signal ** 2) + 1e-10)

    # Synthesize harmonics
    harmonics = np.zeros(n)
    for h in range(2, n_harmonics + 1):
        freq_h = f0 * h
        if freq_h >= fs / 2:
            break

        # Get envelope value at this harmonic's frequency
        bin_idx = int(np.round(freq_h * nfft_env / fs))
        bin_idx = min(bin_idx, len(envelope) - 1)
        env_val = envelope[bin_idx]

        # Amplitude decays with harmonic number, shaped by envelope
        amp = fund_amp * env_val / (env_val.max() + 1e-10) * (1.0 / h)

        # Random phase for naturalness
        phase = np.random.uniform(0, 2 * np.pi)

        harmonics += amp * np.sin(2 * np.pi * freq_h * t + phase)

    # Mix
    output = (1.0 - mix_ratio) * signal + mix_ratio * harmonics

    return output


# ===========================================================================
# SECTION 6: Technique 3 - Cepstral Pitch Sharpening
# ===========================================================================

def cepstral_pitch_sharpening(signal, fs, f0, boost_factor=2.0,
                               frame_ms=30, hop_ms=10):
    """
    Sharpen harmonic structure via cepstral domain processing.

    Computes cepstrum, finds the pitch peak (rahmonic at quefrency = 1/F0),
    boosts it by a factor, and inverse transforms back.

    Parameters
    ----------
    signal : ndarray
    fs : float
    f0 : float
        Detected fundamental frequency.
    boost_factor : float
        How much to boost the pitch peak (2.0 = double).
    frame_ms : float
    hop_ms : float

    Returns
    -------
    output : ndarray
    """
    frame_len = int(frame_ms * fs / 1000.0)
    hop_len = int(hop_ms * fs / 1000.0)
    if frame_len < 4:
        frame_len = 4
    if hop_len < 1:
        hop_len = 1

    n_orig = len(signal)
    n_frames = max(1, 1 + (n_orig - frame_len) // hop_len)
    pad_len = max(n_orig, (n_frames - 1) * hop_len + frame_len)
    x = np.zeros(pad_len)
    x[:n_orig] = signal

    win = np.hanning(frame_len)
    win_sq = win ** 2
    output = np.zeros(pad_len)
    win_sum = np.zeros(pad_len)

    # Expected quefrency of pitch peak
    if f0 > 0:
        pitch_quefrency = int(np.round(fs / f0))
    else:
        pitch_quefrency = int(np.round(fs / 100.0))

    quef_window = max(3, pitch_quefrency // 4)

    for i in range(n_frames):
        start = i * hop_len
        frame = x[start:start + frame_len] * win

        # Compute real cepstrum
        spectrum = np.fft.rfft(frame, n=frame_len)
        log_mag = np.log(np.abs(spectrum) + 1e-10)
        cepstrum = np.fft.irfft(log_mag, n=frame_len)

        # Boost the pitch peak region (rahmonic)
        quef_start = max(1, pitch_quefrency - quef_window)
        quef_end = min(frame_len // 2, pitch_quefrency + quef_window + 1)
        cepstrum[quef_start:quef_end] *= boost_factor

        # Also boost integer multiples (secondary rahmonics)
        for mult in range(2, 4):
            q = pitch_quefrency * mult
            qs = max(1, q - quef_window // 2)
            qe = min(frame_len // 2, q + quef_window // 2 + 1)
            if qs < frame_len // 2:
                cepstrum[qs:qe] *= (boost_factor * 0.5)

        # Inverse cepstrum to get modified log spectrum
        modified_log_mag = np.fft.rfft(cepstrum, n=frame_len)
        modified_log_mag = np.real(modified_log_mag[:len(spectrum)])

        # Reconstruct with original phase
        phase = np.angle(spectrum)
        modified_spectrum = np.exp(modified_log_mag) * np.exp(1j * phase)

        # Inverse FFT
        frame_out = np.fft.irfft(modified_spectrum, n=frame_len)
        output[start:start + frame_len] += frame_out * win
        win_sum[start:start + frame_len] += win_sq

    win_sum = np.maximum(win_sum, 1e-8)
    output = output / win_sum
    return output[:n_orig]


# ===========================================================================
# SECTION 7: Technique 4 - Pitch-Adaptive Comb Filter
# ===========================================================================

def pitch_adaptive_comb_filter(signal, fs, f0_contour, hop_samples, q_factor=10.0):
    """
    Apply a pitch-adaptive comb filter that tracks F0 frame-by-frame.

    Creates a comb filter with teeth at each harmonic of the detected F0
    for each frame. Suppresses inter-harmonic noise.

    Parameters
    ----------
    signal : ndarray
    fs : float
    f0_contour : ndarray
        F0 for each frame.
    hop_samples : int
        Hop size in samples.
    q_factor : float
        Quality factor controlling bandwidth of each comb tooth.

    Returns
    -------
    output : ndarray
    """
    frame_len = int(30 * fs / 1000.0)  # 30ms frames
    if frame_len < 4:
        frame_len = 4

    n_orig = len(signal)
    n_frames = len(f0_contour)
    nfft = frame_len
    freq_bins = nfft // 2 + 1
    freqs = np.linspace(0, fs / 2, freq_bins)

    win = np.hanning(frame_len)
    win_sq = win ** 2

    pad_len = max(n_orig, (n_frames - 1) * hop_samples + frame_len)
    x = np.zeros(pad_len)
    x[:n_orig] = signal

    output = np.zeros(pad_len)
    win_sum = np.zeros(pad_len)

    for i in range(n_frames):
        start = i * hop_samples
        if start + frame_len > pad_len:
            break

        frame = x[start:start + frame_len] * win
        F = np.fft.rfft(frame, n=nfft)

        f0 = f0_contour[i]
        if f0 > 0:
            # Build comb filter response
            comb_gain = np.zeros(freq_bins)
            for h in range(1, 20):
                harmonic_freq = f0 * h
                if harmonic_freq >= fs / 2:
                    break
                # Gaussian-shaped peak at each harmonic
                bandwidth = harmonic_freq / q_factor
                comb_gain += np.exp(-0.5 * ((freqs - harmonic_freq) / (bandwidth + 1e-10)) ** 2)

            # Normalize and apply floor
            comb_gain = np.maximum(comb_gain, 0.05)
            comb_gain = np.minimum(comb_gain, 1.0)

            F_filtered = F * comb_gain
        else:
            # Unvoiced: pass through with slight attenuation
            F_filtered = F * 0.5

        frame_out = np.fft.irfft(F_filtered, n=nfft)
        output[start:start + frame_len] += frame_out * win
        win_sum[start:start + frame_len] += win_sq

    win_sum = np.maximum(win_sum, 1e-8)
    output = output / win_sum
    return output[:n_orig]


# ===========================================================================
# SECTION 8: Technique 5 - Excitation Regeneration (LPC-based)
# ===========================================================================

def levinson_durbin(r, order):
    """Levinson-Durbin recursion for LPC coefficients."""
    a = np.zeros(order + 1)
    a[0] = 1.0
    err = r[0] + 1e-10

    for i in range(1, order + 1):
        lam = 0.0
        for j in range(1, i):
            lam += a[j] * r[i - j]
        lam = -(r[i] + lam) / (err + 1e-10)

        a_new = a.copy()
        for j in range(1, i):
            a_new[j] = a[j] + lam * a[i - j]
        a_new[i] = lam
        a = a_new

        err = err * (1.0 - lam * lam)
        if err <= 0:
            err = 1e-10
            break

    return a, err


def lpc_excitation_regeneration(signal, fs, f0, lpc_order=16, frame_ms=25, hop_ms=10):
    """
    LPC-based excitation regeneration.

    Computes LPC coefficients to model vocal tract, extracts noisy excitation,
    replaces it with a clean pitch pulse train at detected F0, then re-filters
    through the LPC vocal tract model.

    Parameters
    ----------
    signal : ndarray
    fs : float
    f0 : float
        Detected fundamental frequency.
    lpc_order : int
        LPC analysis order.
    frame_ms : float
    hop_ms : float

    Returns
    -------
    output : ndarray
    """
    frame_len = int(frame_ms * fs / 1000.0)
    hop_len = int(hop_ms * fs / 1000.0)
    if frame_len < lpc_order + 2:
        frame_len = lpc_order + 2
    if hop_len < 1:
        hop_len = 1

    n_orig = len(signal)
    n_frames = max(1, 1 + (n_orig - frame_len) // hop_len)
    pad_len = max(n_orig, (n_frames - 1) * hop_len + frame_len)
    x = np.zeros(pad_len)
    x[:n_orig] = signal

    win = np.hanning(frame_len)
    win_sq = win ** 2
    output = np.zeros(pad_len)
    win_sum = np.zeros(pad_len)

    if f0 <= 0:
        f0 = 100.0
    pitch_period = int(np.round(fs / f0))

    for i in range(n_frames):
        start = i * hop_len
        frame = x[start:start + frame_len]
        frame_windowed = frame * win

        # Compute autocorrelation
        r = np.correlate(frame_windowed, frame_windowed, mode='full')
        r = r[frame_len - 1:frame_len + lpc_order]

        if r[0] < 1e-10:
            output[start:start + frame_len] += frame * win
            win_sum[start:start + frame_len] += win_sq
            continue

        # Levinson-Durbin
        a, err = levinson_durbin(r, lpc_order)
        gain = np.sqrt(max(err, 1e-10))

        # Generate clean pitch pulse excitation
        excitation = np.zeros(frame_len)
        for p in range(0, frame_len, pitch_period):
            if p < frame_len:
                excitation[p] = gain

        # Synthesize through LPC filter (all-pole: 1/A(z))
        # y[n] = excitation[n] - sum(a[k]*y[n-k], k=1..order)
        synthesized = lfilter([1.0], a, excitation)

        output[start:start + frame_len] += synthesized * win
        win_sum[start:start + frame_len] += win_sq

    win_sum = np.maximum(win_sum, 1e-8)
    output = output / win_sum
    return output[:n_orig]


# ===========================================================================
# SECTION 9: Technique 6 - WORLD Vocoder-style Resynthesis
# ===========================================================================

def world_vocoder_resynth(signal, fs, f0_contour, hop_samples,
                          n_harmonics=20, aperiodicity_ratio=0.1):
    """
    Simplified WORLD vocoder-style resynthesis.

    Steps:
      1. Extract F0 contour (already provided)
      2. Extract spectral envelope (LPC per frame)
      3. Estimate aperiodicity ratio per frame
      4. Resynthesize: pulse train at F0 + noise, filtered through envelope

    Parameters
    ----------
    signal : ndarray
    fs : float
    f0_contour : ndarray
        F0 for each frame.
    hop_samples : int
    n_harmonics : int
        Max harmonics to synthesize.
    aperiodicity_ratio : float
        Base aperiodicity (noise) ratio.

    Returns
    -------
    output : ndarray
    """
    frame_len = int(30 * fs / 1000.0)
    if frame_len < 4:
        frame_len = 4

    n_orig = len(signal)
    n_frames = len(f0_contour)
    lpc_order = 16

    pad_len = max(n_orig, (n_frames - 1) * hop_samples + frame_len)
    x = np.zeros(pad_len)
    x[:n_orig] = signal

    output = np.zeros(pad_len)
    win = np.hanning(frame_len)
    win_sq = win ** 2
    win_sum = np.zeros(pad_len)

    # Phase accumulator for smooth synthesis
    phase_accum = 0.0

    for i in range(n_frames):
        start = i * hop_samples
        if start + frame_len > pad_len:
            break

        frame = x[start:start + frame_len] * win
        f0 = f0_contour[i]

        # Estimate spectral envelope via LPC
        r = np.correlate(frame, frame, mode='full')
        r = r[frame_len - 1:frame_len + lpc_order]

        if r[0] < 1e-10:
            win_sum[start:start + frame_len] += win_sq
            continue

        a, err = levinson_durbin(r, lpc_order)
        gain = np.sqrt(max(err, 1e-10))

        # Estimate aperiodicity from HNR of original frame
        frame_power = np.mean(frame ** 2)
        if f0 > 0 and frame_power > 1e-10:
            # Simple HNR estimate
            period = int(np.round(fs / f0))
            if period > 0 and period < frame_len:
                n_periods = frame_len // period
                if n_periods >= 2:
                    periods_matrix = []
                    for p in range(n_periods):
                        p_start = p * period
                        if p_start + period <= frame_len:
                            periods_matrix.append(frame[p_start:p_start + period])
                    if len(periods_matrix) >= 2:
                        periods_arr = np.array(periods_matrix)
                        mean_period = np.mean(periods_arr, axis=0)
                        harmonic_power = np.mean(mean_period ** 2)
                        noise_power = frame_power - harmonic_power
                        noise_power = max(noise_power, 1e-10)
                        hnr = harmonic_power / noise_power
                        ap_ratio = 1.0 / (1.0 + hnr)
                    else:
                        ap_ratio = aperiodicity_ratio
                else:
                    ap_ratio = aperiodicity_ratio
            else:
                ap_ratio = aperiodicity_ratio
        else:
            ap_ratio = 0.8  # unvoiced = mostly noise

        # Generate excitation
        excitation = np.zeros(frame_len)
        if f0 > 0:
            # Pulse train + noise
            t_frame = np.arange(frame_len) / fs
            pulse_component = np.zeros(frame_len)
            period = int(np.round(fs / f0))
            for p in range(0, frame_len, max(period, 1)):
                if p < frame_len:
                    pulse_component[p] = 1.0

            noise_component = np.random.randn(frame_len)
            excitation = ((1.0 - ap_ratio) * pulse_component +
                         ap_ratio * noise_component * 0.1) * gain
        else:
            # Unvoiced: noise excitation
            excitation = np.random.randn(frame_len) * gain * 0.3

        # Filter through vocal tract (LPC synthesis)
        synthesized = lfilter([1.0], a, excitation)

        output[start:start + frame_len] += synthesized * win
        win_sum[start:start + frame_len] += win_sq

    win_sum = np.maximum(win_sum, 1e-8)
    output = output / win_sum
    return output[:n_orig]


# ===========================================================================
# SECTION 10: Technique 7 - Combined Pipeline
# ===========================================================================

def combined_pipeline(signal, fs, f0, f0_contour, hop_samples, pitch_shift=1.2):
    """
    Combined pipeline: best combination of techniques.

    AGC -> Bandpass -> Cepstral Pitch Sharpening -> Harmonic Regeneration ->
    PSOLA (slight pitch shift up) -> Normalize

    Parameters
    ----------
    signal : ndarray
    fs : float
    f0 : float
        Median detected F0.
    f0_contour : ndarray
    hop_samples : int
    pitch_shift : float
        Pitch shift factor for PSOLA step.

    Returns
    -------
    output : ndarray
    """
    print("    [Combined] Step 1/5: AGC...")
    sig = agc(signal, fs, target_rms=0.15)

    print("    [Combined] Step 2/5: Bandpass filter...")
    sig = apply_bandpass(sig, fs, low=BANDPASS_LOW, high=BANDPASS_HIGH)

    print("    [Combined] Step 3/5: Cepstral pitch sharpening...")
    sig = cepstral_pitch_sharpening(sig, fs, f0, boost_factor=1.5)

    print("    [Combined] Step 4/5: Harmonic regeneration...")
    sig = harmonic_regeneration(sig, fs, f0, n_harmonics=8, mix_ratio=0.3)

    print("    [Combined] Step 5/5: PSOLA pitch shift...")
    if pitch_shift != 1.0:
        sig = psola_pitch_shift(sig, fs, f0, shift_factor=pitch_shift)

    # Normalize
    sig = normalize_signal(sig)
    return sig


# ===========================================================================
# SECTION 11: Main Pipeline Runner
# ===========================================================================

def save_wav(signal, fs, path):
    """Save signal as 16-bit WAV."""
    import soundfile as sf
    normed = normalize_signal(signal)
    sf.write(path, normed, fs, subtype="PCM_16")
    print(f"    -> Saved: {path}")


def run_all_enhancements(input_path, output_dir, axis="z", gap_mode="uniform",
                         pitch_shift=1.0, f0_min=60, f0_max=300):
    """
    Run ALL pitch enhancement techniques on the input CSV.

    Parameters
    ----------
    input_path : str
        Path to IMU CSV file.
    output_dir : str
        Directory for output WAV files.
    axis : str
        Accelerometer axis.
    gap_mode : str
        Gap handling mode.
    pitch_shift : float
        Pitch shift factor for PSOLA.
    f0_min, f0_max : float
        Expected F0 range.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  PITCH ENHANCEMENT FOR BONE CONDUCTION IMU")
    print("  (Speaker Volume Enhancement)")
    print("=" * 70)
    print(f"  Input:       {input_path}")
    print(f"  Output dir:  {output_dir}")
    print(f"  Axis:        {axis}")
    print(f"  Pitch shift: {pitch_shift}x")
    print(f"  F0 range:    {f0_min}-{f0_max} Hz")
    print("=" * 70)

    # --- Load and preprocess ---
    print("\n[1/9] Loading CSV...")
    timestamps, signal = load_imu_csv(input_path, axis=axis)
    print(f"    Loaded {len(signal)} samples")

    print("\n[2/9] Gap-aware resampling...")
    signal, inferred_sr = gap_aware_resample(timestamps, signal, mode=gap_mode)
    print(f"    Inferred SR: {inferred_sr:.1f} Hz, {len(signal)} samples")

    # Remove DC
    signal = signal - np.mean(signal)

    print("\n[3/9] Preprocessing (bandpass + AGC + MMSE-LSA)...")
    signal_bp = apply_bandpass(signal, inferred_sr)
    signal_agc = agc(signal_bp, inferred_sr, target_rms=0.1)
    signal_denoised = mmse_lsa_denoise(signal_agc, inferred_sr)

    # Detect F0
    print("\n[4/9] Detecting F0...")
    f0_contour, hop_samples = detect_f0_autocorr(
        signal_denoised, inferred_sr, f0_min=f0_min, f0_max=f0_max
    )
    median_f0 = get_median_f0(f0_contour)
    voiced_pct = 100.0 * np.sum(f0_contour > 0) / max(len(f0_contour), 1)
    print(f"    Detected F0: {median_f0:.1f} Hz (median)")
    print(f"    Voiced frames: {voiced_pct:.1f}%")

    # Use preprocessed signal for all techniques
    sig = signal_denoised

    # --- Run each technique ---
    results = {}

    # Technique 1: PSOLA
    print("\n[5/9] Technique 1: PSOLA Pitch Shift...")
    shift = pitch_shift if pitch_shift != 1.0 else 1.2
    psola_out = psola_pitch_shift(sig, inferred_sr, median_f0, shift_factor=shift)
    psola_16k = upsample_to_target(psola_out, inferred_sr, TARGET_SR)
    save_wav(psola_16k, TARGET_SR, os.path.join(output_dir, "01_psola_pitch_shift.wav"))
    results["PSOLA"] = psola_16k

    # Technique 2: Harmonic Regeneration
    print("\n[6/9] Technique 2: Harmonic Regeneration...")
    harm_out = harmonic_regeneration(sig, inferred_sr, median_f0, n_harmonics=12, mix_ratio=0.5)
    harm_16k = upsample_to_target(harm_out, inferred_sr, TARGET_SR)
    save_wav(harm_16k, TARGET_SR, os.path.join(output_dir, "02_harmonic_regeneration.wav"))
    results["Harmonic Regeneration"] = harm_16k

    # Technique 3: Cepstral Pitch Sharpening
    print("\n[7/9] Technique 3: Cepstral Pitch Sharpening...")
    ceps_out = cepstral_pitch_sharpening(sig, inferred_sr, median_f0, boost_factor=2.0)
    ceps_16k = upsample_to_target(ceps_out, inferred_sr, TARGET_SR)
    save_wav(ceps_16k, TARGET_SR, os.path.join(output_dir, "03_cepstral_sharpening.wav"))
    results["Cepstral Sharpening"] = ceps_16k

    # Technique 4: Pitch-Adaptive Comb Filter
    print("\n[8/9] Technique 4: Pitch-Adaptive Comb Filter...")
    comb_out = pitch_adaptive_comb_filter(sig, inferred_sr, f0_contour, hop_samples, q_factor=10.0)
    comb_16k = upsample_to_target(comb_out, inferred_sr, TARGET_SR)
    save_wav(comb_16k, TARGET_SR, os.path.join(output_dir, "04_comb_filter.wav"))
    results["Comb Filter"] = comb_16k

    # Technique 5: Excitation Regeneration (LPC)
    print("\n[9/9] Technique 5: Excitation Regeneration (LPC)...")
    lpc_out = lpc_excitation_regeneration(sig, inferred_sr, median_f0, lpc_order=16)
    lpc_16k = upsample_to_target(lpc_out, inferred_sr, TARGET_SR)
    save_wav(lpc_16k, TARGET_SR, os.path.join(output_dir, "05_lpc_excitation.wav"))
    results["LPC Excitation"] = lpc_16k

    # Technique 6: WORLD Vocoder Resynthesis
    print("\n[10/9] Technique 6: WORLD Vocoder Resynthesis...")
    world_out = world_vocoder_resynth(sig, inferred_sr, f0_contour, hop_samples,
                                       n_harmonics=20, aperiodicity_ratio=0.1)
    world_16k = upsample_to_target(world_out, inferred_sr, TARGET_SR)
    save_wav(world_16k, TARGET_SR, os.path.join(output_dir, "06_world_vocoder.wav"))
    results["WORLD Vocoder"] = world_16k

    # Technique 7: Combined Pipeline
    print("\n[11/9] Technique 7: Combined Pipeline...")
    combined_out = combined_pipeline(sig, inferred_sr, median_f0, f0_contour,
                                     hop_samples, pitch_shift=shift)
    combined_16k = upsample_to_target(combined_out, inferred_sr, TARGET_SR)
    save_wav(combined_16k, TARGET_SR, os.path.join(output_dir, "07_combined_pipeline.wav"))
    results["Combined Pipeline"] = combined_16k

    # Also save the preprocessed (denoised) version for comparison
    baseline_16k = upsample_to_target(sig, inferred_sr, TARGET_SR)
    save_wav(baseline_16k, TARGET_SR, os.path.join(output_dir, "00_baseline_denoised.wav"))

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Detected F0: {median_f0:.1f} Hz")
    print(f"  Voiced frames: {voiced_pct:.1f}%")
    print(f"  Pitch shift applied: {shift}x")
    print()
    print("  Output files:")
    print(f"    00_baseline_denoised.wav    - Preprocessed baseline")
    print(f"    01_psola_pitch_shift.wav    - PSOLA ({shift}x pitch up)")
    print(f"    02_harmonic_regeneration.wav - H2-H12 synthesis")
    print(f"    03_cepstral_sharpening.wav  - Cepstral boost")
    print(f"    04_comb_filter.wav          - Pitch-adaptive comb")
    print(f"    05_lpc_excitation.wav       - LPC pulse train excitation")
    print(f"    06_world_vocoder.wav        - Vocoder resynthesis")
    print(f"    07_combined_pipeline.wav    - Full combined pipeline")
    print()

    # Compute RMS for each to gauge volume enhancement
    print("  RMS levels (higher = louder / more enhanced):")
    for name, sig_out in results.items():
        rms = np.sqrt(np.mean(sig_out ** 2))
        print(f"    {name:25s}: RMS = {rms:.4f}")
    baseline_rms = np.sqrt(np.mean(baseline_16k ** 2))
    print(f"    {'Baseline':25s}: RMS = {baseline_rms:.4f}")

    print()
    print("  RECOMMENDATION:")
    print("    For ASR intelligibility: Combined Pipeline or PSOLA")
    print("    For speaker volume boost: Harmonic Regeneration + Cepstral Sharpening")
    print("    For clean reconstruction: LPC Excitation or WORLD Vocoder")
    print("=" * 70)


# ===========================================================================
# SECTION 12: CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pitch Enhancement for Bone Conduction IMU Signals (Speaker Volume Enhancement)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Techniques implemented:
  1. PSOLA (Pitch-Synchronous Overlap-Add) - pitch shift up
  2. Harmonic Regeneration - synthesize harmonics H2-H12
  3. Cepstral Pitch Sharpening - boost harmonic peaks
  4. Pitch-Adaptive Comb Filter - suppress inter-harmonic noise
  5. Excitation Regeneration (LPC) - clean pulse train excitation
  6. WORLD Vocoder Resynthesis - full control resynthesis
  7. Combined Pipeline - AGC + Bandpass + Cepstral + Harmonics + PSOLA

Examples:
  python imu_pitch_enhance.py recording.csv
  python imu_pitch_enhance.py --input recording.csv --pitch-shift 1.5
  python imu_pitch_enhance.py recording.csv --f0-range 80 200 --output-dir results/
        """,
    )

    parser.add_argument(
        "csv_input", nargs="?", default=None,
        help="Input CSV file (positional argument)")
    parser.add_argument(
        "--input", "-i", dest="input_flag", default=None,
        help="Input CSV file (alternative to positional arg)")
    parser.add_argument(
        "--pitch-shift", type=float, default=1.0,
        help="Factor to shift pitch up (default: 1.0 = no shift, try 1.2 or 1.5)")
    parser.add_argument(
        "--f0-range", nargs=2, type=float, default=[60, 300],
        metavar=("MIN", "MAX"),
        help="Expected F0 range in Hz (default: 60 300)")
    parser.add_argument(
        "--output-dir", default="pitch_enhanced/",
        help="Output directory (default: pitch_enhanced/)")
    parser.add_argument(
        "--axis", default="z", choices=["x", "y", "z"],
        help="Accelerometer axis (default: z)")
    parser.add_argument(
        "--gap-mode", default="uniform", choices=["chunk", "uniform", "all"],
        help="Gap handling mode (default: uniform)")

    args = parser.parse_args()

    # Resolve input path
    input_path = args.csv_input or args.input_flag

    if input_path is None:
        # Use embedded sample data
        print("No input file specified. Using generated sample data for demo.")
        import tempfile
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        tmp.close()
        generate_sample_csv(tmp.name)
        input_path = tmp.name
    elif not os.path.isfile(input_path):
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    run_all_enhancements(
        input_path=input_path,
        output_dir=args.output_dir,
        axis=args.axis,
        gap_mode=args.gap_mode,
        pitch_shift=args.pitch_shift,
        f0_min=args.f0_range[0],
        f0_max=args.f0_range[1],
    )


if __name__ == "__main__":
    main()
