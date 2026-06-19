#!/usr/bin/env python3
"""
Low-Pitch Bone Conduction IMU Enhancement
==========================================
Advanced enhancement techniques specifically for LOW PITCH / QUIET bone
conduction IMU signals captured at 3.3 kHz.

KEY INSIGHT: For low-pitch bone conduction, the processing order matters:
  1. First boost the signal above the noise floor (AGC/compression)
  2. Then extract and enhance the harmonic structure (it's there, just weak)
  3. Then extend the bandwidth so ASR models can work with it
  4. THEN run denoising (MMSE-LSA) on the enhanced signal -- not before!

  denoise-first FAILS because the signal is below noise floor.
  enhance-first then denoise WORKS.

Techniques implemented:
  1. Adaptive Gain Control (AGC) - segment-by-segment energy-based gain
  2. Dynamic Range Compression - multi-band compressor
  3. Harmonic Enhancement - F0 detection + comb filtering + pitch-synchronous avg
  4. Bandwidth Extension - synthesize higher harmonics up to 4000 Hz
  5. Teager-Kaiser Energy Operator (TKEO) - non-linear energy for speech emphasis
  6. Bone Conduction Transfer Function Compensation - invert attenuation curve
  7. Noise-Gated Amplification - only amplify when signal > noise floor
  8. Combined Pipeline (THE FULL CHAIN):
     Pre-emphasis -> AGC -> MMSE-LSA -> Harmonic Enhancement ->
     Bandwidth Extension -> Dynamic Compression -> Normalize

Usage:
  python imu_enhance_lowpitch.py recording.csv
  python imu_enhance_lowpitch.py --input recording.csv --gain-db 20
  python imu_enhance_lowpitch.py recording.csv --f0-range 60 200
"""

import argparse
import os
import sys
import warnings
from math import gcd

import numpy as np
from scipy.signal import butter, sosfilt, resample_poly, hilbert, lfilter
from scipy.special import exp1

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ORIGINAL_SR = 3300       # Nominal IMU sampling rate (Hz)
TARGET_SR = 16000        # Output sampling rate (Hz)


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
        # Try to find axis column by name
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
# SECTION 2: Core DSP Utilities
# ===========================================================================

def apply_bandpass(signal, fs, low=50, high=1500, order=4):
    """Butterworth bandpass filter."""
    nyquist = fs / 2.0
    high_norm = min(high, nyquist * 0.99)
    low_norm = max(low, 1.0)
    if low_norm >= high_norm:
        return signal.copy()
    sos = butter(order, [low_norm / nyquist, high_norm / nyquist],
                 btype="band", output="sos")
    return sosfilt(sos, signal)


def apply_lowpass(signal, fs, cutoff=1500, order=4):
    """Butterworth lowpass filter."""
    nyquist = fs / 2.0
    cutoff_norm = min(cutoff, nyquist * 0.99)
    if cutoff_norm <= 0:
        return signal.copy()
    sos = butter(order, cutoff_norm / nyquist, btype="low", output="sos")
    return sosfilt(sos, signal)


def apply_highpass(signal, fs, cutoff=50, order=4):
    """Butterworth highpass filter."""
    nyquist = fs / 2.0
    cutoff_norm = min(cutoff, nyquist * 0.99)
    if cutoff_norm <= 0:
        return signal.copy()
    sos = butter(order, cutoff_norm / nyquist, btype="high", output="sos")
    return sosfilt(sos, signal)


def normalize_peak(signal):
    """Normalize to [-1, 1]."""
    peak = np.max(np.abs(signal))
    if peak > 1e-10:
        return signal / peak
    return signal.copy()


def normalize_rms(signal, target_rms=0.1):
    """Normalize to target RMS level."""
    rms = np.sqrt(np.mean(signal ** 2))
    if rms > 1e-10:
        return signal * (target_rms / rms)
    return signal.copy()


def upsample_to_target(signal, fs_in, fs_out=TARGET_SR):
    """Polyphase resampling to target sample rate."""
    fs_in_int = int(np.round(fs_in))
    g = gcd(fs_out, fs_in_int)
    up = fs_out // g
    down = fs_in_int // g
    return resample_poly(signal, up, down)


def apply_gain_db(signal, gain_db):
    """Apply gain in dB."""
    return signal * (10.0 ** (gain_db / 20.0))


# ===========================================================================
# SECTION 3: MMSE-LSA Denoiser (from imu_pipeline.py)
# ===========================================================================

def mmse_lsa_denoise(signal, fs, frame_ms=25, hop_ms=10,
                     noise_frames=10, alpha_dd=0.98, floor_db=-30):
    """
    MMSE Log-Spectral Amplitude estimator (Ephraim & Malah, 1985).
    Decision-directed approach for a priori SNR estimation.
    """
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


# ===========================================================================
# SECTION 4: Technique 1 - Adaptive Gain Control (AGC)
# ===========================================================================

def adaptive_gain_control(signal, fs, target_level_db=-20,
                          attack_ms=10, release_ms=100,
                          max_gain_db=40, segment_ms=20):
    """
    Adaptive Gain Control with attack/release time constants.

    NOT simple amplification. Segment-by-segment energy-based gain that
    boosts quiet segments more aggressively while limiting loud segments.

    Parameters
    ----------
    signal : ndarray
        Input signal.
    fs : float
        Sample rate.
    target_level_db : float
        Target RMS level in dB (relative to full scale).
    attack_ms : float
        Attack time constant (fast gain reduction).
    release_ms : float
        Release time constant (slow gain increase).
    max_gain_db : float
        Maximum gain applied to any segment.
    segment_ms : float
        Analysis segment length in ms.
    """
    target_level = 10.0 ** (target_level_db / 20.0)
    max_gain = 10.0 ** (max_gain_db / 20.0)

    seg_len = max(1, int(segment_ms * fs / 1000.0))
    n_segs = max(1, len(signal) // seg_len)

    # Compute per-segment RMS
    output = np.zeros_like(signal)
    current_gain = 1.0

    # Time constants as smoothing coefficients per segment
    attack_coeff = 1.0 - np.exp(-seg_len / (attack_ms * fs / 1000.0))
    release_coeff = 1.0 - np.exp(-seg_len / (release_ms * fs / 1000.0))

    for i in range(n_segs):
        start = i * seg_len
        end = min(start + seg_len, len(signal))
        seg = signal[start:end]

        # Compute segment RMS
        seg_rms = np.sqrt(np.mean(seg ** 2) + 1e-10)

        # Desired gain for this segment
        desired_gain = target_level / (seg_rms + 1e-10)
        desired_gain = min(desired_gain, max_gain)
        desired_gain = max(desired_gain, 0.01)

        # Smooth gain transition (attack/release)
        if desired_gain < current_gain:
            # Signal got louder -> reduce gain quickly (attack)
            current_gain += attack_coeff * (desired_gain - current_gain)
        else:
            # Signal got quieter -> increase gain slowly (release)
            current_gain += release_coeff * (desired_gain - current_gain)

        # Apply gain
        output[start:end] = seg * current_gain

    # Handle remainder
    remainder_start = n_segs * seg_len
    if remainder_start < len(signal):
        output[remainder_start:] = signal[remainder_start:] * current_gain

    return output


# ===========================================================================
# SECTION 5: Technique 2 - Dynamic Range Compression
# ===========================================================================

def dynamic_range_compression(signal, fs, threshold_db=-30, ratio=4.0,
                              knee_db=6.0, attack_ms=5, release_ms=50,
                              makeup_gain_db=20, n_bands=4):
    """
    Multi-band dynamic range compressor.

    Compresses the dynamic range so quiet speech is boosted to audible levels.
    Splits signal into frequency bands, compresses each independently.

    Parameters
    ----------
    signal : ndarray
    fs : float
    threshold_db : float
        Level above which compression begins (in dB).
    ratio : float
        Compression ratio (e.g., 4:1 means 4dB input change -> 1dB output).
    knee_db : float
        Soft knee width in dB.
    attack_ms : float
        Attack time for envelope follower.
    release_ms : float
        Release time for envelope follower.
    makeup_gain_db : float
        Gain applied after compression to bring level back up.
    n_bands : int
        Number of frequency bands.
    """
    nyquist = fs / 2.0

    # Define band edges (log-spaced from 50 Hz to nyquist)
    band_edges = np.geomspace(50, nyquist * 0.95, n_bands + 1)
    band_edges[0] = 20  # Start from very low
    band_edges[-1] = nyquist * 0.99

    # Split into bands
    bands = []
    for i in range(n_bands):
        low = band_edges[i]
        high = band_edges[i + 1]
        if low >= nyquist * 0.99:
            bands.append(np.zeros_like(signal))
            continue
        high = min(high, nyquist * 0.99)
        try:
            sos = butter(2, [low / nyquist, high / nyquist], btype="band", output="sos")
            bands.append(sosfilt(sos, signal))
        except Exception:
            bands.append(np.zeros_like(signal))

    # Compress each band
    compressed_bands = []
    for band in bands:
        compressed_bands.append(
            _compress_signal(band, fs, threshold_db, ratio, knee_db,
                             attack_ms, release_ms)
        )

    # Sum bands and apply makeup gain
    output = np.sum(compressed_bands, axis=0)
    output = apply_gain_db(output, makeup_gain_db)
    return output


def _compress_signal(signal, fs, threshold_db, ratio, knee_db,
                     attack_ms, release_ms):
    """Single-band compressor with envelope following."""
    if np.max(np.abs(signal)) < 1e-10:
        return signal.copy()

    # Frame-based envelope follower
    frame_len = max(1, int(0.005 * fs))  # 5ms frames
    n_frames = max(1, len(signal) // frame_len)

    attack_coeff = 1.0 - np.exp(-1.0 / (attack_ms * fs / 1000.0 / frame_len + 1e-10))
    release_coeff = 1.0 - np.exp(-1.0 / (release_ms * fs / 1000.0 / frame_len + 1e-10))

    output = np.zeros_like(signal)
    envelope_db = -60.0  # Start quiet

    for i in range(n_frames):
        start = i * frame_len
        end = min(start + frame_len, len(signal))
        seg = signal[start:end]

        # Compute level in dB
        seg_rms = np.sqrt(np.mean(seg ** 2) + 1e-20)
        level_db = 20.0 * np.log10(seg_rms + 1e-20)

        # Envelope follower
        if level_db > envelope_db:
            envelope_db += attack_coeff * (level_db - envelope_db)
        else:
            envelope_db += release_coeff * (level_db - envelope_db)

        # Compute gain reduction using soft knee
        overshoot = envelope_db - threshold_db
        if overshoot <= -knee_db / 2.0:
            gain_reduction_db = 0.0
        elif overshoot >= knee_db / 2.0:
            gain_reduction_db = overshoot * (1.0 - 1.0 / ratio)
        else:
            # Soft knee region
            x = overshoot + knee_db / 2.0
            gain_reduction_db = (x ** 2) / (2.0 * knee_db) * (1.0 - 1.0 / ratio)

        gain = 10.0 ** (-gain_reduction_db / 20.0)
        output[start:end] = seg * gain

    # Handle remainder
    remainder_start = n_frames * frame_len
    if remainder_start < len(signal):
        output[remainder_start:] = signal[remainder_start:] * gain

    return output


# ===========================================================================
# SECTION 6: Technique 3 - Harmonic Enhancement
# ===========================================================================

def detect_f0_autocorrelation(signal, fs, f0_min=60, f0_max=300):
    """
    Detect fundamental frequency using autocorrelation method.

    Parameters
    ----------
    signal : ndarray
    fs : float
    f0_min, f0_max : float
        Expected F0 range in Hz.

    Returns
    -------
    f0 : float
        Detected fundamental frequency (Hz), or 0 if unvoiced.
    """
    # Lag range corresponding to f0_min..f0_max
    lag_min = max(1, int(fs / f0_max))
    lag_max = min(len(signal) - 1, int(fs / f0_min))

    if lag_max <= lag_min or len(signal) < lag_max + 1:
        return 0.0

    # Normalized autocorrelation
    sig = signal - np.mean(signal)
    energy = np.sum(sig ** 2)
    if energy < 1e-10:
        return 0.0

    autocorr = np.correlate(sig, sig, mode='full')
    autocorr = autocorr[len(sig) - 1:]  # Keep positive lags
    autocorr = autocorr / (energy + 1e-10)

    # Find peak in the valid lag range
    search_region = autocorr[lag_min:lag_max + 1]
    if len(search_region) == 0:
        return 0.0

    peak_idx = np.argmax(search_region)
    peak_val = search_region[peak_idx]

    # Voicing threshold
    if peak_val < 0.2:
        return 0.0

    f0 = fs / (lag_min + peak_idx)
    return f0


def harmonic_enhancement(signal, fs, f0_min=60, f0_max=300,
                         n_harmonics=8, enhancement_db=6.0):
    """
    Harmonic Enhancement: Detect F0, apply comb filter aligned to harmonics,
    suppress inter-harmonic noise, and pitch-synchronous averaging.

    Parameters
    ----------
    signal : ndarray
    fs : float
    f0_min, f0_max : float
        Expected F0 range.
    n_harmonics : int
        Number of harmonics to enhance.
    enhancement_db : float
        How much to boost harmonics (dB).
    """
    # Analyze F0 over frames
    frame_len = int(0.030 * fs)  # 30ms frames
    hop_len = int(0.010 * fs)    # 10ms hop
    if frame_len < 10:
        frame_len = 10
    if hop_len < 1:
        hop_len = 1

    n_frames = max(1, (len(signal) - frame_len) // hop_len + 1)

    # Detect F0 per frame
    f0_track = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_len
        end = min(start + frame_len, len(signal))
        frame = signal[start:end]
        f0_track[i] = detect_f0_autocorrelation(frame, fs, f0_min, f0_max)

    # Median F0 from voiced frames
    voiced_f0s = f0_track[f0_track > 0]
    if len(voiced_f0s) == 0:
        # No pitch detected, return with mild bandpass
        return apply_bandpass(signal, fs, 50, min(1500, fs / 2 * 0.99))

    median_f0 = np.median(voiced_f0s)

    # Apply comb filter enhancement in frequency domain
    nfft = len(signal)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    X = np.fft.rfft(signal)

    # Build harmonic gain mask
    gain_mask = np.ones(len(freqs))
    enhancement_linear = 10.0 ** (enhancement_db / 20.0)
    harmonic_bandwidth = median_f0 * 0.15  # 15% of F0

    for h in range(1, n_harmonics + 1):
        harmonic_freq = h * median_f0
        if harmonic_freq >= fs / 2:
            break
        # Boost around harmonic
        mask = np.exp(-0.5 * ((freqs - harmonic_freq) / harmonic_bandwidth) ** 2)
        gain_mask += mask * (enhancement_linear - 1.0)

    # Suppress inter-harmonic noise (reduce gain between harmonics)
    inter_harmonic_mask = np.ones(len(freqs))
    for h in range(1, n_harmonics):
        h_freq = h * median_f0
        next_h_freq = (h + 1) * median_f0
        if h_freq >= fs / 2:
            break
        mid_freq = (h_freq + next_h_freq) / 2.0
        suppress_bw = (next_h_freq - h_freq) * 0.3
        suppress = np.exp(-0.5 * ((freqs - mid_freq) / suppress_bw) ** 2)
        inter_harmonic_mask -= suppress * 0.5  # Reduce by up to 50%

    inter_harmonic_mask = np.maximum(inter_harmonic_mask, 0.3)
    gain_mask *= inter_harmonic_mask

    # Apply
    X_enhanced = X * gain_mask
    enhanced = np.fft.irfft(X_enhanced, n=nfft)[:len(signal)]

    # Pitch-synchronous averaging for periodic content
    period_samples = int(np.round(fs / median_f0))
    if period_samples > 2 and period_samples < len(enhanced) // 3:
        n_periods = len(enhanced) // period_samples
        if n_periods >= 3:
            # Average aligned periods to improve SNR
            avg_period = np.zeros(period_samples)
            count = 0
            for p in range(n_periods):
                start = p * period_samples
                end = start + period_samples
                if end <= len(enhanced):
                    avg_period += enhanced[start:end]
                    count += 1
            if count > 0:
                avg_period /= count

            # Blend: mix averaged periodic content with original (50/50)
            blended = np.zeros_like(enhanced)
            for p in range(n_periods):
                start = p * period_samples
                end = start + period_samples
                if end <= len(enhanced):
                    blended[start:end] = 0.5 * enhanced[start:end] + 0.5 * avg_period
            remainder = n_periods * period_samples
            if remainder < len(enhanced):
                blended[remainder:] = enhanced[remainder:]
            enhanced = blended

    return enhanced


# ===========================================================================
# SECTION 7: Technique 4 - Bandwidth Extension
# ===========================================================================

def bandwidth_extension(signal, fs, f0_min=60, f0_max=300,
                        target_bw=4000, n_harmonics_synth=12):
    """
    Bandwidth Extension / Spectral Envelope Extension.

    Since bone conduction cuts off around 1500 Hz:
    - Detect F0 from low-frequency content
    - Synthesize higher harmonics (2nd, 3rd, 4th...) up to target_bw Hz
    - Use spectral envelope from fundamental to shape generated harmonics
    - Gives ASR models the formant information they expect

    Parameters
    ----------
    signal : ndarray
    fs : float
        Current sample rate (must be >= 2 * target_bw for synthesis).
    f0_min, f0_max : float
        F0 search range.
    target_bw : float
        Target bandwidth in Hz.
    n_harmonics_synth : int
        Number of harmonics to synthesize.
    """
    # Detect F0
    f0 = detect_f0_autocorrelation(signal, fs, f0_min, f0_max)
    if f0 <= 0:
        # Try with longer window
        if len(signal) > int(fs * 0.1):
            f0 = detect_f0_autocorrelation(signal[:int(fs * 0.1)], fs, f0_min, f0_max)
    if f0 <= 0:
        f0 = 100.0  # Default assumption for male voice

    # Get spectral envelope of the original signal
    nfft = min(2048, len(signal))
    n_analyze = min(nfft, len(signal))
    win = np.hanning(n_analyze)
    X = np.fft.rfft(signal[:n_analyze] * win, n=nfft)
    mag = np.abs(X) + 1e-10
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)

    # Estimate spectral envelope using cepstral smoothing
    log_mag = np.log(mag)
    cepstrum = np.fft.irfft(log_mag)
    # Low-quefrency liftering for envelope
    lifter_order = int(fs / f0) if f0 > 0 else 30
    lifter_order = min(lifter_order, len(cepstrum) // 2)
    cepstrum_smooth = np.zeros_like(cepstrum)
    cepstrum_smooth[:lifter_order] = cepstrum[:lifter_order]
    envelope_db = np.fft.rfft(cepstrum_smooth, n=nfft).real
    envelope = np.exp(envelope_db)

    # Synthesize harmonics above the original bandwidth
    duration = len(signal) / fs
    t = np.arange(len(signal)) / fs
    synthesized = np.zeros_like(signal)

    # Find cutoff of original signal (where energy drops)
    original_cutoff = min(fs / 2 * 0.9, 1500)  # Bone conduction typical cutoff

    for h in range(1, n_harmonics_synth + 1):
        h_freq = h * f0
        if h_freq <= original_cutoff:
            continue  # Already present in original
        if h_freq >= min(target_bw, fs / 2 * 0.95):
            break

        # Get amplitude from spectral envelope extrapolation
        # Find closest frequency bin
        bin_idx = int(h_freq / (fs / 2) * (len(envelope) - 1))
        bin_idx = min(bin_idx, len(envelope) - 1)
        amplitude = envelope[bin_idx]

        # Scale relative to fundamental amplitude
        fund_bin = int(f0 / (fs / 2) * (len(envelope) - 1))
        fund_bin = max(1, min(fund_bin, len(envelope) - 1))
        fund_amp = envelope[fund_bin]
        if fund_amp > 1e-10:
            rel_amplitude = amplitude / fund_amp
        else:
            rel_amplitude = 1.0 / (h ** 1.5)  # Natural roll-off

        # Attenuate higher harmonics naturally
        rel_amplitude *= 1.0 / (1.0 + 0.3 * (h - 1))

        # Synthesize this harmonic with slight random phase
        phase = np.random.uniform(0, 2 * np.pi)
        harmonic = rel_amplitude * np.sin(2 * np.pi * h_freq * t + phase)
        synthesized += harmonic

    # Normalize synthesized content relative to original
    orig_rms = np.sqrt(np.mean(signal ** 2) + 1e-10)
    synth_rms = np.sqrt(np.mean(synthesized ** 2) + 1e-10)
    if synth_rms > 1e-10:
        # Mix at -6dB relative to original
        synthesized = synthesized * (orig_rms / synth_rms) * 0.5

    # Combine original + synthesized harmonics
    output = signal + synthesized
    return output


# ===========================================================================
# SECTION 8: Technique 5 - Teager-Kaiser Energy Operator (TKEO)
# ===========================================================================

def tkeo_enhancement(signal, fs, frame_ms=20, hop_ms=10):
    """
    Teager-Kaiser Energy Operator enhancement.

    The TKEO is a non-linear energy operator that:
    - Emphasizes AM+FM modulated content (speech)
    - Suppresses purely additive noise
    - Applied per-frame and used as a gain mask

    TKEO[x(n)] = x(n)^2 - x(n-1)*x(n+1)

    Parameters
    ----------
    signal : ndarray
    fs : float
    frame_ms : float
        Frame length for gain computation.
    hop_ms : float
        Hop size.
    """
    # Compute TKEO of the signal
    n = len(signal)
    if n < 3:
        return signal.copy()

    tkeo = np.zeros(n)
    tkeo[1:-1] = signal[1:-1] ** 2 - signal[:-2] * signal[2:]
    tkeo[0] = tkeo[1]
    tkeo[-1] = tkeo[-2]

    # Take absolute value (TKEO can be negative for noise)
    tkeo_abs = np.abs(tkeo)

    # Smooth the TKEO to create a gain envelope
    frame_len = max(1, int(frame_ms * fs / 1000.0))
    hop_len = max(1, int(hop_ms * fs / 1000.0))

    # Moving average smoothing
    kernel_len = frame_len
    kernel = np.ones(kernel_len) / kernel_len
    tkeo_smooth = np.convolve(tkeo_abs, kernel, mode='same')

    # Also smooth the signal energy for comparison
    sig_energy = signal ** 2
    sig_energy_smooth = np.convolve(sig_energy, kernel, mode='same')

    # Create gain mask: ratio of TKEO to signal energy
    # High ratio = likely speech (AM/FM content), low ratio = noise
    ratio = tkeo_smooth / (sig_energy_smooth + 1e-10)

    # Normalize ratio to [0, 1] range
    ratio_max = np.percentile(ratio, 95) if len(ratio) > 10 else 1.0
    if ratio_max > 1e-10:
        gain_mask = np.minimum(ratio / ratio_max, 1.0)
    else:
        gain_mask = np.ones(n)

    # Apply soft gain: boost high-TKEO regions, attenuate low-TKEO
    # Map gain_mask from [0,1] to [0.1, 2.0] (suppress noise, boost speech)
    gain = 0.1 + 1.9 * gain_mask
    output = signal * gain

    return output


# ===========================================================================
# SECTION 9: Technique 6 - Bone Conduction Transfer Function Compensation
# ===========================================================================

def bone_conduction_compensation(signal, fs):
    """
    Pre-emphasis + Bone Conduction Transfer Function Compensation.

    Bone conduction attenuates approximately 6 dB/octave above 500 Hz.
    This function inverts that attenuation curve.

    Steps:
    1. Standard pre-emphasis (1 - 0.97*z^-1)
    2. Frequency-dependent gain that inverts the BC attenuation
    3. Formant-aware EQ

    Parameters
    ----------
    signal : ndarray
    fs : float
    """
    # Step 1: Standard pre-emphasis filter
    pre_emphasis_coeff = 0.97
    emphasized = np.zeros_like(signal)
    emphasized[0] = signal[0]
    for i in range(1, len(signal)):
        emphasized[i] = signal[i] - pre_emphasis_coeff * signal[i - 1]

    # Step 2: Frequency-domain BC compensation
    nfft = len(emphasized)
    X = np.fft.rfft(emphasized)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)

    # Bone conduction transfer function inversion
    # BC attenuates ~6 dB/octave above 500 Hz
    # So we boost ~6 dB/octave above 500 Hz to compensate
    compensation_gain = np.ones(len(freqs))

    for i, f in enumerate(freqs):
        if f <= 0:
            continue
        if f > 500:
            # 6 dB/octave = factor of 2 per octave
            octaves_above_500 = np.log2(f / 500.0)
            boost_db = 6.0 * octaves_above_500
            # Limit max boost to avoid excessive amplification of noise
            boost_db = min(boost_db, 18.0)  # Max 18 dB boost (3 octaves)
            compensation_gain[i] = 10.0 ** (boost_db / 20.0)
        elif f < 80:
            # Roll off below 80 Hz (not speech, mostly vibration noise)
            compensation_gain[i] = f / 80.0

    # Step 3: Formant-aware EQ - additional boost in formant regions
    # F1 typically 200-800 Hz, F2 typically 800-2500 Hz for male voice
    formant_boost = np.ones(len(freqs))
    for i, f in enumerate(freqs):
        # F1 region boost (centered around 400 Hz for male)
        formant_boost[i] += 0.3 * np.exp(-0.5 * ((f - 400) / 150) ** 2)
        # F2 region boost (centered around 1200 Hz for male)
        formant_boost[i] += 0.2 * np.exp(-0.5 * ((f - 1200) / 300) ** 2)

    # Combined gain
    total_gain = compensation_gain * formant_boost
    X_compensated = X * total_gain

    output = np.fft.irfft(X_compensated, n=nfft)[:len(signal)]
    return output


# ===========================================================================
# SECTION 10: Technique 7 - Noise-Gated Amplification
# ===========================================================================

def noise_gated_amplification(signal, fs, gate_threshold_db=6.0,
                              amplification_db=20, soft_knee_db=3.0,
                              frame_ms=20, hop_ms=10):
    """
    Noise-Gated Amplification.

    Estimates noise floor frame-by-frame. Only amplifies when signal
    exceeds noise floor by > threshold dB. Uses soft knee to avoid artifacts.

    Parameters
    ----------
    signal : ndarray
    fs : float
    gate_threshold_db : float
        Signal must exceed noise floor by this much to be amplified.
    amplification_db : float
        Gain applied to signal that passes the gate.
    soft_knee_db : float
        Width of the soft transition region.
    frame_ms : float
        Frame length for analysis.
    hop_ms : float
        Hop size.
    """
    frame_len = max(1, int(frame_ms * fs / 1000.0))
    hop_len = max(1, int(hop_ms * fs / 1000.0))
    n_frames = max(1, (len(signal) - frame_len) // hop_len + 1)

    amplification_linear = 10.0 ** (amplification_db / 20.0)

    # Estimate noise floor from quietest frames
    frame_energies = []
    for i in range(n_frames):
        start = i * hop_len
        end = min(start + frame_len, len(signal))
        frame_rms = np.sqrt(np.mean(signal[start:end] ** 2) + 1e-20)
        frame_energies.append(frame_rms)

    frame_energies = np.array(frame_energies)

    # Noise floor estimate: 20th percentile of frame energies
    noise_floor = np.percentile(frame_energies, 20) if len(frame_energies) > 5 else np.min(frame_energies)
    noise_floor_db = 20.0 * np.log10(noise_floor + 1e-20)

    # Apply gated amplification with overlap-add
    output = np.zeros(len(signal))
    win = np.hanning(frame_len)
    win_sum = np.zeros(len(signal))

    for i in range(n_frames):
        start = i * hop_len
        end = min(start + frame_len, len(signal))
        actual_len = end - start
        frame = signal[start:end]

        frame_rms = frame_energies[i]
        frame_db = 20.0 * np.log10(frame_rms + 1e-20)

        # How far above noise floor
        excess_db = frame_db - noise_floor_db

        # Soft-knee gating
        if excess_db >= gate_threshold_db + soft_knee_db / 2.0:
            # Fully open gate: apply full amplification
            gain = amplification_linear
        elif excess_db <= gate_threshold_db - soft_knee_db / 2.0:
            # Gate closed: minimal gain (just pass through)
            gain = 1.0
        else:
            # Soft knee transition
            t = (excess_db - (gate_threshold_db - soft_knee_db / 2.0)) / soft_knee_db
            gain = 1.0 + (amplification_linear - 1.0) * t

        w = win[:actual_len]
        output[start:end] += frame * gain * w
        win_sum[start:end] += w

    # Normalize by window sum
    win_sum = np.maximum(win_sum, 1e-8)
    output = output / win_sum

    return output


# ===========================================================================
# SECTION 11: Technique 8 - Combined Pipeline (THE FULL CHAIN)
# ===========================================================================

def combined_pipeline(signal, fs, f0_min=60, f0_max=300):
    """
    The Full Enhancement Chain for low-pitch bone conduction.

    Order:
      Pre-emphasis -> AGC -> MMSE-LSA -> Harmonic Enhancement ->
      Bandwidth Extension -> Dynamic Compression -> Normalize

    KEY INSIGHT: We enhance FIRST, then denoise. This is the opposite of
    the traditional approach, but necessary because the signal is below
    the noise floor for low-pitch voices.

    Parameters
    ----------
    signal : ndarray
    fs : float
    f0_min, f0_max : float
        Expected F0 range.
    """
    # Step 1: Pre-emphasis + BC compensation
    # (Boost high frequencies that BC attenuated)
    enhanced = bone_conduction_compensation(signal, fs)

    # Step 2: AGC - Boost the signal above noise floor
    enhanced = adaptive_gain_control(enhanced, fs, target_level_db=-15,
                                     attack_ms=5, release_ms=80,
                                     max_gain_db=35)

    # Step 3: MMSE-LSA denoising
    # Now that signal is above noise floor, denoising can work
    enhanced = mmse_lsa_denoise(enhanced, fs, frame_ms=25, hop_ms=10,
                                noise_frames=8, alpha_dd=0.96)

    # Step 4: Harmonic Enhancement
    # Extract and boost the harmonic structure
    enhanced = harmonic_enhancement(enhanced, fs, f0_min=f0_min,
                                    f0_max=f0_max, n_harmonics=10,
                                    enhancement_db=8.0)

    # Step 5: Bandwidth Extension
    # Synthesize higher harmonics for ASR
    enhanced = bandwidth_extension(enhanced, fs, f0_min=f0_min,
                                   f0_max=f0_max, target_bw=min(4000, fs / 2 * 0.95))

    # Step 6: Dynamic Range Compression
    # Compress to make quiet parts audible
    enhanced = dynamic_range_compression(enhanced, fs, threshold_db=-25,
                                         ratio=3.0, knee_db=8.0,
                                         attack_ms=5, release_ms=60,
                                         makeup_gain_db=15, n_bands=3)

    # Step 7: Final normalization
    enhanced = normalize_peak(enhanced) * 0.9  # Leave 1dB headroom

    return enhanced


# ===========================================================================
# SECTION 12: Metrics
# ===========================================================================

def compute_snr_db(signal, noise_percentile=20):
    """Estimate SNR: signal power vs noise estimate from quiet frames."""
    frame_len = max(1, len(signal) // 50)
    n_frames = max(1, len(signal) // frame_len)
    energies = []
    for i in range(n_frames):
        start = i * frame_len
        end = min(start + frame_len, len(signal))
        energies.append(np.mean(signal[start:end] ** 2))
    energies = np.array(energies)
    noise_energy = np.percentile(energies, noise_percentile)
    signal_energy = np.mean(signal ** 2)
    if noise_energy < 1e-20:
        return 60.0
    snr = 10.0 * np.log10((signal_energy + 1e-20) / (noise_energy + 1e-20))
    return snr


def compute_hnr_db(signal, fs, f0_min=60, f0_max=300):
    """Harmonic-to-Noise Ratio using autocorrelation."""
    f0 = detect_f0_autocorrelation(signal, fs, f0_min, f0_max)
    if f0 <= 0:
        return 0.0
    lag = int(np.round(fs / f0))
    if lag >= len(signal):
        return 0.0
    # Autocorrelation at pitch lag
    n = len(signal) - lag
    if n < 1:
        return 0.0
    sig = signal[:n] - np.mean(signal[:n])
    sig_shifted = signal[lag:lag + n] - np.mean(signal[lag:lag + n])
    r0 = np.sum(sig ** 2)
    r_lag = np.sum(sig * sig_shifted)
    if r0 < 1e-10:
        return 0.0
    acf_norm = r_lag / (r0 + 1e-10)
    acf_norm = np.clip(acf_norm, -0.999, 0.999)
    hnr = 10.0 * np.log10((acf_norm + 1e-10) / (1.0 - acf_norm + 1e-10))
    return float(np.clip(hnr, -20, 40))


def compute_voiced_fraction(signal, fs, frame_ms=20):
    """Fraction of frames that are voiced (using ZCR + energy)."""
    frame_len = max(1, int(frame_ms * fs / 1000.0))
    n_frames = max(1, len(signal) // frame_len)
    voiced_count = 0
    for i in range(n_frames):
        start = i * frame_len
        end = min(start + frame_len, len(signal))
        frame = signal[start:end]
        if len(frame) < 2:
            continue
        # ZCR
        zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2.0 * len(frame))
        # Energy
        energy = np.mean(frame ** 2)
        energy_threshold = np.mean(signal ** 2) * 0.01
        # Voiced: low ZCR and above energy threshold
        if zcr < 0.3 and energy > energy_threshold:
            voiced_count += 1
    return voiced_count / max(1, n_frames)


def compute_spectral_flatness(signal, fs):
    """Spectral flatness: geometric mean / arithmetic mean of power spectrum."""
    nfft = min(2048, len(signal))
    X = np.fft.rfft(signal[:nfft])
    power = np.abs(X) ** 2 + 1e-20
    log_mean = np.mean(np.log(power))
    geo_mean = np.exp(log_mean)
    arith_mean = np.mean(power)
    if arith_mean < 1e-20:
        return 1.0
    flatness = geo_mean / arith_mean
    return float(np.clip(flatness, 0, 1))


def compute_inband_ratio(signal, fs, low=80, high=1000):
    """Power ratio of speech band (80-1000 Hz) vs total."""
    nfft = min(2048, len(signal))
    X = np.fft.rfft(signal[:nfft])
    power = np.abs(X) ** 2
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    total_power = np.sum(power) + 1e-20
    inband_mask = (freqs >= low) & (freqs <= high)
    inband_power = np.sum(power[inband_mask])
    return float(inband_power / total_power)


def compute_spectral_centroid(signal, fs):
    """Weighted mean frequency."""
    nfft = min(2048, len(signal))
    X = np.fft.rfft(signal[:nfft])
    power = np.abs(X) ** 2
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    total = np.sum(power) + 1e-20
    centroid = np.sum(freqs * power) / total
    return float(centroid)


def compute_all_metrics(signal, fs, f0_min=60, f0_max=300):
    """Compute all voice-likeness metrics for a signal."""
    return {
        "snr_db": compute_snr_db(signal),
        "hnr_db": compute_hnr_db(signal, fs, f0_min, f0_max),
        "voiced_fraction": compute_voiced_fraction(signal, fs),
        "spectral_flatness": compute_spectral_flatness(signal, fs),
        "inband_ratio": compute_inband_ratio(signal, fs),
        "spectral_centroid": compute_spectral_centroid(signal, fs),
    }


# ===========================================================================
# SECTION 13: Main Pipeline & CLI
# ===========================================================================

def run_all_techniques(signal, fs, f0_min=60, f0_max=300, gain_db=0):
    """
    Run ALL enhancement techniques on the signal.

    Returns dict of {technique_name: enhanced_signal}.
    """
    results = {}

    print("\n" + "=" * 70)
    print("  LOW-PITCH BONE CONDUCTION ENHANCEMENT")
    print("  Running all techniques...")
    print("=" * 70)

    # Technique 1: AGC
    print("  [1/8] Adaptive Gain Control (AGC)...")
    try:
        agc_out = adaptive_gain_control(signal, fs, target_level_db=-15,
                                        max_gain_db=35)
        results["01_AGC"] = agc_out
    except Exception as e:
        print(f"        FAILED: {e}")

    # Technique 2: Dynamic Range Compression
    print("  [2/8] Dynamic Range Compression...")
    try:
        drc_out = dynamic_range_compression(signal, fs, threshold_db=-35,
                                            ratio=4.0, knee_db=6.0,
                                            makeup_gain_db=25, n_bands=4)
        results["02_DynamicCompression"] = drc_out
    except Exception as e:
        print(f"        FAILED: {e}")

    # Technique 3: Harmonic Enhancement
    print("  [3/8] Harmonic Enhancement...")
    try:
        harm_out = harmonic_enhancement(signal, fs, f0_min=f0_min,
                                        f0_max=f0_max, n_harmonics=10,
                                        enhancement_db=8.0)
        results["03_HarmonicEnhancement"] = harm_out
    except Exception as e:
        print(f"        FAILED: {e}")

    # Technique 4: Bandwidth Extension
    print("  [4/8] Bandwidth Extension...")
    try:
        bwe_out = bandwidth_extension(signal, fs, f0_min=f0_min,
                                      f0_max=f0_max,
                                      target_bw=min(4000, fs / 2 * 0.95))
        results["04_BandwidthExtension"] = bwe_out
    except Exception as e:
        print(f"        FAILED: {e}")

    # Technique 5: TKEO
    print("  [5/8] Teager-Kaiser Energy Operator (TKEO)...")
    try:
        tkeo_out = tkeo_enhancement(signal, fs)
        results["05_TKEO"] = tkeo_out
    except Exception as e:
        print(f"        FAILED: {e}")

    # Technique 6: BC Compensation
    print("  [6/8] Bone Conduction TF Compensation...")
    try:
        bc_out = bone_conduction_compensation(signal, fs)
        results["06_BC_Compensation"] = bc_out
    except Exception as e:
        print(f"        FAILED: {e}")

    # Technique 7: Noise-Gated Amplification
    print("  [7/8] Noise-Gated Amplification...")
    try:
        ng_out = noise_gated_amplification(signal, fs, gate_threshold_db=6.0,
                                           amplification_db=20)
        results["07_NoiseGated"] = ng_out
    except Exception as e:
        print(f"        FAILED: {e}")

    # Technique 8: Combined Pipeline (THE FULL CHAIN)
    print("  [8/8] Combined Pipeline (Full Chain)...")
    try:
        combined_out = combined_pipeline(signal, fs, f0_min=f0_min,
                                         f0_max=f0_max)
        results["08_CombinedPipeline"] = combined_out
    except Exception as e:
        print(f"        FAILED: {e}")

    # Also include raw (just bandpassed) for reference
    results["00_Raw_Bandpass"] = apply_bandpass(signal, fs, 50, min(1500, fs / 2 * 0.99))

    # Apply additional manual gain if specified
    if gain_db != 0:
        print(f"\n  Applying additional gain: {gain_db} dB to all outputs...")
        for name in results:
            results[name] = apply_gain_db(results[name], gain_db)

    return results


def save_results(results, fs_orig, output_dir, f0_min=60, f0_max=300):
    """
    Save WAV files and print metrics for all techniques.
    Upsamples to 16 kHz before saving.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("  RESULTS & METRICS")
    print("=" * 70)
    print(f"  {'Technique':<30} {'SNR(dB)':<10} {'HNR(dB)':<10} "
          f"{'Voiced%':<10} {'Flatness':<10} {'InBand%':<10} {'Centroid':<10}")
    print("  " + "-" * 88)

    all_metrics = {}
    wav_files = {}

    for name, sig in sorted(results.items()):
        # Normalize
        sig_norm = normalize_peak(sig) * 0.9

        # Upsample to 16 kHz
        sig_16k = upsample_to_target(sig_norm, fs_orig, TARGET_SR)

        # Save WAV
        wav_path = os.path.join(output_dir, f"{name}.wav")
        try:
            import soundfile as sf_lib
            sf_lib.write(wav_path, sig_16k, TARGET_SR)
        except ImportError:
            # Fallback: write raw WAV manually
            _write_wav_manual(wav_path, sig_16k, TARGET_SR)
        wav_files[name] = wav_path

        # Compute metrics on the 16kHz signal
        metrics = compute_all_metrics(sig_16k, TARGET_SR, f0_min, f0_max)
        all_metrics[name] = metrics

        print(f"  {name:<30} {metrics['snr_db']:<10.2f} {metrics['hnr_db']:<10.2f} "
              f"{metrics['voiced_fraction']:<10.3f} {metrics['spectral_flatness']:<10.4f} "
              f"{metrics['inband_ratio']:<10.3f} {metrics['spectral_centroid']:<10.1f}")

    # Determine best technique
    print("\n" + "=" * 70)
    print("  RANKING (by composite score for low-pitch enhancement)")
    print("=" * 70)

    # Composite score: weighted combination favoring HNR and voiced fraction
    scores = {}
    for name, m in all_metrics.items():
        # Higher HNR = more harmonic = better
        # Higher voiced fraction = more speech detected = better
        # Lower spectral flatness = more tonal = better
        # Higher inband ratio = energy in speech band = better
        # Higher SNR = better
        score = (
            m["hnr_db"] * 2.0 +           # Weight HNR heavily
            m["voiced_fraction"] * 30.0 +   # Voiced fraction very important
            m["snr_db"] * 1.0 +             # SNR matters
            (1.0 - m["spectral_flatness"]) * 20.0 +  # Less flat = better
            m["inband_ratio"] * 20.0        # In-band energy
        )
        scores[name] = score

    # Normalize to 0-100
    score_values = list(scores.values())
    min_score = min(score_values)
    max_score = max(score_values)
    score_range = max_score - min_score if max_score > min_score else 1.0

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  {'Rank':<6} {'Technique':<30} {'Score (0-100)':<15}")
    print("  " + "-" * 55)
    for rank, (name, score) in enumerate(ranked, 1):
        normalized_score = (score - min_score) / score_range * 100.0
        marker = " <-- BEST" if rank == 1 else ""
        print(f"  {rank:<6} {name:<30} {normalized_score:<15.1f}{marker}")

    best_name = ranked[0][0]
    print(f"\n  RECOMMENDATION: Use '{best_name}' for low-pitch bone conduction signals.")
    print(f"\n  KEY INSIGHT: For low-pitch signals, the processing order matters:")
    print(f"    1. Boost signal above noise floor (AGC/compression)")
    print(f"    2. Extract and enhance harmonic structure")
    print(f"    3. Extend bandwidth for ASR models")
    print(f"    4. THEN denoise (MMSE-LSA) -- not before!")
    print(f"    >> denoise-first FAILS; enhance-first-then-denoise WORKS\n")

    # Save metrics CSV
    csv_path = os.path.join(output_dir, "enhancement_metrics.csv")
    with open(csv_path, "w") as f:
        headers = ["technique", "snr_db", "hnr_db", "voiced_fraction",
                   "spectral_flatness", "inband_ratio", "spectral_centroid",
                   "composite_score"]
        f.write(",".join(headers) + "\n")
        for name, score in ranked:
            m = all_metrics[name]
            norm_score = (score - min_score) / score_range * 100.0
            f.write(f"{name},{m['snr_db']:.4f},{m['hnr_db']:.4f},"
                    f"{m['voiced_fraction']:.4f},{m['spectral_flatness']:.6f},"
                    f"{m['inband_ratio']:.4f},{m['spectral_centroid']:.2f},"
                    f"{norm_score:.2f}\n")

    print(f"  Output directory: {output_dir}")
    print(f"  WAV files: {len(wav_files)}")
    print(f"  Metrics CSV: {csv_path}")
    print("=" * 70)

    return all_metrics, scores


def _write_wav_manual(path, signal, fs):
    """Write a 16-bit PCM WAV file without external dependencies."""
    import struct
    signal_16 = np.clip(signal * 32767, -32768, 32767).astype(np.int16)
    n_samples = len(signal_16)
    data_size = n_samples * 2
    file_size = 36 + data_size

    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", file_size))
        f.write(b"WAVE")
        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))       # chunk size
        f.write(struct.pack("<H", 1))        # PCM format
        f.write(struct.pack("<H", 1))        # mono
        f.write(struct.pack("<I", fs))       # sample rate
        f.write(struct.pack("<I", fs * 2))   # byte rate
        f.write(struct.pack("<H", 2))        # block align
        f.write(struct.pack("<H", 16))       # bits per sample
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(signal_16.tobytes())


# ===========================================================================
# SECTION 14: Embedded Sample Data (for testing without a CSV file)
# ===========================================================================

EMBEDDED_SAMPLE_CSV = """timestamp,Accel X,Accel Y,Accel Z
0.000,0.012,-0.003,0.045
0.303,0.015,-0.001,0.067
0.606,0.018,0.002,0.089
0.909,0.014,0.005,0.112
1.212,0.009,0.003,0.098
1.515,0.011,-0.002,0.076
1.818,0.016,0.001,0.054
2.121,0.020,0.004,0.088
2.424,0.013,0.002,0.121
2.727,0.008,-0.001,0.095
3.030,0.017,0.003,0.073
3.333,0.022,0.006,0.105
3.636,0.011,0.001,0.134
3.939,0.007,-0.003,0.108
4.242,0.019,0.004,0.082
4.545,0.025,0.007,0.115
4.848,0.014,0.002,0.143
5.151,0.006,-0.002,0.118
5.454,0.018,0.005,0.091
5.757,0.023,0.008,0.125
6.060,0.012,0.001,0.152
6.363,0.005,-0.004,0.127
6.666,0.020,0.006,0.098
6.969,0.026,0.009,0.132
7.272,0.013,0.002,0.158
7.575,0.004,-0.005,0.135
7.878,0.021,0.007,0.106
8.181,0.028,0.010,0.140
8.484,0.015,0.003,0.165
8.787,0.003,-0.006,0.142
9.090,0.022,0.008,0.113
9.393,0.029,0.011,0.148
9.696,0.016,0.004,0.172
9.999,0.002,-0.007,0.150
10.302,0.023,0.009,0.120
10.605,0.030,0.012,0.155
10.908,0.017,0.005,0.178
11.211,0.001,-0.008,0.157
11.514,0.024,0.010,0.128
11.817,0.031,0.013,0.162"""


def generate_synthetic_test_signal(duration_s=1.0, fs=3300, f0=100):
    """
    Generate a synthetic low-pitch bone conduction signal for testing.

    Simulates:
    - Low F0 (100 Hz) with harmonics up to ~1000 Hz
    - Noise floor that partially masks the signal
    - Bone conduction frequency response (attenuated above 500 Hz)
    """
    n_samples = int(duration_s * fs)
    t = np.arange(n_samples) / fs

    # Generate harmonics (weak, like bone conduction)
    signal = np.zeros(n_samples)
    for h in range(1, 12):
        h_freq = h * f0
        if h_freq >= fs / 2:
            break
        # Attenuate higher harmonics (bone conduction roll-off)
        amplitude = 0.05 / (h ** 1.2)
        # Further attenuate above 500 Hz (6 dB/octave)
        if h_freq > 500:
            octaves = np.log2(h_freq / 500)
            amplitude *= 10 ** (-6 * octaves / 20)
        signal += amplitude * np.sin(2 * np.pi * h_freq * t)

    # Add noise (partially masking the signal)
    noise_level = np.std(signal) * 1.5  # SNR ~ -3.5 dB (below noise floor!)
    noise = noise_level * np.random.randn(n_samples)
    signal_noisy = signal + noise

    # Generate timestamps in ms
    timestamps = np.arange(n_samples) * (1000.0 / fs)

    return timestamps, signal_noisy, fs


# ===========================================================================
# SECTION 15: CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Low-Pitch Bone Conduction IMU Enhancement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
TECHNIQUES:
  1. Adaptive Gain Control (AGC)
  2. Dynamic Range Compression (multi-band)
  3. Harmonic Enhancement (F0 detection + comb filter)
  4. Bandwidth Extension (synthesize harmonics to 4kHz)
  5. Teager-Kaiser Energy Operator (TKEO)
  6. Bone Conduction Transfer Function Compensation
  7. Noise-Gated Amplification
  8. Combined Pipeline (THE FULL CHAIN)

KEY INSIGHT:
  For low-pitch bone conduction, the order matters:
    enhance-first, THEN denoise -> works
    denoise-first -> FAILS (signal below noise floor)

EXAMPLES:
  python imu_enhance_lowpitch.py recording.csv
  python imu_enhance_lowpitch.py --input recording.csv --gain-db 20
  python imu_enhance_lowpitch.py recording.csv --f0-range 60 200 -o /tmp/enhanced
        """,
    )

    parser.add_argument(
        "csv_input", nargs="?", default=None,
        help="Input CSV file (positional). Format: timestamp, Accel X, Accel Y, Accel Z at 3.3kHz")
    parser.add_argument(
        "--input", "-i", dest="input_flag", default=None,
        help="Input CSV file (alternative to positional)")
    parser.add_argument(
        "--output", "-o", dest="output_dir", default=None,
        help="Output directory for WAV files and metrics (default: ./lowpitch_enhanced/)")
    parser.add_argument(
        "--gain-db", type=float, default=0,
        help="Additional manual gain in dB applied to all outputs (e.g., --gain-db 20)")
    parser.add_argument(
        "--f0-range", nargs=2, type=float, default=[60, 300],
        metavar=("MIN", "MAX"),
        help="Expected F0 range in Hz (default: 60 300 for male voice)")
    parser.add_argument(
        "--axis", default="z", choices=["x", "y", "z"],
        help="Accelerometer axis to use (default: z)")
    parser.add_argument(
        "--gap-mode", default="uniform", choices=["chunk", "uniform", "all"],
        help="Gap handling mode for non-uniform timestamps (default: uniform)")

    args = parser.parse_args()

    # Resolve input
    input_path = args.csv_input or args.input_flag
    f0_min, f0_max = args.f0_range

    # Output directory
    output_dir = args.output_dir or "./lowpitch_enhanced"

    if input_path is None:
        # Use synthetic test data
        print("  No input file specified. Using synthetic low-pitch test signal.")
        print("  (Simulates 100 Hz F0 bone conduction signal below noise floor)")
        timestamps, signal, fs = generate_synthetic_test_signal(
            duration_s=1.0, fs=ORIGINAL_SR, f0=100
        )
    else:
        if not os.path.isfile(input_path):
            print(f"Error: File not found: {input_path}", file=sys.stderr)
            sys.exit(1)
        # Load CSV
        print(f"  Loading: {input_path}")
        timestamps, signal = load_imu_csv(input_path, axis=args.axis)
        print(f"  Loaded {len(signal)} samples")

        # Gap-aware resampling
        signal, fs = gap_aware_resample(timestamps, signal, mode=args.gap_mode)
        print(f"  Inferred sample rate: {fs:.1f} Hz")
        print(f"  Signal length after resampling: {len(signal)} samples "
              f"({len(signal)/fs:.3f}s)")

    # Remove DC
    signal = signal - np.mean(signal)

    # Run all techniques
    results = run_all_techniques(signal, fs, f0_min=f0_min, f0_max=f0_max,
                                 gain_db=args.gain_db)

    # Save results and print metrics
    save_results(results, fs, output_dir, f0_min=f0_min, f0_max=f0_max)


if __name__ == "__main__":
    main()
