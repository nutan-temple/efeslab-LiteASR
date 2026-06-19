#!/usr/bin/env python3
"""
IMU Audio Analyser - Advanced Bone Conduction Signal Processing
================================================================
Applies 20+ denoising methods to accelerometer data captured from a bone
conduction microphone. Features gap-aware resampling, MMSE-LSA denoising,
voice-likeness metrics, and composite scoring for fair method comparison.

Usage:
    python imu_audio_analyser.py                          # Use embedded sample data
    python imu_audio_analyser.py --input data.csv         # Use CSV file
    python imu_audio_analyser.py --output_dir out         # Specify output directory
    python imu_audio_analyser.py --preset full            # Full grid sweep
    python imu_audio_analyser.py --gap_mode chunk         # Gap-aware resampling
"""

import argparse
import base64
import os
import sys
import warnings
from io import BytesIO, StringIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import soundfile as sf
from scipy import signal as sp_signal
from scipy.linalg import solve_toeplitz
from scipy.special import exp1

warnings.filterwarnings("ignore")

# ============================================================
# CONSTANTS AND CONFIGURATION
# ============================================================
ORIGINAL_SR = 3300  # 3.3 kHz nominal IMU sample rate
DEFAULT_TARGET_SR = 16000  # Common speech SR

# Embedded sample CSV data for standalone testing
SAMPLE_CSV = """timestamp,Accel X,Accel Y,Accel Z
1781858475887.0,-14827.0,7286.0,2033.0
1781858475887.3,-14859.0,7313.0,2109.0
1781858475887.6,-14898.0,7255.0,2042.0
1781858475887.9,-14805.0,7324.0,2024.0
1781858475888.2,-14844.0,7319.0,2053.0
1781858475888.5,-14881.0,7271.0,1983.0
1781858475888.8,-14867.0,7279.0,2067.0
1781858475889.1,-14872.0,7244.0,1986.0
1781858475889.4,-14823.0,7248.0,2007.0
1781858475889.7,-14773.0,7266.0,2071.0
1781858475890.0,-14691.0,7249.0,2086.0
1781858475890.3,-14791.0,7242.0,2017.0
1781858475890.6,-14790.0,7259.0,2035.0
1781858475890.9,-14897.0,7283.0,2015.0
1781858475891.2,-14825.0,7239.0,2032.0
1781858475891.5,-14830.0,7244.0,2037.0
1781858475891.8,-14781.0,7218.0,1997.0
1781858475892.1,-14902.0,7268.0,1983.0
1781858475892.4,-14866.0,7259.0,2045.0
1781858475892.7,-14819.0,7243.0,2028.0
1781858475893.0,-14844.0,7287.0,2039.0
1781858475893.3,-14809.0,7198.0,1992.0
1781858475893.6,-14818.0,7190.0,2011.0
1781858475893.9,-14836.0,7258.0,1965.0
1781858475894.2,-14873.0,7262.0,1968.0
1781858475894.5,-14850.0,7289.0,2033.0
1781858475894.8,-14746.0,7252.0,1993.0
1781858475895.1,-14842.0,7257.0,2054.0
1781858475895.4,-14781.0,7142.0,2015.0
1781858475895.7,-14859.0,7252.0,2005.0
1781858475896.0,-14840.0,7307.0,1999.0
1781858475896.3,-14918.0,7295.0,1991.0
1781858475896.6,-14882.0,7321.0,2052.0
1781858475896.9,-14863.0,7262.0,2039.0
1781858475897.2,-14878.0,7279.0,2014.0
1781858475897.5,-14867.0,7413.0,1984.0
1781858475897.8,-14851.0,7312.0,2037.0
1781858475898.1,-14886.0,7282.0,2046.0
1781858475898.4,-14815.0,7203.0,2016.0
1781858475898.7,-14866.0,7222.0,1990.0
"""

# Composite score weights for voice-likeness metrics
METRIC_WEIGHTS = {
    "hnr_db": 0.25,
    "voiced_fraction": 0.20,
    "inband_ratio": 0.20,
    "snr_db": 0.15,
    "spectral_flatness_inv": 0.10,
    "spectral_centroid_norm": 0.05,
    "crest_factor_norm": 0.03,
    "zcr_inv": 0.02,
}



# ============================================================
# SECTION 2: GAP-AWARE RESAMPLING
# ============================================================
def infer_sample_rate(timestamps):
    """Infer actual sample rate from median inter-sample time."""
    if len(timestamps) < 2:
        return ORIGINAL_SR
    diffs = np.diff(timestamps)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return ORIGINAL_SR
    median_dt = np.median(diffs)
    if median_dt <= 0:
        return ORIGINAL_SR
    return 1000.0 / median_dt  # timestamps assumed in ms


def find_gaps(timestamps, gap_factor=2.0):
    """Find gap indices where inter-sample time exceeds gap_factor * median."""
    if len(timestamps) < 2:
        return []
    diffs = np.diff(timestamps)
    diffs_pos = diffs[diffs > 0]
    if len(diffs_pos) == 0:
        return []
    median_dt = np.median(diffs_pos)
    threshold = gap_factor * median_dt
    gap_indices = np.where(diffs > threshold)[0]
    return gap_indices.tolist()


def resample_gap_chunk(timestamps, signal):
    """
    Gap-aware resampling: 'chunk' mode.
    Find the largest contiguous chunk (gaps > 2x median inter-sample time
    are breakpoints) and use only that chunk.
    """
    gap_indices = find_gaps(timestamps)
    if not gap_indices:
        return signal, infer_sample_rate(timestamps)

    # Build chunk boundaries
    boundaries = [0] + [g + 1 for g in gap_indices] + [len(signal)]
    chunks = [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

    # Find the largest chunk
    largest = max(chunks, key=lambda c: c[1] - c[0])
    chunk_signal = signal[largest[0]:largest[1]]
    chunk_ts = timestamps[largest[0]:largest[1]]
    fs = infer_sample_rate(chunk_ts)
    return chunk_signal, fs


def resample_gap_uniform(timestamps, signal):
    """
    Gap-aware resampling: 'uniform' mode.
    Interpolate to a uniform time grid using cubic interpolation.
    """
    fs = infer_sample_rate(timestamps)
    if len(signal) < 4:
        return signal, fs

    # Create uniform time grid
    t_start = timestamps[0]
    t_end = timestamps[-1]
    dt = 1000.0 / fs  # ms per sample
    n_samples = int((t_end - t_start) / dt) + 1
    n_samples = max(n_samples, len(signal))
    uniform_t = np.linspace(t_start, t_end, n_samples)

    # Cubic interpolation
    from scipy.interpolate import interp1d
    interp_func = interp1d(timestamps, signal, kind="cubic",
                           fill_value="extrapolate")
    uniform_signal = interp_func(uniform_t)
    return uniform_signal, fs


def resample_gap_all(timestamps, signal):
    """
    Gap-aware resampling: 'all' mode.
    Use all samples as-is (fastest, may have artifacts at gaps).
    """
    fs = infer_sample_rate(timestamps)
    return signal, fs


def gap_aware_resample(timestamps, signal, mode="uniform"):
    """Dispatch to appropriate gap-aware resampling mode."""
    if mode == "chunk":
        return resample_gap_chunk(timestamps, signal)
    elif mode == "uniform":
        return resample_gap_uniform(timestamps, signal)
    elif mode == "all":
        return resample_gap_all(timestamps, signal)
    else:
        raise ValueError(f"Unknown gap_mode: {mode}. Use 'chunk', 'uniform', or 'all'.")




# ============================================================
# SECTION 3: CORE DSP UTILITIES
# ============================================================
def detrend_signal(sig):
    """Remove DC offset and linear trend."""
    return sp_signal.detrend(sig, type="linear")


def apply_bandpass(sig, fs, low=50, high=1500, order=4):
    """Apply Butterworth band-pass filter."""
    nyq = fs / 2
    if high >= nyq:
        high = nyq * 0.95
    if low >= high:
        low = high * 0.1
    low_norm = max(low / nyq, 0.001)
    high_norm = min(high / nyq, 0.999)
    b, a = sp_signal.butter(order, [low_norm, high_norm], btype="band")
    return sp_signal.filtfilt(b, a, sig)


def norm_rms(sig, target_rms=0.1):
    """Normalize signal to target RMS level."""
    rms = np.sqrt(np.mean(sig ** 2))
    if rms < 1e-10:
        return sig
    return sig * (target_rms / rms)


def trim_silence(sig, threshold_db=-40, frame_ms=25, fs=16000):
    """Trim leading and trailing silence based on energy threshold."""
    frame_len = max(2, int(fs * frame_ms / 1000))
    if len(sig) < frame_len:
        return sig

    n_frames = len(sig) // frame_len
    if n_frames == 0:
        return sig

    energy = np.array([
        np.sum(sig[i * frame_len:(i + 1) * frame_len] ** 2)
        for i in range(n_frames)
    ])

    max_energy = np.max(energy) if np.max(energy) > 0 else 1.0
    energy_db = 10 * np.log10(energy / max_energy + 1e-10)
    threshold = threshold_db

    # Find first and last frame above threshold
    active = np.where(energy_db > threshold)[0]
    if len(active) == 0:
        return sig

    start = active[0] * frame_len
    end = min((active[-1] + 1) * frame_len, len(sig))
    return sig[start:end]


def resample_poly_to(sig, orig_sr, target_sr):
    """Polyphase resampling to target sample rate."""
    if orig_sr == target_sr:
        return sig
    if len(sig) < 2:
        return sig
    return sp_signal.resample_poly(sig, target_sr, orig_sr)


def wav_bytes(sig, sr):
    """Encode signal to WAV bytes (PCM_16)."""
    # Normalize to [-1, 1] with headroom
    s = sig.copy().astype(np.float64)
    s = s - np.mean(s)
    mx = np.max(np.abs(s))
    if mx > 0:
        s = s / mx * 0.95
    buf = BytesIO()
    sf.write(buf, s, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def normalize_audio(sig):
    """Normalize signal to [-0.95, 0.95]."""
    sig = sig - np.mean(sig)
    mx = np.max(np.abs(sig))
    if mx > 0:
        sig = sig / mx * 0.95
    return sig




# ============================================================
# SECTION 4: MMSE-LSA DENOISER (Ephraim & Malah 1985)
# ============================================================
def mmse_lsa_denoise(sig, fs, n_noise_frames=5, alpha_dd=0.98, floor_db=-30):
    """
    Minimum Mean Square Error Log-Spectral Amplitude (MMSE-LSA) estimator.
    Implements Ephraim & Malah (1985) for speech enhancement.

    Parameters:
        sig: input signal
        fs: sample rate
        n_noise_frames: number of initial frames used for noise estimation
        alpha_dd: smoothing factor for decision-directed a priori SNR
        floor_db: spectral floor in dB to prevent musical noise
    """
    # STFT parameters: 25ms frame, 10ms hop
    frame_len = max(4, int(0.025 * fs))
    hop_len = max(2, int(0.010 * fs))
    # Ensure frame_len is even for rfft
    if frame_len % 2 != 0:
        frame_len += 1

    n = len(sig)
    if n < frame_len:
        return sig

    window = np.hanning(frame_len)
    n_frames = (n - frame_len) // hop_len + 1
    if n_frames < n_noise_frames + 1:
        n_noise_frames = max(1, n_frames // 3)

    n_fft = frame_len
    n_bins = n_fft // 2 + 1

    # Compute STFT
    stft = np.zeros((n_frames, n_bins), dtype=complex)
    for i in range(n_frames):
        start = i * hop_len
        frame = sig[start:start + frame_len] * window
        stft[i] = np.fft.rfft(frame, n=n_fft)

    mag = np.abs(stft)
    phase = np.angle(stft)
    power = mag ** 2

    # Noise estimation from first n_noise_frames
    noise_power = np.mean(power[:n_noise_frames], axis=0) + 1e-10

    # Spectral floor
    floor_gain = 10 ** (floor_db / 20)

    # Decision-directed a priori SNR estimation and MMSE-LSA gain
    output_mag = np.zeros_like(mag)

    for i in range(n_frames):
        # A posteriori SNR
        gamma = power[i] / (noise_power + 1e-10)
        gamma = np.maximum(gamma, 1e-5)

        # Decision-directed a priori SNR
        if i == 0:
            xi = np.maximum(gamma - 1, 0.01)
        else:
            # Use previous frame gain-squared for DD estimate
            G_prev = output_mag[i - 1] / (mag[i - 1] + 1e-10)
            xi_ml = G_prev ** 2 * power[i - 1] / (noise_power + 1e-10)
            xi = alpha_dd * xi_ml + (1 - alpha_dd) * np.maximum(gamma - 1, 0)
            xi = np.maximum(xi, 0.01)

        # MMSE-LSA gain computation
        # v = xi / (1 + xi) * gamma
        v = (xi / (1 + xi)) * gamma
        v = np.minimum(v, 500)  # Clamp to avoid overflow

        # G = xi/(1+xi) * exp(0.5 * E1(v))
        # where E1 is the exponential integral (scipy.special.exp1)
        ei_v = exp1(np.maximum(v, 1e-10))
        G = (xi / (1 + xi)) * np.exp(0.5 * ei_v)
        G = np.maximum(G, floor_gain)
        G = np.minimum(G, 1.0)

        output_mag[i] = G * mag[i]

    # Reconstruct via inverse STFT (overlap-add)
    output = np.zeros(n)
    win_sum = np.zeros(n)
    for i in range(n_frames):
        start = i * hop_len
        frame_fft = output_mag[i] * np.exp(1j * phase[i])
        frame = np.fft.irfft(frame_fft, n=n_fft) * window
        end = min(start + frame_len, n)
        output[start:end] += frame[:end - start]
        win_sum[start:end] += window[:end - start] ** 2

    # Normalize
    nonzero = win_sum > 1e-8
    output[nonzero] /= win_sum[nonzero]

    return output




# ============================================================
# SECTION 5: ALL DENOISING METHODS (20+)
# ============================================================

def denoise_bandpass_1500(sig, fs):
    """Bandpass Butterworth 50-1500 Hz."""
    return apply_bandpass(sig, fs, low=50, high=1500, order=4)


def denoise_bandpass_2000(sig, fs):
    """Bandpass Butterworth 50-2000 Hz."""
    return apply_bandpass(sig, fs, low=50, high=2000, order=4)


def denoise_mmse_lsa(sig, fs):
    """MMSE-LSA denoiser (Ephraim & Malah 1985)."""
    return mmse_lsa_denoise(sig, fs)


def denoise_wavelet_db4(sig, fs):
    """Wavelet denoising: db4 with soft universal threshold."""
    n = len(sig)
    max_level = pywt.dwt_max_level(n, "db4")
    level = min(max_level, 5)
    if level < 1:
        return sig
    coeffs = pywt.wavedec(sig, "db4", level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(max(n, 2)))
    denoised_coeffs = [coeffs[0]]
    for c in coeffs[1:]:
        denoised_coeffs.append(pywt.threshold(c, threshold, mode="soft"))
    result = pywt.waverec(denoised_coeffs, "db4")
    return result[:n]


def denoise_wavelet_sym8_bayes(sig, fs):
    """Wavelet denoising: sym8 with BayesShrink adaptive threshold."""
    n = len(sig)
    wavelet = "sym8"
    max_level = pywt.dwt_max_level(n, wavelet)
    level = min(max_level, 5)
    if level < 1:
        return sig
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    denoised_coeffs = [coeffs[0]]
    for c in coeffs[1:]:
        sigma_y_sq = np.mean(c ** 2)
        sigma_x_sq = max(sigma_y_sq - sigma ** 2, 0)
        if sigma_x_sq == 0:
            thresh = np.max(np.abs(c))
        else:
            thresh = sigma ** 2 / np.sqrt(sigma_x_sq)
        denoised_coeffs.append(pywt.threshold(c, thresh, mode="soft"))
    result = pywt.waverec(denoised_coeffs, wavelet)
    return result[:n]


def denoise_wavelet_coif3(sig, fs):
    """Wavelet denoising: coif3 with soft universal threshold."""
    n = len(sig)
    wavelet = "coif3"
    max_level = pywt.dwt_max_level(n, wavelet)
    level = min(max_level, 5)
    if level < 1:
        return sig
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(max(n, 2)))
    denoised_coeffs = [coeffs[0]]
    for c in coeffs[1:]:
        denoised_coeffs.append(pywt.threshold(c, threshold, mode="soft"))
    result = pywt.waverec(denoised_coeffs, wavelet)
    return result[:n]


def denoise_kalman(sig, fs):
    """Kalman filter with constant velocity model."""
    n = len(sig)
    if n < 2:
        return sig
    x = np.array([sig[0], 0.0])
    P = np.eye(2)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.array([[1e-3, 0.0], [0.0, 1e-3]])
    R = np.array([[1.0]])
    filtered = np.zeros(n)
    filtered[0] = sig[0]
    for i in range(1, n):
        x = F @ x
        P = F @ P @ F.T + Q
        y_innov = sig[i] - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T / S[0, 0]
        x = x + K.flatten() * y_innov[0]
        P = (np.eye(2) - K @ H) @ P
        filtered[i] = x[0]
    return filtered


def denoise_bilateral_1d(sig, fs):
    """
    1D bilateral filter - edge-preserving smoothing.
    Skipped for signals longer than 50000 samples due to O(n * window)
    complexity of the pure-Python implementation.
    """
    n = len(sig)
    if n > 50000:
        # Too slow for long signals; fall back to a simple Gaussian smooth
        from scipy.ndimage import gaussian_filter1d as _gf1d
        return _gf1d(sig, sigma=5)
    sigma_d = 5
    sigma_r = np.std(sig) * 0.5
    if sigma_r < 1e-10:
        sigma_r = 1.0
    half_w = min(int(3 * sigma_d), n // 2)
    output = np.zeros(n)
    for i in range(n):
        start = max(0, i - half_w)
        end = min(n, i + half_w + 1)
        idxs = np.arange(start, end)
        spatial = np.exp(-0.5 * ((idxs - i) / sigma_d) ** 2)
        intensity = np.exp(-0.5 * ((sig[start:end] - sig[i]) / sigma_r) ** 2)
        weights = spatial * intensity
        ws = np.sum(weights)
        if ws > 0:
            output[i] = np.sum(weights * sig[start:end]) / ws
        else:
            output[i] = sig[i]
    return output


def denoise_total_variation(sig, fs):
    """Total Variation denoising - preserves edges while smoothing."""
    weight = 0.1
    iterations = 80
    output = sig.copy().astype(np.float64)
    for _ in range(iterations):
        diff = np.diff(output)
        grad = np.zeros_like(output)
        denom = np.abs(diff) + 1e-8
        grad[:-1] -= diff / denom
        grad[1:] += diff / denom
        output = output - weight * grad
        output = output + weight * (sig - output)
    return output


def denoise_lpc(sig, fs):
    """LPC-based enhancement (order 14). Smooths spectral envelope."""
    n = len(sig)
    order = min(14, n // 3)
    if order < 2:
        return sig

    # Compute autocorrelation
    autocorr = np.correlate(sig, sig, mode="full")
    autocorr = autocorr[n - 1:]
    autocorr = autocorr[:order + 1]

    if autocorr[0] == 0:
        return sig

    # Solve Toeplitz system using O(n^2) Levinson-Durbin via solve_toeplitz
    r = autocorr[1:order + 1]
    try:
        a = solve_toeplitz(autocorr[:order], r)
    except (np.linalg.LinAlgError, ValueError):
        return sig

    # LPC prediction using lfilter for speed (replaces O(n*order) Python loop)
    # The prediction filter: y[n] = a[0]*x[n-1] + a[1]*x[n-2] + ... + a[order-1]*x[n-order]
    # This is equivalent to filtering with b=a_coeffs, a=[1] shifted by 'order' samples
    predicted = np.zeros(n)
    # Use lfilter: output[n] = sum(a[k] * sig[n-1-k]) for k=0..order-1
    # Rewrite as FIR filter with coefficients a (reversed indexing already handled)
    b_fir = a  # FIR coefficients
    predicted[order:] = sp_signal.lfilter(b_fir, [1.0], sig)[order - 1:n - 1]

    # Blend prediction with original
    alpha = 0.7
    result = alpha * predicted + (1 - alpha) * sig
    result[:order] = sig[:order]
    return result


def denoise_adaptive_lms(sig, fs):
    """
    Adaptive LMS linear prediction error filter (whitening).
    Predicts the current sample from past samples and outputs the
    prediction residual. This emphasizes transients and removes
    predictable (periodic) content -- useful as a pre-processor
    but not a direct voice enhancer on its own.
    """
    n = len(sig)
    filter_order = min(32, n // 4)
    if filter_order < 2:
        return sig

    step_size = 0.01
    w = np.zeros(filter_order)
    output = np.zeros(n)
    output[:filter_order] = sig[:filter_order]

    for i in range(filter_order, n):
        x = sig[i - filter_order:i][::-1]
        y = np.dot(w, x)
        e = sig[i] - y
        output[i] = e
        norm_x = np.dot(x, x) + 1e-10
        w += 2 * step_size * e * x / norm_x
    return output


def denoise_multiband(sig, fs):
    """Multi-band decomposition: split into octave bands, wavelet denoise each."""
    n = len(sig)
    if n < 16:
        return sig

    nyq = fs / 2
    # Define band edges (octave-spaced in speech range)
    bands = [(50, 125), (125, 250), (250, 500), (500, 1000), (1000, min(2000, nyq * 0.95))]
    output = np.zeros(n)

    for low, high in bands:
        if low >= nyq or high > nyq:
            continue
        low_n = max(low / nyq, 0.001)
        high_n = min(high / nyq, 0.999)
        if low_n >= high_n:
            continue
        try:
            b, a = sp_signal.butter(3, [low_n, high_n], btype="band")
            band_sig = sp_signal.filtfilt(b, a, sig)
            # Wavelet denoise this band
            max_level = pywt.dwt_max_level(len(band_sig), "db4")
            level = min(max_level, 3)
            if level >= 1:
                coeffs = pywt.wavedec(band_sig, "db4", level=level)
                sigma = np.median(np.abs(coeffs[-1])) / 0.6745
                thresh = sigma * np.sqrt(2 * np.log(max(n, 2)))
                dc = [coeffs[0]]
                for c in coeffs[1:]:
                    dc.append(pywt.threshold(c, thresh, mode="soft"))
                band_sig = pywt.waverec(dc, "db4")[:n]
            output += band_sig
        except Exception as e:
            print(f"  [WARN] multiband: band {low}-{high} Hz failed: {e}")
            continue

    if np.max(np.abs(output)) < 1e-10:
        return sig
    return output


def denoise_spectral_gating(sig, fs):
    """Spectral gating with adaptive per-bin threshold from noise estimate."""
    n = len(sig)
    frame_len = max(4, min(256, n))
    hop = frame_len // 2
    window = np.hanning(frame_len)
    n_frames = (n - frame_len) // hop + 1
    if n_frames < 2:
        return sig

    n_bins = frame_len // 2 + 1
    stft = np.zeros((n_frames, n_bins), dtype=complex)
    for i in range(n_frames):
        start = i * hop
        frame = sig[start:start + frame_len] * window
        stft[i] = np.fft.rfft(frame, n=frame_len)

    mag = np.abs(stft)
    phase = np.angle(stft)

    # Noise estimate from quietest 20% of frames
    frame_energy = np.sum(mag ** 2, axis=1)
    n_noise = max(1, n_frames // 5)
    noise_idx = np.argsort(frame_energy)[:n_noise]
    noise_profile = np.mean(mag[noise_idx] ** 2, axis=0)

    # Adaptive threshold per bin
    threshold_factor = 2.0
    output_mag = np.zeros_like(mag)
    for i in range(n_frames):
        gate = mag[i] ** 2 > threshold_factor * noise_profile
        output_mag[i] = np.where(gate, mag[i], mag[i] * 0.1)

    # Inverse STFT
    output = np.zeros(n)
    win_sum = np.zeros(n)
    for i in range(n_frames):
        start = i * hop
        frame_fft = output_mag[i] * np.exp(1j * phase[i])
        frame = np.fft.irfft(frame_fft, n=frame_len) * window
        end = min(start + frame_len, n)
        output[start:end] += frame[:end - start]
        win_sum[start:end] += window[:end - start] ** 2
    nonzero = win_sum > 1e-8
    output[nonzero] /= win_sum[nonzero]
    return output


def denoise_wiener(sig, fs):
    """Wiener filter with noise estimation from high-frequency tail."""
    n = len(sig)
    sig_fft = np.fft.rfft(sig)
    power = np.abs(sig_fft) ** 2

    # Noise estimate from top 30% of spectrum
    high_start = max(1, int(len(sig_fft) * 0.7))
    noise_power = np.mean(power[high_start:])
    if noise_power < 1e-10:
        noise_power = np.mean(power) * 0.1

    wiener_gain = power / (power + noise_power + 1e-10)
    filtered_fft = sig_fft * wiener_gain
    return np.fft.irfft(filtered_fft, n=n)


def denoise_spectral_subtraction(sig, fs):
    """Spectral subtraction with noise estimate from first frames."""
    n = len(sig)
    frame_len = max(4, min(256, n))
    hop = frame_len // 2
    window = np.hanning(frame_len)
    n_frames = (n - frame_len) // hop + 1
    if n_frames < 2:
        return sig

    n_bins = frame_len // 2 + 1
    stft = np.zeros((n_frames, n_bins), dtype=complex)
    for i in range(n_frames):
        start = i * hop
        frame = sig[start:start + frame_len] * window
        stft[i] = np.fft.rfft(frame, n=frame_len)

    noise_frames = max(1, n_frames // 10)
    noise_spec = np.mean(np.abs(stft[:noise_frames]) ** 2, axis=0)

    alpha = 2.0
    beta = 0.02
    output_stft = np.zeros_like(stft)
    for i in range(n_frames):
        mag_sq = np.abs(stft[i]) ** 2
        phase = np.angle(stft[i])
        subtracted = mag_sq - alpha * noise_spec
        subtracted = np.maximum(subtracted, beta * mag_sq)
        output_stft[i] = np.sqrt(subtracted) * np.exp(1j * phase)

    output = np.zeros(n)
    win_sum = np.zeros(n)
    for i in range(n_frames):
        start = i * hop
        frame = np.fft.irfft(output_stft[i], n=frame_len) * window
        end = min(start + frame_len, n)
        output[start:end] += frame[:end - start]
        win_sum[start:end] += window[:end - start] ** 2
    nonzero = win_sum > 1e-8
    output[nonzero] /= win_sum[nonzero]
    return output




# --- Cascaded / Combined Methods ---

def denoise_cascade_wavelet_bp_mmse(sig, fs):
    """Cascaded: Wavelet db4 -> Bandpass -> MMSE-LSA."""
    s = denoise_wavelet_db4(sig, fs)
    s = apply_bandpass(s, fs, low=50, high=1500)
    s = mmse_lsa_denoise(s, fs)
    return s


def denoise_cascade_bp_mmse(sig, fs):
    """Cascaded: Bandpass 50-1500 Hz -> MMSE-LSA."""
    s = apply_bandpass(sig, fs, low=50, high=1500)
    s = mmse_lsa_denoise(s, fs)
    return s


def denoise_cascade_wavelet_mmse(sig, fs):
    """Cascaded: Wavelet sym8 BayesShrink -> MMSE-LSA."""
    s = denoise_wavelet_sym8_bayes(sig, fs)
    s = mmse_lsa_denoise(s, fs)
    return s


def denoise_cascade_multiband_mmse(sig, fs):
    """Cascaded: Multi-band decomposition -> MMSE-LSA."""
    s = denoise_multiband(sig, fs)
    s = mmse_lsa_denoise(s, fs)
    return s


def denoise_savgol_bandpass(sig, fs):
    """Savitzky-Golay smoothing + Bandpass."""
    win = min(21, len(sig))
    if win % 2 == 0:
        win -= 1
    if win < 5:
        return sig
    s = sp_signal.savgol_filter(sig, win, 2)
    s = apply_bandpass(s, fs, low=50, high=1500)
    return s


def denoise_elliptic_bandpass(sig, fs):
    """Elliptic (Cauer) bandpass 50-1500 Hz."""
    nyq = fs / 2
    high = min(1500, nyq * 0.95)
    low = 50
    low_n = max(low / nyq, 0.001)
    high_n = min(high / nyq, 0.999)
    if low_n >= high_n:
        return sig
    b, a = sp_signal.ellip(4, 0.5, 40, [low_n, high_n], btype="band")
    return sp_signal.filtfilt(b, a, sig)


def denoise_chebyshev_bandpass(sig, fs):
    """Chebyshev Type I bandpass 50-1500 Hz."""
    nyq = fs / 2
    high = min(1500, nyq * 0.95)
    low = 50
    low_n = max(low / nyq, 0.001)
    high_n = min(high / nyq, 0.999)
    if low_n >= high_n:
        return sig
    b, a = sp_signal.cheby1(4, 0.5, [low_n, high_n], btype="band")
    return sp_signal.filtfilt(b, a, sig)


# ============================================================
# METHODS REGISTRY
# ============================================================
def get_all_methods():
    """Return ordered dict of all denoising methods."""
    return {
        "bandpass_butter_50_1500": denoise_bandpass_1500,
        "bandpass_butter_50_2000": denoise_bandpass_2000,
        "mmse_lsa": denoise_mmse_lsa,
        "wavelet_db4_soft": denoise_wavelet_db4,
        "wavelet_sym8_bayes": denoise_wavelet_sym8_bayes,
        "wavelet_coif3_soft": denoise_wavelet_coif3,
        "kalman_filter": denoise_kalman,
        "bilateral_1d": denoise_bilateral_1d,
        "total_variation": denoise_total_variation,
        "lpc_enhancement": denoise_lpc,
        "adaptive_lms": denoise_adaptive_lms,
        "multiband_decomp": denoise_multiband,
        "spectral_gating": denoise_spectral_gating,
        "wiener_filter": denoise_wiener,
        "spectral_subtraction": denoise_spectral_subtraction,
        "cascade_wavelet_bp_mmse": denoise_cascade_wavelet_bp_mmse,
        "cascade_bp_mmse": denoise_cascade_bp_mmse,
        "cascade_wavelet_mmse": denoise_cascade_wavelet_mmse,
        "cascade_multiband_mmse": denoise_cascade_multiband_mmse,
        "savgol_bandpass": denoise_savgol_bandpass,
        "elliptic_bandpass": denoise_elliptic_bandpass,
        "chebyshev_bandpass": denoise_chebyshev_bandpass,
    }




# ============================================================
# SECTION 6: VOICE-LIKENESS METRICS
# ============================================================
def compute_hnr_db(sig, fs):
    """
    Harmonic-to-Noise Ratio using autocorrelation method.
    Looks for peaks in 50-500 Hz pitch range.
    """
    n = len(sig)
    if n < 4:
        return 0.0

    # Compute autocorrelation
    autocorr = np.correlate(sig, sig, mode="full")
    autocorr = autocorr[n - 1:]  # positive lags only
    autocorr = autocorr / (autocorr[0] + 1e-10)

    # Search for peak in pitch range 50-500 Hz
    min_lag = max(1, int(fs / 500))
    max_lag = min(n - 1, int(fs / 50))

    if min_lag >= max_lag or max_lag >= n:
        return 0.0

    search_region = autocorr[min_lag:max_lag + 1]
    if len(search_region) == 0:
        return 0.0

    peak_val = np.max(search_region)
    peak_val = np.clip(peak_val, -0.999, 0.999)

    if peak_val <= 0:
        return 0.0

    # HNR = 10 * log10(r / (1 - r))
    hnr = 10 * np.log10(peak_val / (1 - peak_val + 1e-10) + 1e-10)
    return float(hnr)


def compute_voiced_fraction(sig, fs, frame_ms=25):
    """
    Fraction of frames that appear voiced (have energy above threshold
    and periodic structure).
    """
    frame_len = max(2, int(fs * frame_ms / 1000))
    n = len(sig)
    n_frames = n // frame_len
    if n_frames == 0:
        return 0.0

    # Energy threshold: 20% of max frame energy
    frame_energies = np.array([
        np.sum(sig[i * frame_len:(i + 1) * frame_len] ** 2)
        for i in range(n_frames)
    ])
    max_energy = np.max(frame_energies) if np.max(frame_energies) > 0 else 1.0
    energy_threshold = 0.05 * max_energy

    voiced_count = 0
    for i in range(n_frames):
        frame = sig[i * frame_len:(i + 1) * frame_len]
        energy = np.sum(frame ** 2)
        if energy < energy_threshold:
            continue

        # Check for periodicity via zero-crossing rate
        zc = np.sum(np.abs(np.diff(np.sign(frame))) > 0) / (2 * len(frame))
        # Voiced speech typically has ZCR between 0.02 and 0.15
        if 0.01 <= zc <= 0.25:
            voiced_count += 1

    return voiced_count / n_frames


def compute_spectral_flatness(sig, fs):
    """
    Spectral flatness: geometric_mean(PSD) / arithmetic_mean(PSD).
    0 = tonal/harmonic, 1 = noise-like.
    """
    n = len(sig)
    nperseg = min(256, n)
    if nperseg < 4:
        return 1.0

    f, psd = sp_signal.welch(sig, fs=fs, nperseg=nperseg)
    psd = psd + 1e-10  # avoid log(0)

    log_mean = np.mean(np.log(psd))
    geo_mean = np.exp(log_mean)
    arith_mean = np.mean(psd)

    if arith_mean <= 0:
        return 1.0

    flatness = geo_mean / arith_mean
    return float(np.clip(flatness, 0, 1))


def compute_inband_ratio(sig, fs):
    """Power in 80-1000 Hz / total power."""
    n = len(sig)
    nperseg = min(256, n)
    if nperseg < 4:
        return 0.0

    f, psd = sp_signal.welch(sig, fs=fs, nperseg=nperseg)
    total_power = np.sum(psd)
    if total_power <= 0:
        return 0.0

    inband_mask = (f >= 80) & (f <= 1000)
    inband_power = np.sum(psd[inband_mask])
    return float(inband_power / total_power)


def compute_snr_db(sig, fs):
    """Signal power in speech band / estimated noise power."""
    n = len(sig)
    nperseg = min(256, n)
    if nperseg < 4:
        return 0.0

    f, psd = sp_signal.welch(sig, fs=fs, nperseg=nperseg)

    speech_mask = (f >= 80) & (f <= 1000)
    noise_mask = (f > 1200)

    speech_power = np.mean(psd[speech_mask]) if np.any(speech_mask) else 0
    noise_power = np.mean(psd[noise_mask]) if np.any(noise_mask) else 1e-10

    if noise_power <= 0:
        noise_power = 1e-10
    if speech_power <= 0:
        return 0.0

    return float(10 * np.log10(speech_power / noise_power))


def compute_spectral_centroid(sig, fs):
    """Spectral centroid: sum(f * PSD(f)) / sum(PSD(f))."""
    n = len(sig)
    nperseg = min(256, n)
    if nperseg < 4:
        return 0.0

    f, psd = sp_signal.welch(sig, fs=fs, nperseg=nperseg)
    total = np.sum(psd)
    if total <= 0:
        return 0.0

    centroid = np.sum(f * psd) / total
    return float(centroid)


def compute_crest_factor(sig):
    """Crest factor: peak / RMS."""
    rms = np.sqrt(np.mean(sig ** 2))
    if rms < 1e-10:
        return 1.0
    peak = np.max(np.abs(sig))
    return float(peak / rms)


def compute_zcr(sig, fs, frame_ms=25):
    """Mean zero-crossing rate across frames."""
    frame_len = max(2, int(fs * frame_ms / 1000))
    n = len(sig)
    n_frames = n // frame_len
    if n_frames == 0:
        return 0.0

    zcrs = []
    for i in range(n_frames):
        frame = sig[i * frame_len:(i + 1) * frame_len]
        zc = np.sum(np.abs(np.diff(np.sign(frame))) > 0) / (2 * max(len(frame) - 1, 1))
        zcrs.append(zc)

    return float(np.mean(zcrs))


def compute_all_metrics(sig, fs):
    """Compute all voice-likeness metrics for a signal."""
    return {
        "hnr_db": compute_hnr_db(sig, fs),
        "voiced_fraction": compute_voiced_fraction(sig, fs),
        "spectral_flatness": compute_spectral_flatness(sig, fs),
        "inband_ratio": compute_inband_ratio(sig, fs),
        "snr_db": compute_snr_db(sig, fs),
        "spectral_centroid": compute_spectral_centroid(sig, fs),
        "crest_factor": compute_crest_factor(sig),
        "zero_crossing_rate": compute_zcr(sig, fs),
    }


def normalize_metrics_across(all_results):
    """
    Apply min-max normalization across all methods for each metric.
    Returns updated results with normalized metrics.
    """
    if not all_results:
        return all_results

    metric_keys = ["hnr_db", "voiced_fraction", "spectral_flatness",
                   "inband_ratio", "snr_db", "spectral_centroid",
                   "crest_factor", "zero_crossing_rate"]

    # Collect all values for each metric
    metric_vals = {k: [] for k in metric_keys}
    for r in all_results:
        for k in metric_keys:
            metric_vals[k].append(r["metrics"].get(k, 0.0))

    # Normalize each metric
    for k in metric_keys:
        vals = np.array(metric_vals[k], dtype=np.float64)
        vmin = np.min(vals)
        vmax = np.max(vals)
        rng = vmax - vmin
        if rng < 1e-10:
            normed = np.ones_like(vals) * 0.5
        else:
            normed = (vals - vmin) / rng
        for i, r in enumerate(all_results):
            r["metrics_norm"] = r.get("metrics_norm", {})
            r["metrics_norm"][k] = float(normed[i])

    return all_results


def compute_composite_score(metrics_norm):
    """
    Compute composite blind score (0-100) from normalized metrics.
    Higher = more voice-like / better quality.

    NOTE: Scores are relative to the current run only. Because metrics are
    min-max normalized across methods in a single execution, the worst method
    always scores near 0 and the best near 100. Scores are not comparable
    across different runs, presets, or when the method set changes.
    """
    score = 0.0
    # HNR: higher is better
    score += METRIC_WEIGHTS["hnr_db"] * metrics_norm.get("hnr_db", 0.5)
    # Voiced fraction: higher is better
    score += METRIC_WEIGHTS["voiced_fraction"] * metrics_norm.get("voiced_fraction", 0.5)
    # Inband ratio: higher is better
    score += METRIC_WEIGHTS["inband_ratio"] * metrics_norm.get("inband_ratio", 0.5)
    # SNR: higher is better
    score += METRIC_WEIGHTS["snr_db"] * metrics_norm.get("snr_db", 0.5)
    # Spectral flatness: LOWER is better for voice (invert)
    score += METRIC_WEIGHTS["spectral_flatness_inv"] * (1.0 - metrics_norm.get("spectral_flatness", 0.5))
    # Spectral centroid: prefer moderate (not too high) - normalize to speech range
    score += METRIC_WEIGHTS["spectral_centroid_norm"] * (1.0 - metrics_norm.get("spectral_centroid", 0.5))
    # Crest factor: moderate is good, very high is clipping
    score += METRIC_WEIGHTS["crest_factor_norm"] * (1.0 - abs(metrics_norm.get("crest_factor", 0.5) - 0.5) * 2)
    # ZCR: lower is better for voiced speech (invert)
    score += METRIC_WEIGHTS["zcr_inv"] * (1.0 - metrics_norm.get("zero_crossing_rate", 0.5))

    return float(np.clip(score * 100, 0, 100))




# ============================================================
# SECTION 7: REPORT GENERATION
# ============================================================
def spectrogram_png_b64(sig, fs, title="Spectrogram"):
    """Generate spectrogram plot as base64 PNG."""
    fig, ax = plt.subplots(figsize=(6, 2.5))
    nperseg = max(4, min(256, len(sig) // 2))
    if nperseg < 4 or len(sig) < nperseg:
        ax.text(0.5, 0.5, "Signal too short", ha="center", va="center",
                transform=ax.transAxes, color="white")
        ax.set_facecolor("#1e1e1e")
        fig.patch.set_facecolor("#1e1e1e")
    else:
        f, t, Sxx = sp_signal.spectrogram(sig, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        ax.pcolormesh(t, f, Sxx_db, shading="gouraud", cmap="inferno")
        ax.set_ylabel("Freq (Hz)", fontsize=8, color="white")
        ax.set_xlabel("Time (s)", fontsize=8, color="white")
        ax.tick_params(colors="white", labelsize=7)
        ax.set_facecolor("#1e1e1e")
        fig.patch.set_facecolor("#1e1e1e")
    ax.set_title(title, fontsize=9, color="white")
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor="#1e1e1e", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def waveform_png_b64(sig, fs, title="Waveform"):
    """Generate waveform plot as base64 PNG."""
    fig, ax = plt.subplots(figsize=(6, 1.5))
    t = np.arange(len(sig)) / fs
    ax.plot(t, sig, linewidth=0.5, color="#00ff88")
    ax.set_xlabel("Time (s)", fontsize=8, color="white")
    ax.set_ylabel("Amplitude", fontsize=8, color="white")
    ax.set_title(title, fontsize=9, color="white")
    ax.tick_params(colors="white", labelsize=7)
    ax.set_facecolor("#1e1e1e")
    fig.patch.set_facecolor("#1e1e1e")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#555")
    ax.spines["left"].set_color("#555")
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor="#1e1e1e", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def wav_b64(sig, sr):
    """Encode signal to base64 WAV for HTML audio data URI."""
    data = wav_bytes(sig, sr)
    return base64.b64encode(data).decode("ascii")


def write_csv(all_results, output_path):
    """Write metrics CSV summary."""
    rows = []
    for r in all_results:
        row = {"method": r["method"], "composite_score": r.get("composite_score", 0)}
        row.update(r["metrics"])
        rows.append(row)
    df = pd.DataFrame(rows)
    # Sort by composite score
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "rank"
    df.to_csv(output_path)
    return df


def build_html(all_results, signals_dict, target_sr, output_path, input_info=""):
    """
    Build complete HTML report with dark theme, ranked table,
    spectrograms, waveforms, and embedded audio.
    """
    # Sort by composite score
    sorted_results = sorted(all_results, key=lambda x: x.get("composite_score", 0), reverse=True)

    # Build table rows
    table_rows = ""
    for rank, r in enumerate(sorted_results, 1):
        m = r["metrics"]
        score = r.get("composite_score", 0)
        highlight = ' class="best"' if rank == 1 else ""
        table_rows += f"""<tr{highlight}>
            <td>{rank}</td>
            <td>{r['method']}</td>
            <td>{score:.1f}</td>
            <td>{m.get('hnr_db', 0):.2f}</td>
            <td>{m.get('voiced_fraction', 0):.3f}</td>
            <td>{m.get('spectral_flatness', 0):.4f}</td>
            <td>{m.get('inband_ratio', 0):.3f}</td>
            <td>{m.get('snr_db', 0):.2f}</td>
            <td>{m.get('spectral_centroid', 0):.1f}</td>
            <td>{m.get('crest_factor', 0):.2f}</td>
            <td>{m.get('zero_crossing_rate', 0):.4f}</td>
        </tr>\n"""

    # Build top-N spectrograms and audio
    top_n = min(10, len(sorted_results))
    media_html = ""
    for i in range(top_n):
        r = sorted_results[i]
        method = r["method"]
        sig = signals_dict.get(method)
        if sig is None:
            continue

        spec_b64 = spectrogram_png_b64(sig, target_sr, title=f"{method}")
        wave_b64 = waveform_png_b64(sig, target_sr, title=f"{method}")
        audio_b64 = wav_b64(sig, target_sr)

        media_html += f"""
        <div class="method-card">
            <h3>#{i+1} - {method} (Score: {r.get('composite_score', 0):.1f})</h3>
            <div class="plots">
                <img src="data:image/png;base64,{wave_b64}" alt="waveform">
                <img src="data:image/png;base64,{spec_b64}" alt="spectrogram">
            </div>
            <audio controls>
                <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
            </audio>
        </div>
        """

    # Best method callout
    best = sorted_results[0] if sorted_results else {"method": "N/A", "composite_score": 0}
    best_html = f"""
    <div class="best-callout">
        <h2>BEST METHOD: {best['method']}</h2>
        <p>Composite Score: <strong>{best.get('composite_score', 0):.1f} / 100</strong></p>
    </div>
    """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>IMU Audio Analyser Report</title>
<style>
    body {{ background: #121212; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; padding: 20px; }}
    h1, h2, h3 {{ color: #00ff88; }}
    .info {{ color: #aaa; margin-bottom: 20px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 13px; }}
    th, td {{ border: 1px solid #333; padding: 6px 10px; text-align: center; }}
    th {{ background: #1e1e1e; color: #00ff88; }}
    tr:nth-child(even) {{ background: #1a1a2e; }}
    tr:hover {{ background: #16213e; }}
    tr.best {{ background: #0a3d0a; font-weight: bold; }}
    .best-callout {{ background: #0a3d0a; border: 2px solid #00ff88; border-radius: 8px;
                     padding: 20px; margin: 20px 0; text-align: center; }}
    .best-callout h2 {{ color: #00ff88; font-size: 24px; }}
    .method-card {{ background: #1e1e1e; border-radius: 8px; padding: 15px; margin: 15px 0;
                    border: 1px solid #333; }}
    .method-card h3 {{ margin-top: 0; }}
    .plots img {{ max-width: 100%; margin: 5px 0; border-radius: 4px; }}
    audio {{ width: 100%; margin: 10px 0; }}
</style>
</head>
<body>
<h1>IMU Audio Analyser Report</h1>
<p class="info">{input_info}</p>

{best_html}

<h2>Method Ranking (All Methods)</h2>
<table>
<thead>
<tr>
    <th>Rank</th><th>Method</th><th>Score</th><th>HNR (dB)</th>
    <th>Voiced Frac</th><th>Flatness</th><th>Inband</th>
    <th>SNR (dB)</th><th>Centroid</th><th>Crest</th><th>ZCR</th>
</tr>
</thead>
<tbody>
{table_rows}
</tbody>
</table>

<h2>Top {top_n} Methods - Detailed Analysis</h2>
{media_html}

</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
    return output_path




# ============================================================
# SECTION 8: MAIN PIPELINE AND CLI
# ============================================================
def load_data(input_file=None, axis="z"):
    """Load accelerometer data from CSV or embedded sample."""
    axis_map = {"x": 1, "y": 2, "z": 3}
    col_idx = axis_map.get(axis.lower(), 3)

    if input_file and os.path.exists(input_file):
        print(f"[*] Loading data from: {input_file}")
        data = np.genfromtxt(input_file, delimiter=",", skip_header=1)
    else:
        print("[*] Using embedded sample data (40 samples at ~3.3 kHz)")
        data = np.genfromtxt(StringIO(SAMPLE_CSV), delimiter=",", skip_header=1)

    timestamps = data[:, 0]
    accel = data[:, col_idx]

    # Remove NaN
    valid = ~(np.isnan(timestamps) | np.isnan(accel))
    timestamps = timestamps[valid]
    accel = accel[valid]

    return timestamps, accel


def run_pipeline(input_file=None, output_dir="imu_analyser_output",
                 target_sr=DEFAULT_TARGET_SR, axis="z", gap_mode="uniform",
                 preset="default", method_names=None):
    """Main processing pipeline."""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  IMU AUDIO ANALYSER - Advanced Bone Conduction Processing")
    print("=" * 70)

    # --- Load Data ---
    timestamps, raw_signal = load_data(input_file, axis=axis)
    print(f"  Samples loaded: {len(raw_signal)}")
    print(f"  Axis: {axis.upper()}")

    # --- Gap-Aware Resampling ---
    print(f"  Gap mode: {gap_mode}")
    signal, actual_sr = gap_aware_resample(timestamps, raw_signal, mode=gap_mode)
    print(f"  Inferred sample rate: {actual_sr:.1f} Hz")
    print(f"  Signal length after resampling: {len(signal)}")

    # --- Preprocessing ---
    signal = detrend_signal(signal)
    signal = norm_rms(signal, target_rms=0.1)

    fs = actual_sr
    print(f"  Target output SR: {target_sr} Hz")
    print("-" * 70)

    # --- Select Methods ---
    all_methods = get_all_methods()
    if method_names:
        methods = {k: v for k, v in all_methods.items() if k in method_names}
    elif preset == "quick":
        # Quick: subset of key methods
        quick_keys = [
            "bandpass_butter_50_1500", "mmse_lsa", "wavelet_db4_soft",
            "cascade_bp_mmse", "cascade_wavelet_bp_mmse", "spectral_gating",
        ]
        methods = {k: v for k, v in all_methods.items() if k in quick_keys}
    else:
        methods = all_methods

    print(f"  Running {len(methods)} denoising methods...")
    print("-" * 70)

    # --- Process Each Method ---
    all_results = []
    signals_output = {}

    for name, func in methods.items():
        try:
            denoised = func(signal, fs)
            # Ensure same length
            min_len = min(len(denoised), len(signal))
            denoised = denoised[:min_len]

            # RMS normalize
            denoised = norm_rms(denoised, target_rms=0.1)

            # Resample to target SR
            output_sig = resample_poly_to(denoised, int(round(fs)), target_sr)

            # Trim silence
            output_sig = trim_silence(output_sig, fs=target_sr)
            if len(output_sig) < 2:
                output_sig = resample_poly_to(denoised, int(round(fs)), target_sr)

            # Normalize for audio output
            output_sig = normalize_audio(output_sig)

            # Compute metrics on the output signal
            metrics = compute_all_metrics(output_sig, target_sr)

            # Save WAV
            wav_path = os.path.join(output_dir, f"{name}.wav")
            sf.write(wav_path, output_sig, target_sr, subtype="PCM_16")

            all_results.append({
                "method": name,
                "metrics": metrics,
                "wav_path": wav_path,
            })
            signals_output[name] = output_sig

            print(f"  [OK] {name:<35} HNR={metrics['hnr_db']:.1f} dB  "
                  f"Inband={metrics['inband_ratio']:.3f}  "
                  f"SNR={metrics['snr_db']:.1f} dB")

        except Exception as e:
            print(f"  [FAIL] {name:<35} {str(e)[:60]}")

    # --- Grid Sweep for 'full' preset ---
    if preset == "full":
        print("\n  [Grid Sweep] Testing gap_mode variants...")
        extra_modes = [m for m in ["chunk", "uniform", "all"] if m != gap_mode]
        for gm in extra_modes:
            try:
                sig_variant, fs_variant = gap_aware_resample(timestamps, raw_signal, mode=gm)
                sig_variant = detrend_signal(sig_variant)
                sig_variant = norm_rms(sig_variant, target_rms=0.1)
                # Run top methods on this variant
                for mkey in ["mmse_lsa", "cascade_bp_mmse", "cascade_wavelet_bp_mmse"]:
                    if mkey in all_methods:
                        try:
                            d = all_methods[mkey](sig_variant, fs_variant)
                            d = d[:len(sig_variant)]
                            d = norm_rms(d, target_rms=0.1)
                            out = resample_poly_to(d, int(round(fs_variant)), target_sr)
                            out = trim_silence(out, fs=target_sr)
                            if len(out) < 2:
                                out = resample_poly_to(d, int(round(fs_variant)), target_sr)
                            out = normalize_audio(out)
                            metrics = compute_all_metrics(out, target_sr)
                            variant_name = f"{mkey}__gap_{gm}"
                            wav_path = os.path.join(output_dir, f"{variant_name}.wav")
                            sf.write(wav_path, out, target_sr, subtype="PCM_16")
                            all_results.append({
                                "method": variant_name,
                                "metrics": metrics,
                                "wav_path": wav_path,
                            })
                            signals_output[variant_name] = out
                            print(f"  [OK] {variant_name:<35} HNR={metrics['hnr_db']:.1f} dB")
                        except Exception as e:
                            print(f"  [WARN] grid sweep {mkey} (gap={gm}) failed: {e}")
            except Exception as e:
                print(f"  [WARN] grid sweep gap_mode={gm} failed: {e}")

    if not all_results:
        print("\n  [ERROR] No methods produced output.")
        return

    # --- Normalize Metrics Across All Methods ---
    all_results = normalize_metrics_across(all_results)

    # --- Compute Composite Scores ---
    for r in all_results:
        r["composite_score"] = compute_composite_score(r.get("metrics_norm", {}))

    # --- Sort and Rank ---
    all_results.sort(key=lambda x: x["composite_score"], reverse=True)

    print("\n" + "=" * 70)
    print("  RESULTS RANKED BY COMPOSITE SCORE (0-100)")
    print("=" * 70)
    for rank, r in enumerate(all_results, 1):
        star = " *** BEST ***" if rank == 1 else ""
        print(f"  {rank:>3}. {r['method']:<40} Score: {r['composite_score']:.1f}{star}")

    # --- Write CSV ---
    csv_path = os.path.join(output_dir, "metrics.csv")
    write_csv(all_results, csv_path)
    print(f"\n  CSV saved: {csv_path}")

    # --- Write HTML Report ---
    input_info = f"Input: {input_file or 'embedded sample'} | Axis: {axis} | "
    input_info += f"Gap mode: {gap_mode} | Target SR: {target_sr} Hz | "
    input_info += f"Methods: {len(all_results)} | Preset: {preset}"

    html_path = os.path.join(output_dir, "report.html")
    build_html(all_results, signals_output, target_sr, html_path, input_info)
    print(f"  HTML report: {html_path}")

    print(f"\n  Output directory: {os.path.abspath(output_dir)}")
    print(f"  Total WAV files: {len(all_results)}")
    best = all_results[0]
    print(f"\n  >>> BEST METHOD: {best['method']} (Score: {best['composite_score']:.1f}/100) <<<")
    print("=" * 70)

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="IMU Audio Analyser - Advanced Bone Conduction Signal Processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python imu_audio_analyser.py                             # Embedded sample
  python imu_audio_analyser.py -i data.csv -o output/     # CSV input
  python imu_audio_analyser.py --preset full              # Full grid sweep
  python imu_audio_analyser.py --gap_mode chunk --axis z  # Gap-aware chunk mode
        """
    )
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Input CSV file (timestamp, Accel X, Accel Y, Accel Z)")
    parser.add_argument("--output_dir", "-o", type=str, default="imu_analyser_output",
                        help="Output directory (default: imu_analyser_output)")
    parser.add_argument("--target_sr", type=int, default=DEFAULT_TARGET_SR,
                        help=f"Target sample rate (default: {DEFAULT_TARGET_SR})")
    parser.add_argument("--axis", type=str, default="z", choices=["x", "y", "z"],
                        help="Accelerometer axis to use (default: z)")
    parser.add_argument("--gap_mode", type=str, default="uniform",
                        choices=["chunk", "uniform", "all"],
                        help="Gap-aware resampling mode (default: uniform)")
    parser.add_argument("--preset", type=str, default="default",
                        choices=["quick", "default", "full"],
                        help="Preset: quick (6 methods), default (22), full (grid sweep)")
    parser.add_argument("--methods", type=str, nargs="*", default=None,
                        help="Specific method names to run")

    args = parser.parse_args()

    run_pipeline(
        input_file=args.input,
        output_dir=args.output_dir,
        target_sr=args.target_sr,
        axis=args.axis,
        gap_mode=args.gap_mode,
        preset=args.preset,
        method_names=args.methods,
    )


if __name__ == "__main__":
    main()
