#!/usr/bin/env python3
"""
Bone Conduction IMU Signal Denoising and Audio Conversion
=========================================================
Applies multiple denoising methods to accelerometer Z-axis data
captured at 3.3 kHz from a bone conduction microphone.

Usage:
    python bone_conduction_denoising.py                  # Use embedded sample data
    python bone_conduction_denoising.py --input data.csv # Use CSV file
    python bone_conduction_denoising.py --output_dir out # Specify output directory
    python bone_conduction_denoising.py --target_sr 16000 # Upsample to 16kHz
"""

import argparse
import os
import sys
import warnings
from io import StringIO

import numpy as np
import pywt
import soundfile as sf
from scipy import signal as sp_signal
from scipy.ndimage import median_filter

warnings.filterwarnings("ignore")

# ============================================================
# EMBEDDED SAMPLE DATA (Accel Z column from bone conduction IMU)
# ============================================================
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

# Original sampling rate of the IMU
ORIGINAL_SR = 3300  # 3.3 kHz

# Default target sample rate for audio output
DEFAULT_TARGET_SR = 16000  # 16 kHz (common for speech)

# Available target sample rates
AVAILABLE_TARGET_RATES = [8000, 16000, 22050, 44100, 48000]


# ============================================================
# DATA LOADING
# ============================================================
def load_data(input_file=None):
    """Load accelerometer Z data from CSV file or embedded sample."""
    if input_file and os.path.exists(input_file):
        print(f"Loading data from: {input_file}")
        data = np.genfromtxt(input_file, delimiter=",", skip_header=1)
    else:
        print("Using embedded sample data")
        data = np.genfromtxt(StringIO(SAMPLE_CSV), delimiter=",", skip_header=1)

    # Extract Accel Z (4th column, index 3)
    accel_z = data[:, 3]

    # Remove any NaN values
    accel_z = accel_z[~np.isnan(accel_z)]

    print(f"  Loaded {len(accel_z)} samples at {ORIGINAL_SR} Hz")
    print(f"  Duration: {len(accel_z)/ORIGINAL_SR:.4f} seconds")
    print(f"  Raw range: [{accel_z.min():.1f}, {accel_z.max():.1f}]")

    return accel_z


def normalize_signal(sig):
    """Normalize signal to [-1, 1] range for audio output."""
    sig = sig - np.mean(sig)  # Remove DC offset
    max_val = np.max(np.abs(sig))
    if max_val > 0:
        sig = sig / max_val
    return sig * 0.95  # Leave small headroom


def upsample_signal(sig, original_sr, target_sr):
    """Upsample signal from original_sr to target_sr using polyphase resampling."""
    if target_sr == original_sr:
        return sig
    num_samples = int(len(sig) * target_sr / original_sr)
    resampled = sp_signal.resample_poly(sig, target_sr, original_sr)
    return resampled


# ============================================================
# SNR COMPUTATION
# ============================================================
def compute_snr(original, denoised):
    """
    Compute SNR improvement estimate.
    Uses the assumption that noise = original - denoised.
    Returns SNR in dB.
    """
    # Remove DC from both
    orig = original - np.mean(original)
    den = denoised - np.mean(denoised)

    # Noise estimate
    noise = orig - den

    signal_power = np.mean(den ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return float("inf")

    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db


def compute_original_snr(sig):
    """
    Estimate original SNR using spectral method.
    Assumes speech is below 1500 Hz for bone conduction.
    """
    sig = sig - np.mean(sig)
    f, psd = sp_signal.welch(sig, fs=ORIGINAL_SR, nperseg=min(256, len(sig)))

    # Speech band: 50-1500 Hz for bone conduction
    speech_mask = (f >= 50) & (f <= 1500)
    noise_mask = f > 1500

    if np.sum(speech_mask) == 0 or np.sum(noise_mask) == 0:
        return 0.0

    signal_power = np.mean(psd[speech_mask])
    noise_power = np.mean(psd[noise_mask])

    if noise_power == 0:
        return float("inf")

    return 10 * np.log10(signal_power / noise_power)


# ============================================================
# DENOISING METHODS
# ============================================================

def denoise_lowpass_butterworth(sig, fs, cutoff=1500, order=5):
    """Low-pass Butterworth filter - removes high frequency noise."""
    nyq = fs / 2
    normalized_cutoff = min(cutoff / nyq, 0.99)
    b, a = sp_signal.butter(order, normalized_cutoff, btype='low')
    return sp_signal.filtfilt(b, a, sig)


def denoise_highpass_butterworth(sig, fs, cutoff=50, order=4):
    """High-pass Butterworth filter - removes DC drift and very low freq noise."""
    nyq = fs / 2
    normalized_cutoff = min(cutoff / nyq, 0.99)
    b, a = sp_signal.butter(order, normalized_cutoff, btype='high')
    return sp_signal.filtfilt(b, a, sig)


def denoise_bandpass_butterworth(sig, fs, low=50, high=1500, order=4):
    """Band-pass Butterworth filter - keeps speech frequencies only."""
    nyq = fs / 2
    low_norm = max(low / nyq, 0.01)
    high_norm = min(high / nyq, 0.99)
    b, a = sp_signal.butter(order, [low_norm, high_norm], btype='band')
    return sp_signal.filtfilt(b, a, sig)


def denoise_bandpass_butterworth_wide(sig, fs, low=20, high=1650, order=3):
    """Wide band-pass Butterworth - broader speech range."""
    nyq = fs / 2
    low_norm = max(low / nyq, 0.01)
    high_norm = min(high / nyq, 0.99)
    b, a = sp_signal.butter(order, [low_norm, high_norm], btype='band')
    return sp_signal.filtfilt(b, a, sig)


def denoise_chebyshev_type1_lowpass(sig, fs, cutoff=1500, order=5, ripple=0.5):
    """Chebyshev Type I low-pass - sharper roll-off with passband ripple."""
    nyq = fs / 2
    normalized_cutoff = min(cutoff / nyq, 0.99)
    b, a = sp_signal.cheby1(order, ripple, normalized_cutoff, btype='low')
    return sp_signal.filtfilt(b, a, sig)


def denoise_chebyshev_type2_lowpass(sig, fs, cutoff=1500, order=5, rs=40):
    """Chebyshev Type II low-pass - flat passband, stopband ripple."""
    nyq = fs / 2
    normalized_cutoff = min(cutoff / nyq, 0.99)
    b, a = sp_signal.cheby2(order, rs, normalized_cutoff, btype='low')
    return sp_signal.filtfilt(b, a, sig)


def denoise_chebyshev_bandpass(sig, fs, low=50, high=1500, order=4, ripple=0.5):
    """Chebyshev Type I band-pass filter."""
    nyq = fs / 2
    low_norm = max(low / nyq, 0.01)
    high_norm = min(high / nyq, 0.99)
    b, a = sp_signal.cheby1(order, ripple, [low_norm, high_norm], btype='band')
    return sp_signal.filtfilt(b, a, sig)


def denoise_elliptic_lowpass(sig, fs, cutoff=1500, order=5, rp=0.5, rs=40):
    """Elliptic (Cauer) low-pass - sharpest roll-off."""
    nyq = fs / 2
    normalized_cutoff = min(cutoff / nyq, 0.99)
    b, a = sp_signal.ellip(order, rp, rs, normalized_cutoff, btype='low')
    return sp_signal.filtfilt(b, a, sig)


def denoise_elliptic_bandpass(sig, fs, low=50, high=1500, order=4, rp=0.5, rs=40):
    """Elliptic (Cauer) band-pass filter."""
    nyq = fs / 2
    low_norm = max(low / nyq, 0.01)
    high_norm = min(high / nyq, 0.99)
    b, a = sp_signal.ellip(order, rp, rs, [low_norm, high_norm], btype='band')
    return sp_signal.filtfilt(b, a, sig)


def denoise_bessel_lowpass(sig, fs, cutoff=1500, order=4):
    """Bessel low-pass filter - linear phase response, preserves waveform shape."""
    nyq = fs / 2
    normalized_cutoff = min(cutoff / nyq, 0.99)
    b, a = sp_signal.bessel(order, normalized_cutoff, btype='low', norm='phase')
    return sp_signal.filtfilt(b, a, sig)


def denoise_moving_average(sig, window_size=11):
    """Simple moving average filter."""
    kernel = np.ones(window_size) / window_size
    return np.convolve(sig, kernel, mode='same')


def denoise_weighted_moving_average(sig, window_size=11):
    """Weighted moving average (triangular window)."""
    weights = np.arange(1, window_size + 1, dtype=float)
    weights = np.concatenate([weights, weights[-2::-1]])
    weights = weights / weights.sum()
    return np.convolve(sig, weights, mode='same')


def denoise_exponential_moving_average(sig, alpha=0.3):
    """Exponential moving average (EMA) filter."""
    result = np.zeros_like(sig)
    result[0] = sig[0]
    for i in range(1, len(sig)):
        result[i] = alpha * sig[i] + (1 - alpha) * result[i - 1]
    return result


def denoise_savitzky_golay(sig, window_length=11, polyorder=3):
    """Savitzky-Golay filter - polynomial smoothing."""
    # Window length must be odd and > polyorder
    if window_length % 2 == 0:
        window_length += 1
    if window_length <= polyorder:
        window_length = polyorder + 2
        if window_length % 2 == 0:
            window_length += 1
    return sp_signal.savgol_filter(sig, window_length, polyorder)


def denoise_savitzky_golay_aggressive(sig, window_length=21, polyorder=2):
    """Savitzky-Golay filter with larger window - more smoothing."""
    if window_length % 2 == 0:
        window_length += 1
    return sp_signal.savgol_filter(sig, window_length, polyorder)


def denoise_wavelet_db4(sig, level=None, threshold_mode='soft'):
    """Wavelet denoising using Daubechies-4 wavelet."""
    if level is None:
        level = min(pywt.dwt_max_level(len(sig), 'db4'), 5)
    coeffs = pywt.wavedec(sig, 'db4', level=level)

    # Universal threshold (VisuShrink)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(sig)))

    # Apply threshold to detail coefficients
    denoised_coeffs = [coeffs[0]]  # Keep approximation
    for c in coeffs[1:]:
        denoised_coeffs.append(pywt.threshold(c, threshold, mode=threshold_mode))

    return pywt.waverec(denoised_coeffs, 'db4')[:len(sig)]


def denoise_wavelet_sym8(sig, level=None, threshold_mode='soft'):
    """Wavelet denoising using Symlet-8 wavelet."""
    if level is None:
        level = min(pywt.dwt_max_level(len(sig), 'sym8'), 5)
    coeffs = pywt.wavedec(sig, 'sym8', level=level)

    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(sig)))

    denoised_coeffs = [coeffs[0]]
    for c in coeffs[1:]:
        denoised_coeffs.append(pywt.threshold(c, threshold, mode=threshold_mode))

    return pywt.waverec(denoised_coeffs, 'sym8')[:len(sig)]


def denoise_wavelet_coif3(sig, level=None, threshold_mode='soft'):
    """Wavelet denoising using Coiflet-3 wavelet."""
    if level is None:
        level = min(pywt.dwt_max_level(len(sig), 'coif3'), 5)
    coeffs = pywt.wavedec(sig, 'coif3', level=level)

    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(sig)))

    denoised_coeffs = [coeffs[0]]
    for c in coeffs[1:]:
        denoised_coeffs.append(pywt.threshold(c, threshold, mode=threshold_mode))

    return pywt.waverec(denoised_coeffs, 'coif3')[:len(sig)]


def denoise_wavelet_hard_threshold(sig, level=None):
    """Wavelet denoising with hard thresholding."""
    return denoise_wavelet_db4(sig, level=level, threshold_mode='hard')


def denoise_wavelet_bayes_shrink(sig, wavelet='db4', level=None):
    """Wavelet denoising using BayesShrink (adaptive threshold per level)."""
    if level is None:
        level = min(pywt.dwt_max_level(len(sig), wavelet), 5)
    coeffs = pywt.wavedec(sig, wavelet, level=level)

    # BayesShrink: adaptive threshold for each level
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745

    denoised_coeffs = [coeffs[0]]
    for c in coeffs[1:]:
        # Estimate signal variance at this level
        sigma_y_sq = np.mean(c ** 2)
        sigma_x_sq = max(sigma_y_sq - sigma ** 2, 0)
        if sigma_x_sq == 0:
            threshold = np.max(np.abs(c))
        else:
            threshold = sigma ** 2 / np.sqrt(sigma_x_sq)
        denoised_coeffs.append(pywt.threshold(c, threshold, mode='soft'))

    return pywt.waverec(denoised_coeffs, wavelet)[:len(sig)]


def denoise_wiener(sig, noise_power=None):
    """Wiener filter in frequency domain."""
    n = len(sig)
    sig_fft = np.fft.rfft(sig)
    power_spectrum = np.abs(sig_fft) ** 2

    if noise_power is None:
        # Estimate noise from high frequency content
        high_freq_start = max(1, int(len(sig_fft) * 0.7))
        noise_power = np.mean(power_spectrum[high_freq_start:])
        if noise_power == 0:
            noise_power = np.mean(power_spectrum) * 0.1  # Fallback

    # Wiener filter transfer function
    wiener_filter = power_spectrum / (power_spectrum + noise_power + 1e-10)
    filtered_fft = sig_fft * wiener_filter

    return np.fft.irfft(filtered_fft, n=n)


def denoise_median_filter(sig, kernel_size=5):
    """Median filter - good for impulsive noise."""
    return median_filter(sig, size=kernel_size)


def denoise_median_filter_large(sig, kernel_size=11):
    """Median filter with larger kernel."""
    return median_filter(sig, size=kernel_size)


def denoise_kalman(sig, process_noise=1e-3, measurement_noise=1.0):
    """
    Simple 1D Kalman filter.
    Assumes constant velocity model.
    """
    n = len(sig)
    # State: [position, velocity]
    x = np.array([sig[0], 0.0])
    P = np.eye(2) * 1.0

    # State transition
    F = np.array([[1, 1], [0, 1]])
    # Measurement matrix
    H = np.array([[1, 0]])
    # Process noise
    Q = np.array([[process_noise, 0], [0, process_noise]])
    # Measurement noise
    R = np.array([[measurement_noise]])

    filtered = np.zeros(n)
    filtered[0] = sig[0]

    for i in range(1, n):
        # Predict
        x = F @ x
        P = F @ P @ F.T + Q

        # Update
        y = sig[i] - H @ x  # Innovation
        S = H @ P @ H.T + R  # Innovation covariance
        K = P @ H.T @ np.linalg.inv(S)  # Kalman gain

        x = x + (K @ np.array([y])).flatten()
        P = (np.eye(2) - K @ H) @ P

        filtered[i] = x[0]

    return filtered


def denoise_kalman_smooth(sig, process_noise=1e-4, measurement_noise=0.5):
    """Kalman filter with more smoothing (lower process noise)."""
    return denoise_kalman(sig, process_noise, measurement_noise)


def denoise_spectral_subtraction(sig, fs):
    """
    Spectral subtraction - estimates noise spectrum and subtracts it.
    Assumes first 10% of signal is noise-only or representative.
    """
    n = len(sig)
    # Use overlap-add with short windows
    win_len = min(256, n)
    hop = win_len // 2
    window = np.hanning(win_len)

    # Estimate noise from beginning of signal
    noise_frames = max(1, int(0.1 * n / hop))

    # STFT
    num_frames = (n - win_len) // hop + 1
    if num_frames < 2:
        return sig

    stft = np.zeros((num_frames, win_len // 2 + 1), dtype=complex)
    for i in range(num_frames):
        start = i * hop
        frame = sig[start:start + win_len] * window
        stft[i] = np.fft.rfft(frame)

    # Noise estimate from first few frames
    noise_spec = np.mean(np.abs(stft[:noise_frames]) ** 2, axis=0)

    # Spectral subtraction with flooring
    alpha = 2.0  # Over-subtraction factor
    beta = 0.01  # Spectral floor

    output_stft = np.zeros_like(stft)
    for i in range(num_frames):
        mag = np.abs(stft[i]) ** 2
        phase = np.angle(stft[i])
        subtracted = mag - alpha * noise_spec
        subtracted = np.maximum(subtracted, beta * mag)
        output_stft[i] = np.sqrt(subtracted) * np.exp(1j * phase)

    # Inverse STFT (overlap-add)
    output = np.zeros(n)
    window_sum = np.zeros(n)
    for i in range(num_frames):
        start = i * hop
        frame = np.fft.irfft(output_stft[i], n=win_len) * window
        end = min(start + win_len, n)
        output[start:end] += frame[:end - start]
        window_sum[start:end] += window[:end - start] ** 2

    # Normalize by window sum
    nonzero = window_sum > 1e-8
    output[nonzero] /= window_sum[nonzero]

    return output


def denoise_spectral_gating(sig, fs, threshold_db=-20):
    """
    Spectral gating - zeros out frequency bins below threshold.
    Like a noise gate but in frequency domain.
    """
    n = len(sig)
    sig_fft = np.fft.rfft(sig)
    magnitude = np.abs(sig_fft)
    phase = np.angle(sig_fft)

    # Threshold in linear scale
    max_mag = np.max(magnitude)
    threshold = max_mag * 10 ** (threshold_db / 20)

    # Gate: zero out bins below threshold
    gated_magnitude = np.where(magnitude > threshold, magnitude, magnitude * 0.1)

    result_fft = gated_magnitude * np.exp(1j * phase)
    return np.fft.irfft(result_fft, n=n)


def denoise_notch_filter(sig, fs, freq=50, Q=30):
    """Notch filter to remove specific frequency (e.g., power line interference)."""
    nyq = fs / 2
    w0 = freq / nyq
    if w0 >= 1.0:
        return sig
    b, a = sp_signal.iirnotch(w0, Q)
    return sp_signal.filtfilt(b, a, sig)


def denoise_adaptive_lms(sig, step_size=0.01, filter_order=32):
    """
    Adaptive LMS (Least Mean Squares) noise cancellation.
    Uses delayed version of signal as reference.
    """
    n = len(sig)
    if n <= filter_order:
        return sig

    w = np.zeros(filter_order)
    output = np.zeros(n)

    for i in range(filter_order, n):
        x = sig[i - filter_order:i][::-1]  # Reference signal (delayed input)
        y = np.dot(w, x)
        e = sig[i] - y
        output[i] = e
        w += 2 * step_size * e * x

    # Fill initial samples
    output[:filter_order] = sig[:filter_order]
    return output


def denoise_gaussian_smooth(sig, sigma=2.0):
    """Gaussian smoothing filter."""
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(sig, sigma=sigma)


def denoise_bilateral_1d(sig, sigma_d=5, sigma_r=None):
    """
    1D bilateral filter - edge-preserving smoothing.
    sigma_d: spatial sigma, sigma_r: range sigma.
    """
    if sigma_r is None:
        sigma_r = np.std(sig) * 0.5

    n = len(sig)
    half_w = int(3 * sigma_d)
    output = np.zeros(n)

    for i in range(n):
        start = max(0, i - half_w)
        end = min(n, i + half_w + 1)

        # Spatial weights
        spatial = np.exp(-0.5 * ((np.arange(start, end) - i) / sigma_d) ** 2)
        # Range weights
        intensity = np.exp(-0.5 * ((sig[start:end] - sig[i]) / sigma_r) ** 2)

        weights = spatial * intensity
        weights_sum = np.sum(weights)
        if weights_sum > 0:
            output[i] = np.sum(weights * sig[start:end]) / weights_sum
        else:
            output[i] = sig[i]

    return output


def denoise_total_variation(sig, weight=0.1, iterations=100):
    """Total Variation denoising - preserves edges while smoothing."""
    output = sig.copy().astype(float)
    for _ in range(iterations):
        diff = np.diff(output)
        # Gradient of TV regularization
        grad = np.zeros_like(output)
        grad[:-1] -= diff / (np.abs(diff) + 1e-8)
        grad[1:] += diff / (np.abs(diff) + 1e-8)
        output = output - weight * grad
        # Data fidelity
        output = output + weight * (sig - output)
    return output


def denoise_fft_frequency_filter(sig, fs, low_cut=50, high_cut=1500):
    """Direct FFT-based band-pass filtering."""
    n = len(sig)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    sig_fft = np.fft.rfft(sig)

    # Create smooth band-pass mask
    mask = np.zeros_like(freqs)
    for i, f in enumerate(freqs):
        if low_cut <= f <= high_cut:
            mask[i] = 1.0
        elif f < low_cut and f > low_cut * 0.5:
            mask[i] = (f - low_cut * 0.5) / (low_cut * 0.5)
        elif f > high_cut and f < high_cut * 1.2:
            mask[i] = 1.0 - (f - high_cut) / (high_cut * 0.2)

    filtered_fft = sig_fft * mask
    return np.fft.irfft(filtered_fft, n=n)


def denoise_lpc_residual(sig, order=12):
    """
    Linear Predictive Coding (LPC) based enhancement.
    Reconstructs signal from LPC coefficients, reducing noise.
    """
    from scipy.linalg import toeplitz, solve

    n = len(sig)
    if n <= order + 1:
        return sig

    # Compute autocorrelation
    autocorr = np.correlate(sig, sig, mode='full')
    autocorr = autocorr[n - 1:]  # Take positive lags
    autocorr = autocorr[:order + 1]

    # Solve Yule-Walker equations
    r = autocorr[1:order + 1]
    R = toeplitz(autocorr[:order])

    try:
        a = solve(R, r)
    except np.linalg.LinAlgError:
        return sig

    # Filter signal with inverse LPC filter then re-synthesize
    # This smooths the spectral envelope
    predicted = np.zeros(n)
    for i in range(order, n):
        predicted[i] = np.dot(a, sig[i - order:i][::-1])

    # Mix prediction with original (reduces noise)
    alpha = 0.7  # Blending factor
    return alpha * predicted + (1 - alpha) * sig


def denoise_combined_wavelet_butterworth(sig, fs):
    """Combined: Wavelet denoising followed by Butterworth band-pass."""
    # First: wavelet denoising
    wavelet_out = denoise_wavelet_db4(sig)
    # Then: band-pass filter
    return denoise_bandpass_butterworth(wavelet_out, fs)


def denoise_combined_kalman_bandpass(sig, fs):
    """Combined: Kalman filter followed by band-pass."""
    kalman_out = denoise_kalman(sig)
    return denoise_bandpass_butterworth(kalman_out, fs)



# ============================================================
# REGISTRY OF ALL DENOISING METHODS
# ============================================================
def get_all_methods():
    """Return a dictionary of all denoising methods."""
    return {
        # IIR Filters
        "01_lowpass_butterworth_1500Hz": lambda s, fs: denoise_lowpass_butterworth(s, fs, cutoff=1500),
        "02_lowpass_butterworth_1000Hz": lambda s, fs: denoise_lowpass_butterworth(s, fs, cutoff=1000),
        "03_highpass_butterworth_50Hz": lambda s, fs: denoise_highpass_butterworth(s, fs, cutoff=50),
        "04_bandpass_butterworth_50_1500Hz": lambda s, fs: denoise_bandpass_butterworth(s, fs, low=50, high=1500),
        "05_bandpass_butterworth_wide_20_1650Hz": lambda s, fs: denoise_bandpass_butterworth_wide(s, fs),
        "06_chebyshev1_lowpass_1500Hz": lambda s, fs: denoise_chebyshev_type1_lowpass(s, fs),
        "07_chebyshev2_lowpass_1500Hz": lambda s, fs: denoise_chebyshev_type2_lowpass(s, fs),
        "08_chebyshev1_bandpass_50_1500Hz": lambda s, fs: denoise_chebyshev_bandpass(s, fs),
        "09_elliptic_lowpass_1500Hz": lambda s, fs: denoise_elliptic_lowpass(s, fs),
        "10_elliptic_bandpass_50_1500Hz": lambda s, fs: denoise_elliptic_bandpass(s, fs),
        "11_bessel_lowpass_1500Hz": lambda s, fs: denoise_bessel_lowpass(s, fs),
        # Smoothing Filters
        "12_moving_average_5": lambda s, fs: denoise_moving_average(s, window_size=5),
        "13_moving_average_11": lambda s, fs: denoise_moving_average(s, window_size=11),
        "14_moving_average_21": lambda s, fs: denoise_moving_average(s, window_size=21),
        "15_weighted_moving_average": lambda s, fs: denoise_weighted_moving_average(s),
        "16_exponential_moving_average": lambda s, fs: denoise_exponential_moving_average(s, alpha=0.3),
        "17_savitzky_golay_11_3": lambda s, fs: denoise_savitzky_golay(s, window_length=11, polyorder=3),
        "18_savitzky_golay_21_2": lambda s, fs: denoise_savitzky_golay_aggressive(s),
        "19_gaussian_smooth_sigma2": lambda s, fs: denoise_gaussian_smooth(s, sigma=2.0),
        "20_gaussian_smooth_sigma5": lambda s, fs: denoise_gaussian_smooth(s, sigma=5.0),
        # Wavelet Methods
        "21_wavelet_db4_soft": lambda s, fs: denoise_wavelet_db4(s),
        "22_wavelet_db4_hard": lambda s, fs: denoise_wavelet_hard_threshold(s),
        "23_wavelet_sym8_soft": lambda s, fs: denoise_wavelet_sym8(s),
        "24_wavelet_coif3_soft": lambda s, fs: denoise_wavelet_coif3(s),
        "25_wavelet_bayes_shrink": lambda s, fs: denoise_wavelet_bayes_shrink(s),
        # Statistical/Adaptive Filters
        "26_wiener_filter": lambda s, fs: denoise_wiener(s),
        "27_median_filter_5": lambda s, fs: denoise_median_filter(s, kernel_size=5),
        "28_median_filter_11": lambda s, fs: denoise_median_filter_large(s),
        "29_kalman_filter": lambda s, fs: denoise_kalman(s),
        "30_kalman_filter_smooth": lambda s, fs: denoise_kalman_smooth(s),
        "31_adaptive_lms": lambda s, fs: denoise_adaptive_lms(s),
        # Spectral Methods
        "32_spectral_subtraction": lambda s, fs: denoise_spectral_subtraction(s, fs),
        "33_spectral_gating": lambda s, fs: denoise_spectral_gating(s, fs),
        "34_fft_bandpass_50_1500Hz": lambda s, fs: denoise_fft_frequency_filter(s, fs),
        # Advanced Methods
        "35_bilateral_filter": lambda s, fs: denoise_bilateral_1d(s),
        "36_total_variation": lambda s, fs: denoise_total_variation(s),
        "37_notch_50Hz": lambda s, fs: denoise_notch_filter(s, fs, freq=50),
        "38_lpc_enhancement": lambda s, fs: denoise_lpc_residual(s),
        # Combined Methods
        "39_combined_wavelet_butterworth": lambda s, fs: denoise_combined_wavelet_butterworth(s, fs),
        "40_combined_kalman_bandpass": lambda s, fs: denoise_combined_kalman_bandpass(s, fs),
    }


# ============================================================
# OUTPUT AND REPORTING
# ============================================================
def save_wav(signal, sr, filepath):
    """Save signal as WAV file."""
    # Ensure signal is normalized float
    normalized = normalize_signal(signal)
    sf.write(filepath, normalized, sr, subtype='PCM_16')


def generate_report(results, original_snr, output_dir):
    """Generate a text report comparing all methods."""
    report_lines = [
        "=" * 70,
        "BONE CONDUCTION SIGNAL DENOISING - COMPARISON REPORT",
        "=" * 70,
        "",
        f"Original Signal Estimated SNR: {original_snr:.2f} dB",
        f"Original Sampling Rate: {ORIGINAL_SR} Hz",
        "",
        "-" * 70,
        f"{'Method':<45} {'SNR (dB)':>10} {'File':>15}",
        "-" * 70,
    ]

    # Sort by SNR (descending)
    sorted_results = sorted(results, key=lambda x: x['snr'], reverse=True)

    for r in sorted_results:
        if np.isnan(r['snr']):
            snr_str = "N/A"
        elif np.isinf(r['snr']):
            snr_str = "Inf"
        else:
            snr_str = f"{r['snr']:.2f}"
        report_lines.append(f"{r['method']:<45} {snr_str:>10} {'saved':>15}")

    report_lines.extend([
        "-" * 70,
        "",
        "INTERPRETATION:",
        "  - Higher SNR indicates more aggressive noise removal",
        "  - Very high SNR may indicate over-smoothing (loss of speech detail)",
        "  - Best methods for bone conduction speech typically: band-pass",
        "    Butterworth, wavelet denoising, or combined approaches",
        "  - Listen to each WAV file to judge perceptual quality",
        "",
        "RECOMMENDED METHODS FOR BONE CONDUCTION SPEECH:",
        "  1. Band-pass Butterworth (50-1500 Hz) - preserves speech fundamentals",
        "  2. Wavelet denoising (db4 or sym8) - adaptive noise removal",
        "  3. Combined wavelet + Butterworth - best of both approaches",
        "  4. Chebyshev/Elliptic band-pass - sharper frequency cutoffs",
        "  5. Spectral subtraction - if noise is stationary",
        "",
        "=" * 70,
    ])

    report_text = "\n".join(report_lines)
    report_path = os.path.join(output_dir, "denoising_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    print("\n" + report_text)
    return report_path


# ============================================================
# MAIN PROCESSING
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Bone Conduction IMU Signal Denoising Tool"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Input CSV file with columns: timestamp, Accel X, Accel Y, Accel Z"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="denoised_output",
        help="Output directory for WAV files (default: denoised_output)"
    )
    parser.add_argument(
        "--target_sr",
        type=int,
        default=DEFAULT_TARGET_SR,
        choices=AVAILABLE_TARGET_RATES,
        help=f"Target sample rate for output WAV files (default: {DEFAULT_TARGET_SR})"
    )
    parser.add_argument(
        "--no_upsample",
        action="store_true",
        help="Skip upsampling, output at original 3.3 kHz"
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="*",
        default=None,
        help="Specific methods to apply (by number prefix, e.g., 01 04 21)"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("\n" + "=" * 70)
    print("BONE CONDUCTION IMU SIGNAL DENOISING")
    print("=" * 70)
    raw_signal = load_data(args.input)

    # Determine output sample rate
    if args.no_upsample:
        output_sr = ORIGINAL_SR
        print(f"\n  Output sample rate: {output_sr} Hz (no upsampling)")
    else:
        output_sr = args.target_sr
        print(f"\n  Output sample rate: {output_sr} Hz (upsampled from {ORIGINAL_SR} Hz)")

    # Remove DC offset from raw signal
    raw_signal = raw_signal - np.mean(raw_signal)

    # Estimate original SNR
    original_snr = compute_original_snr(raw_signal)
    print(f"  Estimated original SNR: {original_snr:.2f} dB")

    # Save original (unprocessed) as reference
    if args.no_upsample:
        original_out = raw_signal
    else:
        original_out = upsample_signal(raw_signal, ORIGINAL_SR, output_sr)

    original_path = os.path.join(args.output_dir, "00_original_raw.wav")
    save_wav(original_out, output_sr, original_path)
    print(f"\n  Saved original: {original_path}")

    # Get methods
    all_methods = get_all_methods()

    # Filter methods if specified
    if args.methods:
        selected = {}
        for key, func in all_methods.items():
            prefix = key.split("_")[0]
            if prefix in args.methods:
                selected[key] = func
        methods_to_apply = selected
    else:
        methods_to_apply = all_methods

    print(f"\n  Applying {len(methods_to_apply)} denoising methods...")
    print("-" * 70)

    # Process each method
    results = []
    for method_name, method_func in methods_to_apply.items():
        try:
            # Apply denoising at original sample rate
            denoised = method_func(raw_signal, ORIGINAL_SR)

            # Ensure same length
            min_len = min(len(denoised), len(raw_signal))
            denoised = denoised[:min_len]
            raw_trimmed = raw_signal[:min_len]

            # Compute SNR
            snr = compute_snr(raw_trimmed, denoised)

            # Upsample if needed
            if args.no_upsample:
                output_signal = denoised
            else:
                output_signal = upsample_signal(denoised, ORIGINAL_SR, output_sr)

            # Save WAV
            wav_path = os.path.join(args.output_dir, f"{method_name}.wav")
            save_wav(output_signal, output_sr, wav_path)

            results.append({
                "method": method_name,
                "snr": snr,
                "path": wav_path,
            })
            print(f"  [OK] {method_name:<45} SNR: {snr:>8.2f} dB")

        except Exception as e:
            print(f"  [FAIL] {method_name:<45} Error: {str(e)}")
            results.append({
                "method": method_name,
                "snr": float('-inf'),
                "path": None,
            })

    # Generate report
    print("\n")
    report_path = generate_report(results, original_snr, args.output_dir)
    print(f"\n  Report saved: {report_path}")
    print(f"  Output directory: {os.path.abspath(args.output_dir)}")
    print(f"  Total WAV files: {sum(1 for r in results if r['path'])} + 1 (original)")
    print("=" * 70)


if __name__ == "__main__":
    main()
