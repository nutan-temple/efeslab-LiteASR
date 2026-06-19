#!/usr/bin/env python3
"""
IMU Neural Speech Enhancement
==============================
Advanced neural/AI-based speech enhancement for bone conduction IMU data.
Uses Facebook Denoiser DNS64, SpeechBrain MetricGAN+, VoiceFixer, and
cascaded multi-stage pipelines to produce the clearest possible speech
from 3.3kHz bone conduction accelerometer CSV input.

Usage:
    python imu_neural_enhance.py                          # Use embedded sample data
    python imu_neural_enhance.py recording.csv            # Use CSV file
    python imu_neural_enhance.py --input recording.csv    # Alternative input flag
    python imu_neural_enhance.py --output-dir out         # Specify output directory
    python imu_neural_enhance.py --axis z --gap-mode chunk
"""

import argparse
import os
import subprocess
import sys
import tempfile
import warnings
from io import StringIO
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, resample_poly
from scipy.special import exp1

warnings.filterwarnings("ignore")

# ============================================================
# CONSTANTS
# ============================================================
ORIGINAL_SR = 3300       # Nominal IMU sampling rate (Hz)
TARGET_SR = 16000        # Neural model input sample rate (Hz)
BANDPASS_LOW = 50        # Hz
BANDPASS_HIGH = 1500     # Hz
BUTTER_ORDER = 4

# ============================================================
# EMBEDDED SAMPLE CSV (for standalone testing)
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


# ============================================================
# SECTION 1: Auto-install missing packages
# ============================================================
def ensure_package(package_name, pip_name=None):
    """Try importing a package; if missing, install via pip."""
    if pip_name is None:
        pip_name = package_name
    try:
        __import__(package_name)
        return True
    except ImportError:
        print(f"  [INSTALL] {pip_name} not found, installing...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pip_name, "-q"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            __import__(package_name)
            print(f"  [INSTALL] {pip_name} installed successfully.")
            return True
        except Exception as e:
            print(f"  [INSTALL] Failed to install {pip_name}: {e}")
            return False


# ============================================================
# SECTION 2: CSV Loading
# ============================================================
def load_imu_csv(path, axis="z"):
    """
    Load IMU CSV with columns: timestamp, Accel X, Accel Y, Accel Z.

    Parameters
    ----------
    path : str
        Path to CSV file.
    axis : str
        Which axis to extract: x, y, or z.

    Returns
    -------
    timestamps : ndarray
    signal : ndarray
    """
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


# ============================================================
# SECTION 3: Gap-Aware Resampling
# ============================================================
def gap_aware_resample(timestamps, signal, mode="uniform"):
    """
    Handle non-uniform/burst IMU timestamps.

    Parameters
    ----------
    timestamps : ndarray
        Timestamps in milliseconds.
    signal : ndarray
        Raw accelerometer signal.
    mode : str
        chunk - use largest contiguous chunk
        uniform - interpolate onto uniform grid
        all - use all samples as-is

    Returns
    -------
    resampled_signal : ndarray
    inferred_sr : float
    """
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


# ============================================================
# SECTION 4: Core DSP - Bandpass, AGC, MMSE-LSA, Upsample
# ============================================================
def apply_bandpass(signal, fs, low=BANDPASS_LOW, high=BANDPASS_HIGH, order=BUTTER_ORDER):
    """Apply Butterworth band-pass filter using sos (second-order sections)."""
    nyquist = fs / 2.0
    high_norm = min(high, nyquist * 0.99)
    low_norm = max(low, 1.0)
    if low_norm >= high_norm:
        return signal.copy()
    sos = butter(order, [low_norm / nyquist, high_norm / nyquist],
                 btype="band", output="sos")
    return sosfilt(sos, signal)


def adaptive_gain_control(signal, fs, target_level_db=-20,
                          attack_ms=10, release_ms=100,
                          max_gain_db=35, segment_ms=20):
    """
    Adaptive Gain Control with attack/release time constants.
    Segment-by-segment energy-based gain that boosts quiet segments
    more aggressively while limiting loud segments.
    Max gain: 35 dB.
    """
    target_level = 10.0 ** (target_level_db / 20.0)
    max_gain = 10.0 ** (max_gain_db / 20.0)
    seg_len = max(1, int(segment_ms * fs / 1000.0))
    n_segs = max(1, len(signal) // seg_len)
    output = np.zeros_like(signal)
    current_gain = 1.0
    attack_coeff = 1.0 - np.exp(-seg_len / (attack_ms * fs / 1000.0))
    release_coeff = 1.0 - np.exp(-seg_len / (release_ms * fs / 1000.0))

    for i in range(n_segs):
        start = i * seg_len
        end = min(start + seg_len, len(signal))
        seg = signal[start:end]
        seg_rms = np.sqrt(np.mean(seg ** 2) + 1e-10)
        desired_gain = target_level / (seg_rms + 1e-10)
        desired_gain = min(desired_gain, max_gain)
        desired_gain = max(desired_gain, 0.01)
        if desired_gain < current_gain:
            current_gain += attack_coeff * (desired_gain - current_gain)
        else:
            current_gain += release_coeff * (desired_gain - current_gain)
        output[start:end] = seg * current_gain

    remainder_start = n_segs * seg_len
    if remainder_start < len(signal):
        output[remainder_start:] = signal[remainder_start:] * current_gain
    return output


def mmse_lsa_denoise(signal, fs, frame_ms=25, hop_ms=10,
                     noise_frames=10, alpha_dd=0.98, floor_db=-30):
    """
    MMSE Log-Spectral Amplitude estimator (Ephraim & Malah, 1985).
    Decision-directed approach using scipy.special.exp1.
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


def upsample_to_target(signal, fs_in, fs_out=TARGET_SR):
    """Polyphase resampling to target sample rate."""
    fs_in_int = int(np.round(fs_in))
    g = gcd(fs_out, fs_in_int)
    up = fs_out // g
    down = fs_in_int // g
    return resample_poly(signal, up, down)


def normalize_peak(signal):
    """Normalize to [-0.95, 0.95] range."""
    peak = np.max(np.abs(signal))
    if peak > 1e-10:
        return signal / peak * 0.95
    return signal.copy()


# ============================================================
# SECTION 5: Preprocessing for Neural Models
# ============================================================
def preprocess_for_neural(timestamps, signal, fs_orig, gap_mode="uniform"):
    """
    Full preprocessing pipeline to prepare bone conduction IMU signal
    for neural speech enhancement models.

    Steps:
      1. Gap-aware resampling
      2. DC removal / detrend
      3. Bandpass filter 50-1500 Hz
      4. MMSE-LSA denoising
      5. AGC to bring above noise floor
      6. Polyphase upsample to 16kHz
      7. Normalize to [-1, 1]

    Returns
    -------
    signal_16k : ndarray
        Preprocessed signal at 16kHz, normalized.
    """
    print("  [Preprocess] Gap-aware resampling...")
    sig, inferred_sr = gap_aware_resample(timestamps, signal, mode=gap_mode)
    print(f"    Inferred SR: {inferred_sr:.1f} Hz, {len(sig)} samples")

    # DC removal
    sig = sig - np.mean(sig)

    # Bandpass 50-1500 Hz
    print("  [Preprocess] Bandpass filter 50-1500 Hz...")
    sig = apply_bandpass(sig, inferred_sr, low=BANDPASS_LOW, high=BANDPASS_HIGH)

    # MMSE-LSA denoising
    print("  [Preprocess] MMSE-LSA denoising...")
    sig = mmse_lsa_denoise(sig, inferred_sr)

    # AGC to bring above noise floor
    print("  [Preprocess] Adaptive Gain Control (max 35dB)...")
    sig = adaptive_gain_control(sig, inferred_sr, target_level_db=-20,
                                max_gain_db=35)

    # Upsample to 16kHz
    print("  [Preprocess] Upsampling to 16kHz...")
    sig_16k = upsample_to_target(sig, inferred_sr, TARGET_SR)

    # Normalize to [-1, 1]
    sig_16k = normalize_peak(sig_16k)

    print(f"  [Preprocess] Done: {len(sig_16k)} samples at 16kHz")
    print(f"    ({len(sig_16k) / TARGET_SR:.3f}s duration)")
    return sig_16k


# ============================================================
# SECTION 6: Facebook Denoiser DNS64
# ============================================================
def enhance_denoiser_dns64(signal_16k):
    """
    Enhance using Facebook/Meta Denoiser (DNS64 model).

    The model expects 16kHz mono input as a tensor of shape
    (batch, channels, time).

    Parameters
    ----------
    signal_16k : ndarray
        Preprocessed signal at 16kHz, normalized.

    Returns
    -------
    enhanced : ndarray or None
        Enhanced signal at 16kHz, or None if failed.
    """
    print("  [DNS64] Loading Facebook Denoiser DNS64 model...")
    try:
        import torch
        from denoiser import pretrained

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"    Device: {device}")

        model = pretrained.dns64()
        model = model.to(device)
        model.eval()

        # Prepare tensor: (batch=1, channels=1, time)
        wav_tensor = torch.from_numpy(signal_16k.copy()).float()
        wav_tensor = wav_tensor.unsqueeze(0).unsqueeze(0).to(device)

        print("  [DNS64] Running inference...")
        with torch.no_grad():
            enhanced = model(wav_tensor)

        enhanced_np = enhanced.squeeze().cpu().numpy()

        # Match length
        if len(enhanced_np) > len(signal_16k):
            enhanced_np = enhanced_np[:len(signal_16k)]
        elif len(enhanced_np) < len(signal_16k):
            enhanced_np = np.pad(enhanced_np, (0, len(signal_16k) - len(enhanced_np)))

        print("  [DNS64] Done.")
        return enhanced_np

    except Exception as e:
        print(f"  [DNS64] FAILED: {e}")
        return None


# ============================================================
# SECTION 7: SpeechBrain MetricGAN+
# ============================================================
def enhance_metricgan_plus(signal_16k):
    """
    Enhance using SpeechBrain MetricGAN+ (trained to maximize PESQ).

    The model expects 16kHz mono input tensor.

    Parameters
    ----------
    signal_16k : ndarray
        Preprocessed signal at 16kHz, normalized.

    Returns
    -------
    enhanced : ndarray or None
        Enhanced signal at 16kHz, or None if failed.
    """
    print("  [MetricGAN+] Loading SpeechBrain MetricGAN+ model...")
    try:
        import torch
        from speechbrain.inference.enhancement import SpectralMaskEnhancement

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"    Device: {device}")

        model = SpectralMaskEnhancement.from_hparams(
            source="speechbrain/metricgan-plus-voicebank",
            run_opts={"device": device},
        )

        # MetricGAN+ expects a 1D tensor at 16kHz
        wav_tensor = torch.from_numpy(signal_16k.copy()).float().unsqueeze(0)

        print("  [MetricGAN+] Running inference...")
        enhanced = model.enhance_batch(wav_tensor, lengths=torch.tensor([1.0]))
        enhanced_np = enhanced.squeeze().cpu().numpy()

        # Match length
        if len(enhanced_np) > len(signal_16k):
            enhanced_np = enhanced_np[:len(signal_16k)]
        elif len(enhanced_np) < len(signal_16k):
            enhanced_np = np.pad(enhanced_np, (0, len(signal_16k) - len(enhanced_np)))

        print("  [MetricGAN+] Done.")
        return enhanced_np

    except Exception as e:
        print(f"  [MetricGAN+] FAILED: {e}")
        return None


# ============================================================
# SECTION 8: VoiceFixer
# ============================================================
def enhance_voicefixer(signal_16k, output_path=None):
    """
    Enhance using VoiceFixer (denoising + super-resolution + bandwidth extension).

    VoiceFixer works with file paths - it reads a WAV, processes it, and
    writes the output. It has 3 modes (0, 1, 2); mode 0 is default.
    Output is at 44.1kHz, which we resample back to 16kHz.

    Parameters
    ----------
    signal_16k : ndarray
        Preprocessed signal at 16kHz, normalized.
    output_path : str or None
        If provided, save VoiceFixer full-resolution output here.

    Returns
    -------
    enhanced : ndarray or None
        Enhanced signal resampled back to 16kHz, or None if failed.
    """
    print("  [VoiceFixer] Loading VoiceFixer model...")
    tmp_in_path = None
    tmp_out_path = None
    try:
        from voicefixer import VoiceFixer

        vf = VoiceFixer()

        # VoiceFixer needs file paths, so use temp files
        tmp_in = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_in_path = tmp_in.name
        tmp_out_path = tmp_out.name
        tmp_in.close()
        tmp_out.close()

        # Write input to temp WAV at 16kHz
        sf.write(tmp_in_path, normalize_peak(signal_16k), TARGET_SR)

        print("  [VoiceFixer] Running inference (mode=0)...")
        vf.restore(input=tmp_in_path, output=tmp_out_path, cuda=False, mode=0)

        # Read the output (VoiceFixer outputs at 44.1kHz)
        enhanced_44k, sr_out = sf.read(tmp_out_path)

        # Resample back to 16kHz for consistency
        if sr_out != TARGET_SR:
            g = gcd(TARGET_SR, sr_out)
            up = TARGET_SR // g
            down = sr_out // g
            enhanced_16k = resample_poly(enhanced_44k, up, down)
        else:
            enhanced_16k = enhanced_44k

        # If stereo, convert to mono
        if enhanced_16k.ndim > 1:
            enhanced_16k = np.mean(enhanced_16k, axis=1)

        # Cleanup temp files
        try:
            os.unlink(tmp_in_path)
            os.unlink(tmp_out_path)
        except OSError:
            pass

        # Save full-res if output path provided
        if output_path:
            sf.write(output_path, normalize_peak(enhanced_44k), sr_out)

        print("  [VoiceFixer] Done.")
        return enhanced_16k

    except Exception as e:
        print(f"  [VoiceFixer] FAILED: {e}")
        # Cleanup on failure
        for p in [tmp_in_path, tmp_out_path]:
            if p is not None:
                try:
                    os.unlink(p)
                except OSError:
                    pass
        return None


# ============================================================
# SECTION 9: Multi-Stage Neural Pipeline
# ============================================================
def enhance_multistage(signal_16k):
    """
    Multi-stage neural enhancement pipeline:
      Stage 1: Aggressive AGC preprocessing (bring above noise floor)
      Stage 2: MMSE-LSA denoising (clean up before neural)
      Stage 3: First neural pass (Denoiser DNS64)
      Stage 4: Second neural pass (VoiceFixer for quality + bandwidth)

    Each stage feeds its output to the next.

    Parameters
    ----------
    signal_16k : ndarray
        Preprocessed signal at 16kHz.

    Returns
    -------
    enhanced : ndarray or None
        Multi-stage enhanced signal, or None if all neural models failed.
    """
    separator = "=" * 50
    print(f"\n{separator}")
    print("  MULTI-STAGE NEURAL PIPELINE")
    print(separator)

    # Stage 1: Aggressive AGC
    print("  [Stage 1/4] Aggressive AGC...")
    sig = adaptive_gain_control(signal_16k, TARGET_SR, target_level_db=-15,
                                max_gain_db=35, attack_ms=5, release_ms=50)
    sig = normalize_peak(sig)

    # Stage 2: MMSE-LSA on the 16kHz signal
    print("  [Stage 2/4] MMSE-LSA denoising at 16kHz...")
    sig = mmse_lsa_denoise(sig, TARGET_SR)
    sig = normalize_peak(sig)

    # Stage 3: DNS64
    print("  [Stage 3/4] Denoiser DNS64...")
    dns_result = enhance_denoiser_dns64(sig)
    if dns_result is not None:
        sig = normalize_peak(dns_result)
    else:
        print("    Skipping DNS64 (failed), continuing with current signal.")

    # Stage 4: VoiceFixer
    print("  [Stage 4/4] VoiceFixer (quality + bandwidth extension)...")
    vf_result = enhance_voicefixer(sig)
    if vf_result is not None:
        sig = normalize_peak(vf_result)
    else:
        print("    Skipping VoiceFixer (failed), using DNS64 output.")

    # If no neural model succeeded at all, return None
    if dns_result is None and vf_result is None:
        print("  [Multi-stage] All neural models failed.")
        return None

    print("  [Multi-stage] Pipeline complete.")
    return sig


# ============================================================
# SECTION 10: Cascaded Enhancement (DNS64 -> MetricGAN+ -> VoiceFixer)
# ============================================================
def enhance_cascaded_full(signal_16k):
    """
    Cascaded Enhancement: Run models in sequence, each catching what
    the previous one missed.

    Order: Denoiser DNS64 -> MetricGAN+ -> VoiceFixer

    Parameters
    ----------
    signal_16k : ndarray
        Preprocessed signal at 16kHz.

    Returns
    -------
    enhanced : ndarray or None
        Cascaded enhanced signal, or None if all models failed.
    """
    separator = "=" * 50
    print(f"\n{separator}")
    print("  CASCADED ENHANCEMENT (DNS64 -> MetricGAN+ -> VoiceFixer)")
    print(separator)

    sig = signal_16k.copy()
    any_succeeded = False

    # Step 1: DNS64
    print("  [Cascade 1/3] Denoiser DNS64...")
    result = enhance_denoiser_dns64(sig)
    if result is not None:
        sig = normalize_peak(result)
        any_succeeded = True
    else:
        print("    DNS64 failed, passing input to next stage.")

    # Step 2: MetricGAN+
    print("  [Cascade 2/3] MetricGAN+...")
    result = enhance_metricgan_plus(sig)
    if result is not None:
        sig = normalize_peak(result)
        any_succeeded = True
    else:
        print("    MetricGAN+ failed, passing input to next stage.")

    # Step 3: VoiceFixer
    print("  [Cascade 3/3] VoiceFixer...")
    result = enhance_voicefixer(sig)
    if result is not None:
        sig = normalize_peak(result)
        any_succeeded = True
    else:
        print("    VoiceFixer failed, using previous stage output.")

    if not any_succeeded:
        print("  [Cascaded] All models failed.")
        return None

    print("  [Cascaded] Pipeline complete.")
    return sig


# ============================================================
# SECTION 11: Main Pipeline Runner
# ============================================================
def run_all_enhancements(input_path, output_dir, axis="z", gap_mode="uniform"):
    """
    Run ALL enhancement approaches and save results as WAV files.

    Approaches:
      1. Preprocessed only (baseline)
      2. DNS64 individually
      3. MetricGAN+ individually
      4. VoiceFixer individually
      5. Multi-stage pipeline (AGC -> MMSE -> DNS64 -> VoiceFixer)
      6. Cascaded pipeline (DNS64 -> MetricGAN+ -> VoiceFixer)

    Parameters
    ----------
    input_path : str
        Path to input CSV file.
    output_dir : str
        Output directory for WAV files.
    axis : str
        Accelerometer axis (x, y, z).
    gap_mode : str
        Gap handling mode (chunk, uniform, all).
    """
    separator = "=" * 60
    print(f"\n{separator}")
    print("  IMU Neural Speech Enhancement")
    print(separator)
    print(f"  Input:      {input_path}")
    print(f"  Output dir: {output_dir}")
    print(f"  Axis:       {axis}")
    print(f"  Gap mode:   {gap_mode}")
    print(separator)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load CSV
    print("\n[1/7] Loading IMU CSV data...")
    timestamps, signal = load_imu_csv(input_path, axis=axis)
    print(f"  Loaded {len(signal)} samples")
    if len(timestamps) > 1:
        duration = (timestamps[-1] - timestamps[0]) / 1000.0
        print(f"  Duration: ~{duration:.3f}s")

    # Preprocess
    print("\n[2/7] Preprocessing for neural models...")
    signal_16k = preprocess_for_neural(timestamps, signal, ORIGINAL_SR, gap_mode)

    # Save preprocessed baseline
    baseline_path = os.path.join(output_dir, "00_preprocessed_baseline.wav")
    sf.write(baseline_path, normalize_peak(signal_16k), TARGET_SR)
    print(f"  Saved baseline: {baseline_path}")

    results = {}

    # --- Individual Models ---
    # DNS64
    print("\n[3/7] Facebook Denoiser DNS64 (individual)...")
    dns_result = enhance_denoiser_dns64(signal_16k)
    if dns_result is not None:
        path = os.path.join(output_dir, "01_dns64.wav")
        sf.write(path, normalize_peak(dns_result), TARGET_SR)
        results["dns64"] = path
        print(f"  Saved: {path}")
    else:
        print("  SKIPPED (model unavailable)")

    # MetricGAN+
    print("\n[4/7] SpeechBrain MetricGAN+ (individual)...")
    mg_result = enhance_metricgan_plus(signal_16k)
    if mg_result is not None:
        path = os.path.join(output_dir, "02_metricgan_plus.wav")
        sf.write(path, normalize_peak(mg_result), TARGET_SR)
        results["metricgan_plus"] = path
        print(f"  Saved: {path}")
    else:
        print("  SKIPPED (model unavailable)")

    # VoiceFixer
    print("\n[5/7] VoiceFixer (individual)...")
    vf_result = enhance_voicefixer(signal_16k)
    if vf_result is not None:
        path = os.path.join(output_dir, "03_voicefixer.wav")
        sf.write(path, normalize_peak(vf_result), TARGET_SR)
        results["voicefixer"] = path
        print(f"  Saved: {path}")
    else:
        print("  SKIPPED (model unavailable)")

    # --- Multi-stage Pipeline ---
    print("\n[6/7] Multi-stage Neural Pipeline...")
    ms_result = enhance_multistage(signal_16k)
    if ms_result is not None:
        path = os.path.join(output_dir, "04_multistage_pipeline.wav")
        sf.write(path, normalize_peak(ms_result), TARGET_SR)
        results["multistage"] = path
        print(f"  Saved: {path}")
    else:
        print("  SKIPPED (all neural models failed)")

    # --- Cascaded Pipeline ---
    print("\n[7/7] Cascaded Enhancement Pipeline...")
    casc_result = enhance_cascaded_full(signal_16k)
    if casc_result is not None:
        path = os.path.join(output_dir, "05_cascaded_full.wav")
        sf.write(path, normalize_peak(casc_result), TARGET_SR)
        results["cascaded"] = path
        print(f"  Saved: {path}")
    else:
        print("  SKIPPED (all neural models failed)")

    # Summary
    print(f"\n{separator}")
    print("  SUMMARY")
    print(separator)
    print(f"  Output directory: {output_dir}")
    print(f"  Baseline:         {baseline_path}")
    for name, path in results.items():
        print(f"  {name:18s} {path}")
    n_success = len(results)
    n_total = 5  # dns64, metricgan+, voicefixer, multistage, cascaded
    print(f"\n  Models succeeded: {n_success}/{n_total}")
    if n_success == 0:
        print("  WARNING: No neural models ran successfully.")
        print("  Try: pip install denoiser speechbrain voicefixer torch torchaudio")
    print(separator)


# ============================================================
# SECTION 12: CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="IMU Neural Speech Enhancement: bone conduction to clear speech",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python imu_neural_enhance.py                                  # Embedded sample data
  python imu_neural_enhance.py recording.csv                    # CSV input
  python imu_neural_enhance.py --input recording.csv --axis z   # With flags
  python imu_neural_enhance.py recording.csv --output-dir out   # Custom output dir
  python imu_neural_enhance.py recording.csv --gap-mode chunk   # Gap handling
""",
    )

    parser.add_argument(
        "csv_input", nargs="?", default=None,
        help="Input CSV file (positional argument)")
    parser.add_argument(
        "--input", "-i", dest="input_flag", default=None,
        help="Input CSV file (alternative to positional arg)")
    parser.add_argument(
        "--output-dir", default="neural_enhanced",
        help="Output directory for WAV files (default: neural_enhanced)")
    parser.add_argument(
        "--axis", default="z", choices=["x", "y", "z"],
        help="Accelerometer axis to use (default: z)")
    parser.add_argument(
        "--gap-mode", default="uniform", choices=["chunk", "uniform", "all"],
        help="Gap handling mode (default: uniform)")

    args = parser.parse_args()

    # Resolve input path
    input_path = args.csv_input or args.input_flag

    if input_path is None:
        # Use embedded sample data
        print("No input file specified. Using embedded sample data.")
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, prefix="imu_sample_")
        tmp.write(SAMPLE_CSV.strip())
        tmp.close()
        input_path = tmp.name
        use_sample = True
    else:
        use_sample = False
        if not os.path.isfile(input_path):
            print(f"Error: Input file not found: {input_path}", file=sys.stderr)
            sys.exit(1)

    # Ensure required packages are available
    print("\nChecking neural model dependencies...")
    ensure_package("torch")
    ensure_package("torchaudio")
    ensure_package("denoiser")
    ensure_package("speechbrain")
    ensure_package("voicefixer")

    # Run all enhancements
    run_all_enhancements(
        input_path=input_path,
        output_dir=args.output_dir,
        axis=args.axis,
        gap_mode=args.gap_mode,
    )

    # Cleanup temp file if we used sample data
    if use_sample:
        try:
            os.unlink(input_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
