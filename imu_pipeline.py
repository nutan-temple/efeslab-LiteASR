#!/usr/bin/env python3
"""
IMU-to-Audio Pipeline
=====================
Converts 3.3 kHz IMU accelerometer data to a 16 kHz speech-quality WAV file.

Pipeline stages:
  1. Load IMU CSV (timestamp, Accel X, Accel Y, Accel Z)
  2. Gap-aware resampling (detect gaps, handle via chunk/uniform/all)
  3. Bandpass filter 50-1500 Hz (Butterworth order 4)
  4. MMSE-LSA denoising (Ephraim & Malah 1985 decision-directed)
  5. Upsample to 16 kHz (polyphase resampling)
  6. Optional: neural speech enhancement (Facebook denoiser / DNS)

Usage:
  python imu_pipeline.py recording.csv --output out.wav --axis z
  python imu_pipeline.py --input recording.csv --skip-neural
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, resample_poly
from scipy.special import exp1

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ORIGINAL_SR = 3300       # Nominal IMU sampling rate (Hz)
TARGET_SR = 16000        # Output sampling rate (Hz)
BANDPASS_LOW = 50        # Hz
BANDPASS_HIGH = 1500     # Hz
BUTTER_ORDER = 4


# ---------------------------------------------------------------------------
# Utility: timing decorator
# ---------------------------------------------------------------------------
class PipelineTimer:
    """Context manager that prints timing for each pipeline stage."""

    def __init__(self, stage_name):
        self.stage_name = stage_name

    def __enter__(self):
        self.start = time.perf_counter()
        print(f"  [{self.stage_name}] starting...")
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        print(f"  [{self.stage_name}] done in {elapsed:.4f}s")


# ---------------------------------------------------------------------------
# Stage 1: Load CSV
# ---------------------------------------------------------------------------
def load_imu_csv(path, axis="z"):
    """
    Load IMU CSV with columns: timestamp, Accel X, Accel Y, Accel Z

    Parameters
    ----------
    path : str
        Path to the CSV file.
    axis : str
        Which axis to extract: 'x', 'y', or 'z'.

    Returns
    -------
    timestamps : ndarray
        Timestamps in milliseconds.
    signal : ndarray
        Accelerometer values for the chosen axis.
    """
    import pandas as pd

    df = pd.read_csv(path)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Extract timestamp
    ts_col = None
    for candidate in ["timestamp", "time", "ts", "t"]:
        if candidate in df.columns:
            ts_col = candidate
            break
    if ts_col is None:
        ts_col = df.columns[0]

    timestamps = df[ts_col].values.astype(np.float64)

    # Extract axis
    axis_map = {"x": 1, "y": 2, "z": 3}
    axis_lower = axis.lower()
    signal_col = None
    for col in df.columns:
        if f"accel {axis_lower}" in col or f"accel_{axis_lower}" in col:
            signal_col = col
            break
    if signal_col is None:
        # Fall back to positional index
        idx = axis_map.get(axis_lower, 3)
        signal_col = df.columns[idx] if idx < len(df.columns) else df.columns[-1]

    signal = df[signal_col].values.astype(np.float64)

    return timestamps, signal


# ---------------------------------------------------------------------------
# Stage 2: Gap-aware resampling
# ---------------------------------------------------------------------------
def gap_aware_resample(timestamps, signal, mode="uniform"):
    """
    Handle non-uniform / burst IMU timestamps.

    Parameters
    ----------
    timestamps : ndarray
        Timestamps in milliseconds.
    signal : ndarray
        Raw accelerometer signal.
    mode : str
        'chunk' - use largest contiguous chunk (gap < 2x median dt)
        'uniform' - interpolate onto uniform grid
        'all' - use all samples as-is

    Returns
    -------
    resampled_signal : ndarray
        Signal on a uniform time grid.
    inferred_sr : float
        Inferred sample rate (Hz).
    """
    diffs = np.diff(timestamps)

    if len(diffs) == 0:
        return signal, ORIGINAL_SR

    median_dt = np.median(diffs)
    inferred_sr = 1000.0 / median_dt  # timestamps in ms

    if mode == "all":
        return signal, inferred_sr

    elif mode == "chunk":
        # Find contiguous chunks where gap < 2 * median_dt
        gap_threshold = 2.0 * median_dt
        gap_indices = np.where(diffs > gap_threshold)[0]

        # Build chunk boundaries
        boundaries = np.concatenate([[0], gap_indices + 1, [len(signal)]])
        chunk_lengths = np.diff(boundaries)
        largest_idx = np.argmax(chunk_lengths)
        start = boundaries[largest_idx]
        end = boundaries[largest_idx + 1]

        chunk_signal = signal[start:end]
        chunk_ts = timestamps[start:end]

        # Infer SR from this chunk
        if len(chunk_ts) > 1:
            chunk_dt = np.median(np.diff(chunk_ts))
            inferred_sr = 1000.0 / chunk_dt
        else:
            inferred_sr = ORIGINAL_SR

        # Interpolate chunk onto uniform grid
        duration_ms = chunk_ts[-1] - chunk_ts[0]
        n_samples = int(np.round(duration_ms * inferred_sr / 1000.0))
        if n_samples < 2:
            return chunk_signal, inferred_sr

        uniform_ts = np.linspace(chunk_ts[0], chunk_ts[-1], n_samples)
        resampled = np.interp(uniform_ts, chunk_ts, chunk_signal)
        return resampled, inferred_sr

    else:  # uniform
        # Interpolate entire signal onto uniform grid at inferred SR
        duration_ms = timestamps[-1] - timestamps[0]
        n_samples = int(np.round(duration_ms * inferred_sr / 1000.0))
        if n_samples < 2:
            return signal, inferred_sr

        uniform_ts = np.linspace(timestamps[0], timestamps[-1], n_samples)
        resampled = np.interp(uniform_ts, timestamps, signal)
        return resampled, inferred_sr


# ---------------------------------------------------------------------------
# Stage 3: Bandpass filter
# ---------------------------------------------------------------------------
def apply_bandpass(signal, fs, low=BANDPASS_LOW, high=BANDPASS_HIGH,
                   order=BUTTER_ORDER):
    """
    Apply a Butterworth bandpass filter.

    Parameters
    ----------
    signal : ndarray
    fs : float
        Sample rate.
    low, high : float
        Cutoff frequencies in Hz.
    order : int
        Filter order.

    Returns
    -------
    filtered : ndarray
    """
    nyquist = fs / 2.0
    # Clamp high to slightly below Nyquist
    high_norm = min(high, nyquist * 0.99)
    low_norm = max(low, 1.0)

    if low_norm >= high_norm:
        return signal

    sos = butter(order, [low_norm / nyquist, high_norm / nyquist],
                 btype="band", output="sos")
    filtered = sosfilt(sos, signal)
    return filtered


# ---------------------------------------------------------------------------
# Stage 4: MMSE-LSA Denoising
# ---------------------------------------------------------------------------
def mmse_lsa_denoise(signal, fs, frame_ms=25, hop_ms=10,
                     noise_frames=10, alpha_dd=0.98, floor_db=-30):
    """
    MMSE Log-Spectral Amplitude estimator (Ephraim & Malah, 1985).

    Uses the decision-directed approach for a priori SNR estimation
    and the exponential integral (E1) for the MMSE-LSA gain.

    Parameters
    ----------
    signal : ndarray
        Input time-domain signal.
    fs : float
        Sampling rate.
    frame_ms : float
        Frame length in milliseconds.
    hop_ms : float
        Hop size in milliseconds.
    noise_frames : int
        Number of initial frames used to estimate noise spectrum.
    alpha_dd : float
        Smoothing factor for decision-directed SNR (0.9-0.99).
    floor_db : float
        Gain floor in dB.

    Returns
    -------
    enhanced : ndarray
        Enhanced time-domain signal (same length as input).
    """
    frame_len = int(np.round(frame_ms * fs / 1000.0))
    hop_len = int(np.round(hop_ms * fs / 1000.0))

    # Ensure minimum frame size
    if frame_len < 4:
        frame_len = 4
    if hop_len < 1:
        hop_len = 1

    # Window
    win = np.hanning(frame_len)
    win_sq = win ** 2

    # Pad signal
    n_orig = len(signal)
    n_frames = max(1, 1 + (n_orig - frame_len) // hop_len)
    pad_len = max(n_orig, (n_frames - 1) * hop_len + frame_len)
    # Recompute n_frames based on actual pad_len
    n_frames = max(1, 1 + (pad_len - frame_len) // hop_len)
    x = np.zeros(pad_len)
    x[:n_orig] = signal

    # STFT
    nfft = frame_len
    freq_bins = nfft // 2 + 1

    # Compute spectrogram
    X = np.zeros((n_frames, freq_bins), dtype=np.complex128)
    for i in range(n_frames):
        start = i * hop_len
        frame = x[start:start + frame_len] * win
        X[i, :] = np.fft.rfft(frame, n=nfft)

    # Power spectrum
    P = np.abs(X) ** 2

    # Noise estimation from initial frames
    n_noise = min(noise_frames, n_frames)
    noise_psd = np.mean(P[:n_noise, :], axis=0) + 1e-10

    # Gain floor
    G_floor = 10.0 ** (floor_db / 20.0)

    # Decision-directed MMSE-LSA
    enhanced_X = np.zeros_like(X)
    prev_gain = np.ones(freq_bins)
    prev_P = P[0, :] if n_frames > 0 else np.ones(freq_bins)

    for i in range(n_frames):
        # A posteriori SNR
        gamma = P[i, :] / (noise_psd + 1e-10)
        gamma = np.maximum(gamma, 1e-3)

        # A priori SNR (decision-directed)
        if i == 0:
            xi = np.maximum(gamma - 1.0, 0.0)
        else:
            # Decision-directed: xi = alpha * (G^2 * |X|^2 / noise) + (1-alpha) * max(gamma-1, 0)
            xi_ml = np.maximum(gamma - 1.0, 0.0)
            xi_dd = (prev_gain ** 2) * prev_P / (noise_psd + 1e-10)
            xi = alpha_dd * xi_dd + (1.0 - alpha_dd) * xi_ml

        xi = np.maximum(xi, 1e-3)

        # MMSE-LSA gain
        # G_LSA = xi / (1 + xi) * exp(0.5 * E1(v))
        # where v = xi * gamma / (1 + xi)
        v = xi * gamma / (1.0 + xi)
        v = np.maximum(v, 1e-10)

        # Exponential integral E1
        e1_v = exp1(v)

        # MMSE-LSA gain
        gain = (xi / (1.0 + xi)) * np.exp(0.5 * e1_v)
        gain = np.real(gain)
        gain = np.maximum(gain, G_floor)
        gain = np.minimum(gain, 1.0)

        # Apply gain
        enhanced_X[i, :] = gain * X[i, :]

        # Store for next frame
        prev_gain = gain
        prev_P = P[i, :]

    # Inverse STFT (overlap-add)
    output = np.zeros(pad_len)
    win_sum = np.zeros(pad_len)

    for i in range(n_frames):
        start = i * hop_len
        frame = np.fft.irfft(enhanced_X[i, :], n=nfft)
        output[start:start + frame_len] += frame * win
        win_sum[start:start + frame_len] += win_sq

    # Normalize by window sum
    win_sum = np.maximum(win_sum, 1e-8)
    output = output / win_sum

    return output[:n_orig]


# ---------------------------------------------------------------------------
# Stage 5: Upsample to 16 kHz
# ---------------------------------------------------------------------------
def upsample_to_target(signal, fs_in, fs_out=TARGET_SR):
    """
    Upsample signal from fs_in to fs_out using polyphase resampling.

    Parameters
    ----------
    signal : ndarray
    fs_in : float
        Input sample rate.
    fs_out : int
        Target sample rate.

    Returns
    -------
    resampled : ndarray
    """
    # Find rational approximation
    from math import gcd
    fs_in_int = int(np.round(fs_in))
    g = gcd(fs_out, fs_in_int)
    up = fs_out // g
    down = fs_in_int // g

    resampled = resample_poly(signal, up, down)
    return resampled


# ---------------------------------------------------------------------------
# Stage 6: Neural speech enhancement (optional)
# ---------------------------------------------------------------------------
def neural_enhance(signal, fs=TARGET_SR):
    """
    Apply neural speech enhancement using Facebook's denoiser (DNS) model.

    Falls back to spectral envelope enhancement if denoiser is not available.

    Parameters
    ----------
    signal : ndarray
        Input signal at fs Hz.
    fs : int
        Sample rate (should be 16000).

    Returns
    -------
    enhanced : ndarray
    """
    try:
        import torch
        from denoiser import pretrained
        from denoiser.dsp import convert_audio

        # Load pretrained DNS model
        model = pretrained.dns64()
        model.eval()

        # Prepare tensor: (batch, channels, time)
        wav = torch.from_numpy(signal).float().unsqueeze(0).unsqueeze(0)

        # The model expects 16kHz mono
        if fs != model.sample_rate:
            wav = convert_audio(wav, fs, model.sample_rate, model.chin)

        with torch.no_grad():
            enhanced = model(wav)

        enhanced_np = enhanced.squeeze().cpu().numpy()

        # Match length
        if len(enhanced_np) > len(signal):
            enhanced_np = enhanced_np[:len(signal)]
        elif len(enhanced_np) < len(signal):
            enhanced_np = np.pad(enhanced_np, (0, len(signal) - len(enhanced_np)))

        return enhanced_np

    except Exception as e:
        print(f"    [neural] denoiser failed ({e}), using spectral envelope enhancement")
        return spectral_envelope_enhance(signal, fs)


def spectral_envelope_enhance(signal, fs):
    """
    Spectral envelope enhancement as a fallback for neural enhancement.

    Applies frequency-dependent gain shaping to emphasize speech formant
    regions (300-3000 Hz) while attenuating non-speech frequencies.

    Parameters
    ----------
    signal : ndarray
    fs : int

    Returns
    -------
    enhanced : ndarray
    """
    frame_len = int(0.025 * fs)  # 25 ms frames
    hop_len = int(0.010 * fs)    # 10 ms hop

    n_orig = len(signal)
    n_frames = max(1, 1 + (n_orig - frame_len) // hop_len)
    pad_len = (n_frames - 1) * hop_len + frame_len
    x = np.zeros(pad_len)
    x[:n_orig] = signal

    win = np.hanning(frame_len)
    win_sq = win ** 2
    nfft = frame_len
    freq_bins = nfft // 2 + 1

    # Build frequency-dependent gain curve
    # Emphasize 300-3000 Hz (speech formants), attenuate below 80 Hz and above 4000 Hz
    freqs = np.linspace(0, fs / 2, freq_bins)
    gain_curve = np.ones(freq_bins)

    # Low-frequency roll-off
    for i, f in enumerate(freqs):
        if f < 80:
            gain_curve[i] = 0.3
        elif f < 300:
            gain_curve[i] = 0.3 + 0.7 * (f - 80) / (300 - 80)
        elif f <= 3000:
            gain_curve[i] = 1.0
        elif f <= 5000:
            gain_curve[i] = 1.0 - 0.5 * (f - 3000) / (5000 - 3000)
        else:
            gain_curve[i] = 0.5

    # Apply
    output = np.zeros(pad_len)
    win_sum = np.zeros(pad_len)

    for i in range(n_frames):
        start = i * hop_len
        frame = x[start:start + frame_len] * win
        F = np.fft.rfft(frame, n=nfft)
        F_enhanced = F * gain_curve
        frame_out = np.fft.irfft(F_enhanced, n=nfft)
        output[start:start + frame_len] += frame_out * win
        win_sum[start:start + frame_len] += win_sq

    win_sum = np.maximum(win_sum, 1e-8)
    output = output / win_sum

    return output[:n_orig]


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------
def normalize_signal(signal):
    """Normalize signal to [-1, 1] range."""
    peak = np.max(np.abs(signal))
    if peak > 0:
        return signal / peak
    return signal


def save_wav(signal, fs, path):
    """Save signal as WAV file."""
    # Normalize before saving
    normed = normalize_signal(signal)
    sf.write(path, normed, fs)
    print(f"    -> Saved: {path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(input_path, output_path, axis="z", gap_mode="uniform",
                 skip_neural=False):
    """
    Execute the full IMU-to-audio pipeline.

    Parameters
    ----------
    input_path : str
        Path to the input CSV file.
    output_path : str
        Path for the output 16kHz WAV file.
    axis : str
        Accelerometer axis to use ('x', 'y', or 'z').
    gap_mode : str
        Gap handling mode: 'chunk', 'uniform', or 'all'.
    skip_neural : bool
        If True, skip the neural enhancement step.
    """
    print("=" * 60)
    print("  IMU-to-Audio Pipeline")
    print("=" * 60)
    print(f"  Input:       {input_path}")
    print(f"  Output:      {output_path}")
    print(f"  Axis:        {axis}")
    print(f"  Gap mode:    {gap_mode}")
    print(f"  Neural:      {'skip' if skip_neural else 'enabled'}")
    print("=" * 60)

    total_start = time.perf_counter()
    output_dir = os.path.dirname(output_path) or "."
    base_name = os.path.splitext(os.path.basename(output_path))[0]

    # Stage 1: Load CSV
    with PipelineTimer("Stage 1: Load CSV"):
        timestamps, signal = load_imu_csv(input_path, axis=axis)
        print(f"    Loaded {len(signal)} samples, "
              f"duration ~{(timestamps[-1] - timestamps[0]) / 1000:.2f}s")

    # Stage 2: Gap-aware resampling
    with PipelineTimer("Stage 2: Gap-aware resample"):
        signal, inferred_sr = gap_aware_resample(timestamps, signal, mode=gap_mode)
        print(f"    Inferred SR: {inferred_sr:.1f} Hz, "
              f"{len(signal)} samples after resampling")

    # Remove DC offset
    signal = signal - np.mean(signal)

    # Stage 3: Bandpass filter
    with PipelineTimer("Stage 3: Bandpass filter (50-1500 Hz)"):
        signal_bp = apply_bandpass(signal, inferred_sr,
                                   low=BANDPASS_LOW, high=BANDPASS_HIGH)
        # Save intermediate
        intermediate_bp = os.path.join(output_dir, f"{base_name}_01_bandpass.wav")
        # Save at inferred SR for now (pre-upsample)
        save_wav(signal_bp, int(np.round(inferred_sr)), intermediate_bp)

    # Stage 4: MMSE-LSA denoising
    with PipelineTimer("Stage 4: MMSE-LSA denoise"):
        signal_mmse = mmse_lsa_denoise(signal_bp, inferred_sr)
        intermediate_mmse = os.path.join(output_dir, f"{base_name}_02_mmse_lsa.wav")
        save_wav(signal_mmse, int(np.round(inferred_sr)), intermediate_mmse)

    # Stage 5: Upsample to 16 kHz
    with PipelineTimer("Stage 5: Upsample to 16 kHz"):
        signal_up = upsample_to_target(signal_mmse, inferred_sr, TARGET_SR)
        intermediate_up = os.path.join(output_dir, f"{base_name}_03_upsample16k.wav")
        save_wav(signal_up, TARGET_SR, intermediate_up)

    # Stage 6: Neural enhancement (optional)
    if not skip_neural:
        with PipelineTimer("Stage 6: Neural enhancement"):
            signal_final = neural_enhance(signal_up, TARGET_SR)
            intermediate_neural = os.path.join(
                output_dir, f"{base_name}_04_neural.wav")
            save_wav(signal_final, TARGET_SR, intermediate_neural)
    else:
        print("  [Stage 6: Neural enhancement] SKIPPED (--skip-neural)")
        signal_final = signal_up

    # Save final output
    save_wav(signal_final, TARGET_SR, output_path)

    total_elapsed = time.perf_counter() - total_start
    print("=" * 60)
    print(f"  Pipeline complete in {total_elapsed:.3f}s")
    print(f"  Final output: {output_path} "
          f"({len(signal_final)} samples, {len(signal_final)/TARGET_SR:.2f}s @ 16kHz)")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="IMU-to-Audio Pipeline: Convert 3.3kHz IMU data to 16kHz WAV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python imu_pipeline.py recording.csv
  python imu_pipeline.py --input recording.csv --output enhanced.wav --axis z
  python imu_pipeline.py recording.csv --gap-mode chunk --skip-neural
        """,
    )

    parser.add_argument(
        "csv_input", nargs="?", default=None,
        help="Input CSV file (positional argument)")
    parser.add_argument(
        "--input", "-i", dest="input_flag", default=None,
        help="Input CSV file (alternative to positional arg)")
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output WAV file path (default: <input_stem>_pipeline.wav)")
    parser.add_argument(
        "--axis", default="z", choices=["x", "y", "z"],
        help="Accelerometer axis to use (default: z)")
    parser.add_argument(
        "--gap-mode", default="uniform", choices=["chunk", "uniform", "all"],
        help="Gap handling mode (default: uniform)")
    parser.add_argument(
        "--skip-neural", action="store_true",
        help="Skip the neural enhancement step")

    args = parser.parse_args()

    # Resolve input path
    input_path = args.csv_input or args.input_flag
    if input_path is None:
        parser.error("Input CSV file required (positional or --input)")

    if not os.path.isfile(input_path):
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Resolve output path
    if args.output:
        output_path = args.output
    else:
        stem = os.path.splitext(os.path.basename(input_path))[0]
        output_path = f"{stem}_pipeline.wav"

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    run_pipeline(
        input_path=input_path,
        output_path=output_path,
        axis=args.axis,
        gap_mode=args.gap_mode,
        skip_neural=args.skip_neural,
    )


if __name__ == "__main__":
    main()
