#!/usr/bin/env python3
"""
IMU Volume Boost Utility - Make Bone Conduction Audio Loud Enough for ASR
=========================================================================
Takes WAV or CSV input from bone conduction IMU processing and outputs
MUCH LOUDER versions that ASR models (Whisper, AssemblyAI) will not ignore.

The problem: bone conduction output typically has RMS of -40 to -60 dBFS,
which ASR treats as silence. This script boosts to broadcast loudness (-16 dBFS)
and provides multiple output levels for testing.

Pipeline for CSV input:
  1. Gap-aware resampling (infer rate from timestamps, uniform interpolation)
  2. Bandpass filter 50-1500 Hz (Butterworth order 4)
  3. MMSE-LSA denoising (Ephraim & Malah 1985)
  4. AGC (35 dB max gain)
  5. Upsample to 16 kHz (polyphase)

Volume boost methods:
  - RMS normalization to target level (default -16 dBFS)
  - Multi-pass dynamic range compression (10:1 ratio) + makeup gain
  - Hard limiter / maximizer (brickwall at 0.99)
  - Multiple output levels (+6, +12, +20, +30 dB, maximized, RMS normalized)

Usage:
  python imu_boost_volume.py input.wav
  python imu_boost_volume.py input.csv --target-rms -16 --gain-db 6
  python imu_boost_volume.py --input recording.csv --axis z
"""

import argparse
import os
import sys
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
AGC_MAX_GAIN_DB = 35     # Maximum AGC gain


# ===========================================================================
# CSV PREPROCESSING PIPELINE
# ===========================================================================

def load_imu_csv(path, axis="z"):
    """Load IMU CSV with columns: timestamp, Accel X, Accel Y, Accel Z."""
    import pandas as pd

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Find timestamp column
    ts_col = None
    for candidate in ["timestamp", "time", "ts", "t"]:
        if candidate in df.columns:
            ts_col = candidate
            break
    if ts_col is None:
        ts_col = df.columns[0]

    timestamps = df[ts_col].values.astype(np.float64)

    # Find axis column
    axis_lower = axis.lower()
    signal_col = None
    for col in df.columns:
        if f"accel {axis_lower}" in col or f"accel_{axis_lower}" in col:
            signal_col = col
            break
    if signal_col is None:
        axis_map = {"x": 1, "y": 2, "z": 3}
        idx = axis_map.get(axis_lower, 3)
        signal_col = df.columns[idx] if idx < len(df.columns) else df.columns[-1]

    signal = df[signal_col].values.astype(np.float64)
    return timestamps, signal


def gap_aware_resample(timestamps, signal):
    """
    Gap-aware resampling: infer rate from timestamps, uniform interpolation.
    """
    diffs = np.diff(timestamps)
    if len(diffs) == 0:
        return signal, ORIGINAL_SR

    median_dt = np.median(diffs)
    inferred_sr = 1000.0 / median_dt  # timestamps assumed in ms

    # Interpolate onto uniform grid
    duration_ms = timestamps[-1] - timestamps[0]
    n_samples = int(np.round(duration_ms * inferred_sr / 1000.0))
    if n_samples < 2:
        return signal, inferred_sr

    uniform_ts = np.linspace(timestamps[0], timestamps[-1], n_samples)
    resampled = np.interp(uniform_ts, timestamps, signal)
    return resampled, inferred_sr


def apply_bandpass(signal, fs, low=BANDPASS_LOW, high=BANDPASS_HIGH,
                   order=BUTTER_ORDER):
    """Apply Butterworth bandpass filter."""
    nyquist = fs / 2.0
    high_norm = min(high, nyquist * 0.99)
    low_norm = max(low, 1.0)

    if low_norm >= high_norm:
        return signal

    sos = butter(order, [low_norm / nyquist, high_norm / nyquist],
                 btype="band", output="sos")
    return sosfilt(sos, signal)


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
    n_orig = len(signal)
    n_frames = max(1, 1 + (n_orig - frame_len) // hop_len)
    pad_len = max(n_orig, (n_frames - 1) * hop_len + frame_len)
    n_frames = max(1, 1 + (pad_len - frame_len) // hop_len)
    x = np.zeros(pad_len)
    x[:n_orig] = signal

    nfft = frame_len
    freq_bins = nfft // 2 + 1

    # STFT
    X = np.zeros((n_frames, freq_bins), dtype=np.complex128)
    for i in range(n_frames):
        start = i * hop_len
        frame = x[start:start + frame_len] * win
        X[i, :] = np.fft.rfft(frame, n=nfft)

    P = np.abs(X) ** 2

    # Noise estimation from initial frames
    n_noise = min(noise_frames, n_frames)
    noise_psd = np.mean(P[:n_noise, :], axis=0) + 1e-10

    G_floor = 10.0 ** (floor_db / 20.0)

    # Process frames
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

    # Inverse STFT (overlap-add)
    output = np.zeros(pad_len)
    win_sum = np.zeros(pad_len)
    for i in range(n_frames):
        start = i * hop_len
        frame = np.fft.irfft(enhanced_X[i, :], n=nfft)
        output[start:start + frame_len] += frame * win
        win_sum[start:start + frame_len] += win ** 2

    # Normalize by window sum
    mask = win_sum > 1e-8
    output[mask] /= win_sum[mask]

    return output[:n_orig]


def apply_agc(signal, max_gain_db=AGC_MAX_GAIN_DB, target_rms=0.1):
    """
    Automatic Gain Control - boost signal level up to max_gain_db.
    """
    rms = np.sqrt(np.mean(signal ** 2)) + 1e-10
    desired_gain = target_rms / rms
    max_gain = 10.0 ** (max_gain_db / 20.0)
    gain = min(desired_gain, max_gain)
    return signal * gain


def upsample_to_16k(signal, fs_in):
    """Upsample to 16 kHz using polyphase resampling."""
    if abs(fs_in - TARGET_SR) < 1.0:
        return signal

    # Find rational ratio
    fs_in_int = int(np.round(fs_in))
    from math import gcd
    g = gcd(TARGET_SR, fs_in_int)
    up = TARGET_SR // g
    down = fs_in_int // g

    # Limit to reasonable factors
    if up > 1000 or down > 1000:
        # Fallback: simple interpolation
        n_out = int(len(signal) * TARGET_SR / fs_in)
        x_old = np.linspace(0, 1, len(signal))
        x_new = np.linspace(0, 1, n_out)
        return np.interp(x_new, x_old, signal)

    return resample_poly(signal, up, down)


def preprocess_csv(path, axis="z"):
    """
    Full CSV preprocessing pipeline:
      1. Load CSV
      2. Gap-aware resample
      3. Bandpass 50-1500 Hz
      4. MMSE-LSA denoise
      5. AGC (35 dB max)
      6. Upsample to 16 kHz
    Returns (signal_16k, 16000).
    """
    print("  [CSV] Loading IMU data...")
    timestamps, raw = load_imu_csv(path, axis=axis)
    print(f"  [CSV] Loaded {len(raw)} samples, axis={axis}")

    print("  [CSV] Gap-aware resampling...")
    resampled, inferred_sr = gap_aware_resample(timestamps, raw)
    print(f"  [CSV] Inferred sample rate: {inferred_sr:.1f} Hz, {len(resampled)} samples")

    # Remove DC offset
    resampled = resampled - np.mean(resampled)

    print("  [CSV] Bandpass filter 50-1500 Hz...")
    filtered = apply_bandpass(resampled, inferred_sr)

    print("  [CSV] MMSE-LSA denoising...")
    denoised = mmse_lsa_denoise(filtered, inferred_sr)

    print("  [CSV] Applying AGC (max 35 dB)...")
    gained = apply_agc(denoised)

    print("  [CSV] Upsampling to 16 kHz...")
    signal_16k = upsample_to_16k(gained, inferred_sr)
    print(f"  [CSV] Output: {len(signal_16k)} samples at 16 kHz")

    return signal_16k, TARGET_SR


# ===========================================================================
# VOLUME BOOST FUNCTIONS
# ===========================================================================

def compute_rms_dbfs(signal):
    """Compute RMS level in dBFS."""
    rms = np.sqrt(np.mean(signal ** 2))
    if rms < 1e-10:
        return -100.0
    return 20.0 * np.log10(rms)


def brickwall_limiter(signal, threshold=0.99):
    """Hard clip signal at threshold to prevent digital clipping artifacts."""
    return np.clip(signal, -threshold, threshold)


def rms_normalize(signal, target_dbfs=-16.0):
    """
    Normalize signal so its RMS matches the target dBFS level.
    Standard broadcast loudness is -16 dBFS.

    If the signal has a very high crest factor (peaks >> RMS), we first
    compress dynamics so that RMS normalization can reach the target
    without excessive clipping.
    """
    current_rms = np.sqrt(np.mean(signal ** 2))
    if current_rms < 1e-10:
        return signal

    target_rms = 10.0 ** (target_dbfs / 20.0)
    gain = target_rms / current_rms
    peak_after = np.max(np.abs(signal)) * gain

    # If gain would cause heavy clipping (peak > 2x threshold), compress first
    if peak_after > 2.0:
        # Compress to reduce crest factor, then normalize
        compressed = multipass_compress_and_boost(signal, passes=2,
                                                  threshold_db=-25.0,
                                                  ratio=8.0,
                                                  target_dbfs=target_dbfs)
        # Fine-tune RMS to exact target
        comp_rms = np.sqrt(np.mean(compressed ** 2))
        if comp_rms > 1e-10:
            compressed = compressed * (target_rms / comp_rms)
        return brickwall_limiter(compressed)

    boosted = signal * gain
    return brickwall_limiter(boosted)


def apply_gain_db(signal, gain_db):
    """Apply a fixed gain in dB, then brickwall limit."""
    gain_linear = 10.0 ** (gain_db / 20.0)
    boosted = signal * gain_linear
    return brickwall_limiter(boosted)


def dynamic_range_compress(signal, threshold_db=-20.0, ratio=10.0,
                           attack_ms=5.0, release_ms=50.0, fs=16000):
    """
    Multi-pass dynamic range compressor with aggressive ratio.

    Parameters
    ----------
    signal : ndarray
        Input signal (normalized to [-1, 1] range).
    threshold_db : float
        Compression threshold in dB.
    ratio : float
        Compression ratio (10:1 = very aggressive).
    attack_ms : float
        Attack time in milliseconds.
    release_ms : float
        Release time in milliseconds.
    fs : int
        Sample rate.

    Returns
    -------
    compressed : ndarray
    """
    threshold_lin = 10.0 ** (threshold_db / 20.0)
    attack_coeff = np.exp(-1.0 / (attack_ms * fs / 1000.0))
    release_coeff = np.exp(-1.0 / (release_ms * fs / 1000.0))

    envelope = np.zeros(len(signal))
    env = 0.0

    # Envelope follower
    abs_signal = np.abs(signal)
    for i in range(len(signal)):
        if abs_signal[i] > env:
            env = attack_coeff * env + (1.0 - attack_coeff) * abs_signal[i]
        else:
            env = release_coeff * env + (1.0 - release_coeff) * abs_signal[i]
        envelope[i] = env

    # Apply compression
    compressed = signal.copy()
    for i in range(len(signal)):
        if envelope[i] > threshold_lin:
            # Gain reduction
            over_db = 20.0 * np.log10(envelope[i] / threshold_lin + 1e-10)
            reduction_db = over_db * (1.0 - 1.0 / ratio)
            gain = 10.0 ** (-reduction_db / 20.0)
            compressed[i] = signal[i] * gain

    return compressed


def multipass_compress_and_boost(signal, fs=16000, passes=3,
                                 threshold_db=-20.0, ratio=10.0,
                                 target_dbfs=-6.0):
    """
    Multi-pass compression + makeup gain.
    Compress dynamic range aggressively, then apply makeup gain to bring
    everything up to near 0 dBFS.
    """
    result = signal.copy()

    for p in range(passes):
        result = dynamic_range_compress(result, threshold_db=threshold_db,
                                        ratio=ratio, fs=fs)
        # Makeup gain after each pass to restore level
        current_rms = np.sqrt(np.mean(result ** 2))
        if current_rms > 1e-10:
            target_rms = 10.0 ** (target_dbfs / 20.0)
            gain = target_rms / current_rms
            result = result * gain

    # Final brickwall limit
    return brickwall_limiter(result)


def maximizer(signal, target_peak=0.99):
    """
    Absolute maximum loudness - like a mastering maximizer.
    Boost gain until peak hits target, with multi-pass compression first.
    """
    # First compress heavily
    compressed = multipass_compress_and_boost(signal, passes=3,
                                              threshold_db=-30.0,
                                              ratio=20.0,
                                              target_dbfs=-3.0)

    # Then peak normalize to target
    peak = np.max(np.abs(compressed))
    if peak < 1e-10:
        return compressed

    gain = target_peak / peak
    maximized = compressed * gain

    # Ensure minimum amplitude requirement
    if np.max(np.abs(maximized)) < 0.5:
        maximized = maximized * (0.5 / (np.max(np.abs(maximized)) + 1e-10))
        maximized = brickwall_limiter(maximized)

    return maximized


def save_wav_16bit(signal, path, fs=16000):
    """
    Save signal as 16-bit PCM WAV.
    Ensures signal peak is at least 0.5 for ASR compatibility.
    This guarantees that ASR models see sufficient amplitude.
    """
    out = signal.copy()
    peak = np.max(np.abs(out))

    # If peak is very low, boost so ASR does not treat it as silence
    if peak < 0.5 and peak > 1e-10:
        out = out * (0.5 / peak)
        out = brickwall_limiter(out)

    # Final safety clip
    out = np.clip(out, -1.0, 1.0)

    sf.write(path, out.astype(np.float64), fs, subtype='PCM_16')


def print_levels(label, signal):
    """Print RMS and peak levels for a signal."""
    rms_db = compute_rms_dbfs(signal)
    peak = np.max(np.abs(signal))
    peak_db = 20.0 * np.log10(peak) if peak > 1e-10 else -100.0
    print(f"  {label}: RMS={rms_db:.1f} dBFS, Peak={peak_db:.1f} dBFS (peak_lin={peak:.4f})")


# ===========================================================================
# MAIN PROCESSING
# ===========================================================================

def process_input(input_path, output_dir, target_rms_dbfs=-16.0,
                  extra_gain_db=0.0, axis="z"):
    """
    Main processing function.

    Parameters
    ----------
    input_path : str
        Path to WAV or CSV file.
    output_dir : str
        Output directory for boosted files.
    target_rms_dbfs : float
        Target RMS in dBFS for RMS normalization (default: -16).
    extra_gain_db : float
        Additional gain to apply on top of normalization.
    axis : str
        IMU axis for CSV input (default: z).
    """
    os.makedirs(output_dir, exist_ok=True)

    ext = os.path.splitext(input_path)[1].lower()
    basename = os.path.splitext(os.path.basename(input_path))[0]

    # Load input
    if ext == ".csv":
        print(f"\n{'='*60}")
        print(f"INPUT: {input_path} (CSV - running full preprocessing pipeline)")
        print(f"{'='*60}")
        signal, fs = preprocess_csv(input_path, axis=axis)
    elif ext in (".wav", ".flac", ".ogg", ".mp3"):
        print(f"\n{'='*60}")
        print(f"INPUT: {input_path} (audio file)")
        print(f"{'='*60}")
        signal, fs = sf.read(input_path, dtype='float64')
        # Convert stereo to mono if needed
        if signal.ndim > 1:
            signal = np.mean(signal, axis=1)
        print(f"  Loaded: {len(signal)} samples at {fs} Hz")
    else:
        print(f"ERROR: Unsupported file type '{ext}'. Use .wav or .csv")
        sys.exit(1)

    # Normalize to [-1, 1]
    peak = np.max(np.abs(signal))
    if peak > 1e-10:
        signal = signal / peak

    # Print input levels
    print(f"\n--- Input Levels ---")
    print_levels("Input", signal)
    input_rms_db = compute_rms_dbfs(signal)

    # Apply extra gain if specified
    if extra_gain_db != 0.0:
        print(f"\n  Applying extra gain: +{extra_gain_db:.1f} dB")
        signal = apply_gain_db(signal, extra_gain_db)
        print_levels("After extra gain", signal)

    # Generate all output versions
    print(f"\n--- Generating Boosted Outputs ---")
    outputs = {}

    # 1. Fixed gain boosts
    for boost_db in [6, 12, 20, 30]:
        label = f"boost_{boost_db}dB"
        boosted = apply_gain_db(signal, boost_db)
        outputs[label] = boosted

    # 2. Maximized (absolute maximum loudness, hard limited)
    outputs["maximized"] = maximizer(signal)

    # 3. RMS normalized to target level
    outputs["rms_normalized"] = rms_normalize(signal, target_dbfs=target_rms_dbfs)

    # Save all outputs and print stats
    print(f"\n--- Output Files ---")
    print(f"  Output directory: {output_dir}")
    print(f"  Target RMS: {target_rms_dbfs:.1f} dBFS")
    print()

    saved_files = []
    for label, out_signal in outputs.items():
        filename = f"{basename}_{label}.wav"
        filepath = os.path.join(output_dir, filename)
        save_wav_16bit(out_signal, filepath, fs=int(fs))
        saved_files.append(filepath)

        # Compute stats
        out_rms_db = compute_rms_dbfs(out_signal)
        gain_applied = out_rms_db - input_rms_db
        peak_val = np.max(np.abs(out_signal))
        print(f"  {filename}")
        print(f"    RMS: {out_rms_db:.1f} dBFS | Peak: {peak_val:.4f} | "
              f"Gain applied: {gain_applied:+.1f} dB")

    print(f"\n{'='*60}")
    print(f"DONE: Saved {len(saved_files)} boosted WAV files to {output_dir}/")
    print(f"{'='*60}")
    print(f"\nTIP: Try '_maximized.wav' first with your ASR. If still too quiet,")
    print(f"     the signal may need better preprocessing (try CSV input mode).")
    print(f"     Standard ASR expects RMS around -20 to -10 dBFS.")

    return saved_files


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Boost volume of bone conduction audio for ASR compatibility.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python imu_boost_volume.py recording.wav
  python imu_boost_volume.py recording.csv --axis z
  python imu_boost_volume.py --input my_audio.wav --target-rms -12
  python imu_boost_volume.py input.wav --gain-db 10 --target-rms -16
        """,
    )

    parser.add_argument("input_file", nargs="?", default=None,
                        help="Input WAV or CSV file (positional)")
    parser.add_argument("--input", "-i", dest="input_flag", default=None,
                        help="Input WAV or CSV file (flag alternative)")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Output directory (default: <input_name>_boosted/)")
    parser.add_argument("--target-rms", type=float, default=-16.0,
                        help="Target RMS in dBFS for normalization (default: -16)")
    parser.add_argument("--gain-db", type=float, default=0.0,
                        help="Additional gain in dB on top of normalization (default: 0)")
    parser.add_argument("--axis", default="z", choices=["x", "y", "z"],
                        help="IMU axis for CSV input (default: z)")

    args = parser.parse_args()

    # Determine input file
    input_path = args.input_flag or args.input_file
    if input_path is None:
        parser.print_help()
        print("\nERROR: No input file specified. Provide a WAV or CSV file.")
        sys.exit(1)

    if not os.path.exists(input_path):
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        basename = os.path.splitext(os.path.basename(input_path))[0]
        output_dir = f"{basename}_boosted"

    # Run processing
    process_input(
        input_path=input_path,
        output_dir=output_dir,
        target_rms_dbfs=args.target_rms,
        extra_gain_db=args.gain_db,
        axis=args.axis,
    )


if __name__ == "__main__":
    main()
