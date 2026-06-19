#!/usr/bin/env python3
"""Preprocessing demonstration and hyperparameter verification.

Takes a single .wav file, runs the full IMU preprocessing pipeline step-by-step,
and generates a multi-panel plot that visually proves each DSP choice:

 1. Raw waveform (time domain)
 2. After high-pass Butterworth filter (HP_CUTOFF=25 Hz)
 3. After peak normalization
 4. After crop_max_energy (highest-energy fixed window)
 5. STFT spectrogram of the processed signal (annotated with Nyquist, HP cutoff)
 6. Log-mel spectrogram (40 bins, fmin/fmax annotated)
 7. Per-frame energy distribution histogram

All hyperparameter values are annotated directly on the plots so you can verify
that the DSP choices are correct for the 3333 Hz IMU signal.

Example
-------
    python bcresnet_imu_kws/preproc_demo.py --wav recording.wav --output demo.png
"""

import argparse
import sys
import os

import numpy as np


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Visualize the IMU preprocessing pipeline step-by-step with "
                    "annotated hyperparameter values.")
    parser.add_argument("--wav", "-w", required=True,
                        help="Path to the input .wav file (mono, any sample rate; "
                             "resampled to 3333 Hz if needed).")
    parser.add_argument("--output", "-o", default="preproc_demo.png",
                        help="Path to save the output plot image (default: preproc_demo.png).")
    parser.add_argument("--window-seconds", type=float, default=2.0,
                        help="Crop window duration in seconds (default: 2.0).")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Output image DPI (default: 150).")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Lazy imports so --help is instant
    import torch
    import torchaudio
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    # Add the package to path so imports work when run from repo root
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

    from imu_kws.preprocessing import (
        apply_hp, norm_peak, crop_max_energy,
        HP_CUTOFF, FMIN, FMAX, N_FFT, HOP, WIN_LENGTH,
    )
    from imu_kws.dataset import TARGET_SR

    # ---- Load wav ----
    import soundfile as sf_load
    raw_data, sr = sf_load.read(args.wav, dtype="float32")
    # Handle multi-channel: average to mono
    if raw_data.ndim > 1:
        raw_data = raw_data.mean(axis=1)
    wav_tensor = torch.from_numpy(raw_data).unsqueeze(0)  # [1, L]
    if sr != TARGET_SR:
        wav_tensor = torchaudio.functional.resample(wav_tensor, orig_freq=sr, new_freq=TARGET_SR)
    raw = wav_tensor.squeeze(0).numpy().astype(np.float32)

    window_samples = int(args.window_seconds * TARGET_SR)

    # ---- Step-by-step preprocessing ----
    hp_filtered = apply_hp(raw, sample_rate=TARGET_SR, cutoff=HP_CUTOFF)
    normalized = norm_peak(hp_filtered)
    cropped = crop_max_energy(normalized, window_samples, sample_rate=TARGET_SR)

    # ---- Create the plot (4 rows x 2 columns = 8 panels, use 7+) ----
    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    fig.suptitle(
        "IMU Preprocessing Pipeline Verification\n"
        "File: %s | TARGET_SR = %d Hz" % (os.path.basename(args.wav), TARGET_SR),
        fontsize=13, fontweight="bold", y=0.995)

    # Helper: time axis
    def t_axis(x):
        return np.arange(len(x)) / TARGET_SR

    # --- Panel 1: Raw waveform ---
    ax = axes[0, 0]
    t = t_axis(raw)
    ax.plot(t, raw, linewidth=0.4, color="steelblue")
    ax.set_title("1. Raw Waveform", fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(t[0], t[-1])
    ax.annotate(
        "Duration: %.3f s\nSamples: %d\nSR: %d Hz" % (len(raw) / TARGET_SR, len(raw), TARGET_SR),
        xy=(0.02, 0.95), xycoords="axes fraction", fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))

    # --- Panel 2: After HP filter ---
    ax = axes[0, 1]
    t = t_axis(hp_filtered)
    ax.plot(t, hp_filtered, linewidth=0.4, color="darkorange")
    ax.set_title("2. After High-Pass Filter", fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(t[0], t[-1])
    ax.annotate(
        "Butterworth order=4\nCutoff: %.0f Hz\n(Removes gravity DC + body motion)" % HP_CUTOFF,
        xy=(0.02, 0.95), xycoords="axes fraction", fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))

    # --- Panel 3: After peak normalization ---
    ax = axes[1, 0]
    t = t_axis(normalized)
    ax.plot(t, normalized, linewidth=0.4, color="seagreen")
    ax.set_title("3. After Peak Normalization", fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (peak=1.0)")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-1.1, 1.1)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(-1.0, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.annotate(
        "Peak normalized to [-1, 1]\n(Single scalar gain per utterance)",
        xy=(0.02, 0.95), xycoords="axes fraction", fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))

    # --- Panel 4: After crop_max_energy ---
    ax = axes[1, 1]
    t = t_axis(cropped)
    ax.plot(t, cropped, linewidth=0.4, color="purple")
    ax.set_title("4. After crop_max_energy", fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(t[0], t[-1])
    ax.annotate(
        "Window: %.2f s (%d samples)\n"
        "Selects highest-energy segment\n"
        "(No dilution with silence/padding)" % (args.window_seconds, window_samples),
        xy=(0.02, 0.95), xycoords="axes fraction", fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))

    # --- Panel 5: STFT Spectrogram of processed signal ---
    ax = axes[2, 0]
    # Compute STFT
    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / TARGET_SR)
    n_frames = 1 + (len(cropped) - WIN_LENGTH) // HOP
    if n_frames > 0:
        spec = np.zeros((len(freqs), n_frames), dtype=np.float64)
        window = np.hanning(WIN_LENGTH)
        for i in range(n_frames):
            start = i * HOP
            frame = cropped[start:start + WIN_LENGTH]
            # Apply window, then zero-pad to N_FFT for FFT
            windowed = frame * window
            spec[:, i] = np.abs(np.fft.rfft(windowed, n=N_FFT)) ** 2
        spec_db = 10.0 * np.log10(spec + 1e-10)
        t_spec = np.arange(n_frames) * HOP / TARGET_SR
        im = ax.pcolormesh(t_spec, freqs, spec_db, shading="auto", cmap="inferno")
        fig.colorbar(im, ax=ax, label="Power (dB)")
    ax.axhline(TARGET_SR / 2.0, color="cyan", linestyle="--", linewidth=1.2,
               label="Nyquist = %d Hz" % (TARGET_SR // 2))
    ax.axhline(HP_CUTOFF, color="lime", linestyle="--", linewidth=1.2,
               label="HP cutoff = %.0f Hz" % HP_CUTOFF)
    ax.set_title("5. STFT Spectrogram (processed)", fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.legend(loc="upper right", fontsize=8)
    ax.annotate(
        "N_FFT=%d | WIN=%d (~%.1f ms)\nHOP=%d (~%.1f ms)\nNyquist=%d Hz"
        % (N_FFT, WIN_LENGTH, WIN_LENGTH / TARGET_SR * 1000,
           HOP, HOP / TARGET_SR * 1000, TARGET_SR // 2),
        xy=(0.02, 0.95), xycoords="axes fraction", fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.6),
        color="white")

    # --- Panel 6: Log-Mel Spectrogram ---
    ax = axes[2, 1]
    # Build mel filterbank using torchaudio
    mel_spec_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SR, n_fft=N_FFT, win_length=WIN_LENGTH,
        hop_length=HOP, n_mels=40, f_min=FMIN, f_max=FMAX, power=2.0)
    cropped_tensor = torch.from_numpy(cropped).unsqueeze(0)  # [1, T]
    mel_spec = mel_spec_transform(cropped_tensor)  # [1, 40, frames]
    log_mel = torch.log(mel_spec + 1e-6).squeeze(0).numpy()
    # Normalize for display
    log_mel_norm = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-5)
    n_mel_frames = log_mel.shape[1]
    t_mel = np.arange(n_mel_frames) * HOP / TARGET_SR
    mel_bins = np.arange(41)  # edges for 40 bins
    im2 = ax.pcolormesh(t_mel, mel_bins[:-1], log_mel_norm, shading="auto", cmap="magma")
    fig.colorbar(im2, ax=ax, label="Log-Mel (normalized)")
    ax.set_title("6. Log-Mel Spectrogram (40 bins)", fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel Bin")
    ax.annotate(
        "n_mels=40 | fmin=%.0f Hz | fmax=%.0f Hz\n"
        "Band-limited to informative IMU range\n"
        "Per-utterance mean/var normalized" % (FMIN, FMAX),
        xy=(0.02, 0.95), xycoords="axes fraction", fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.6),
        color="white")

    # --- Panel 7: Per-frame energy distribution ---
    ax = axes[3, 0]
    # Compute per-frame energy
    n_energy_frames = max(1, (len(cropped) - WIN_LENGTH) // HOP + 1)
    frame_energies = np.array([
        np.sum(cropped[i * HOP:i * HOP + WIN_LENGTH] ** 2)
        for i in range(n_energy_frames)
    ])
    frame_energies_db = 10.0 * np.log10(frame_energies + 1e-10)
    ax.hist(frame_energies_db, bins=50, color="teal", alpha=0.7, edgecolor="black",
            linewidth=0.3)
    ax.axvline(np.median(frame_energies_db), color="red", linestyle="--", linewidth=1.5,
               label="Median = %.1f dB" % np.median(frame_energies_db))
    ax.set_title("7. Per-Frame Energy Distribution", fontweight="bold")
    ax.set_xlabel("Frame Energy (dB)")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right", fontsize=9)
    ax.annotate(
        "Frame size: %d samples (~%.1f ms)\n"
        "Total frames: %d\n"
        "crop_max_energy concentrates energy\n"
        "in the selected window"
        % (WIN_LENGTH, WIN_LENGTH / TARGET_SR * 1000, n_energy_frames),
        xy=(0.02, 0.95), xycoords="axes fraction", fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))

    # --- Panel 8: Summary / parameter table ---
    ax = axes[3, 1]
    ax.axis("off")
    summary_text = (
        "DSP Hyperparameter Summary\n"
        "==========================\n\n"
        "TARGET_SR     = %d Hz\n"
        "HP_CUTOFF     = %.0f Hz (Butterworth, order=4)\n"
        "FMIN          = %.0f Hz (mel lower bound)\n"
        "FMAX          = %.0f Hz (mel upper bound)\n"
        "Nyquist       = %d Hz (SR/2)\n"
        "N_FFT         = %d\n"
        "WIN_LENGTH    = %d samples (~%.1f ms)\n"
        "HOP           = %d samples (~%.1f ms)\n"
        "n_mels        = 40 (required by BC-ResNet)\n"
        "Window        = %.2f s (%d samples)\n"
        "Normalization = peak (per-utterance)\n\n"
        "Pipeline: HP filter -> peak norm -> crop_max_energy\n"
        "Feature:  log(MelSpec) + mean/var norm"
        % (TARGET_SR, HP_CUTOFF, FMIN, FMAX, TARGET_SR // 2,
           N_FFT, WIN_LENGTH, WIN_LENGTH / TARGET_SR * 1000,
           HOP, HOP / TARGET_SR * 1000,
           args.window_seconds, window_samples)
    )
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", alpha=0.9))
    ax.set_title("8. Hyperparameter Summary", fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print("Saved preprocessing demo plot to: %s" % args.output)
    print("  Panels: raw -> HP filter -> peak norm -> crop -> spectrogram -> "
          "log-mel -> energy distribution -> summary")


if __name__ == "__main__":
    main()
