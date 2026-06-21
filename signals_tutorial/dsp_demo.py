#!/usr/bin/env python3
"""
dsp_demo.py - a from-scratch tour of sampling, aliasing, the FFT, and reconstruction.

Each section prints concrete numbers AND saves a figure to ./figures/. Run it top to bottom:

    python dsp_demo.py [path/to/audio.wav_or_flac]

If no audio path is given it falls back to the LibriSpeech clip shipped with this repo.

Deps: numpy, scipy, matplotlib, soundfile.
The whole point: SEE the theory happen on real numbers and a real voice recording.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")                      # headless: write PNGs, no display needed
import matplotlib.pyplot as plt

FIG = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG, exist_ok=True)


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.tight_layout()
    fig.savefig(p, dpi=110)
    plt.close(fig)
    print(f"   saved figure -> {os.path.relpath(p)}")


# ======================================================================== 1. SAMPLING
def demo_sampling():
    print("\n[1] SAMPLING -- turning a continuous wave into numbers")
    f = 5.0                                       # 5 Hz tone
    fs = 40.0                                      # sample 40 times/sec (well above Nyquist=2*f=10)
    t_cont = np.linspace(0, 1, 2000)               # pretend-continuous
    x_cont = np.sin(2 * np.pi * f * t_cont)
    n = np.arange(0, int(fs))                      # sample indices
    t_samp = n / fs
    x_samp = np.sin(2 * np.pi * f * t_samp)
    print(f"   tone f={f} Hz, sample rate fs={fs} Hz, Nyquist={fs/2} Hz")
    print(f"   one period of the tone = {1/f*1000:.0f} ms; spacing between samples = {1/fs*1000:.0f} ms")
    print(f"   -> {fs/f:.0f} samples per period (plenty to capture the shape)")

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(t_cont, x_cont, lw=1, alpha=0.6, label=f"continuous {f} Hz sine")
    ax.stem(t_samp, x_samp, linefmt="C1-", markerfmt="C1o", basefmt=" ", label=f"samples @ {fs} Hz")
    ax.set(xlabel="time (s)", ylabel="amplitude", title="1. Sampling a continuous sine")
    ax.legend(loc="upper right")
    save(fig, "01_sampling.png")


# ======================================================================== 2. ALIASING
def alias_freq(f, fs):
    """The frequency a tone f Hz APPEARS to be after sampling at fs (folding about Nyquist)."""
    f_mod = f % fs
    return f_mod if f_mod <= fs / 2 else fs - f_mod


def demo_aliasing():
    print("\n[2] ALIASING -- too-few samples make a high tone masquerade as a low one")
    fs = 20.0                                      # Nyquist = 10 Hz
    true_f = 18.0                                  # ABOVE Nyquist -> will alias
    a = alias_freq(true_f, fs)
    print(f"   fs={fs} Hz, Nyquist={fs/2} Hz")
    print(f"   true tone = {true_f} Hz (above Nyquist) -> APPEARS as {a} Hz after sampling")
    print(f"   formula: alias = |f - round(f/fs)*fs| = {a} Hz")

    t_cont = np.linspace(0, 1, 4000)
    x_true = np.sin(2 * np.pi * true_f * t_cont)
    x_alias = np.sin(2 * np.pi * a * t_cont)
    n = np.arange(0, int(fs))
    t_s = n / fs
    x_s = np.sin(2 * np.pi * true_f * t_s)

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(t_cont, x_true, lw=1, alpha=0.5, label=f"true {true_f} Hz")
    ax.plot(t_cont, x_alias, "C2--", lw=1.5, alpha=0.9, label=f"alias {a} Hz (what you 'see')")
    ax.stem(t_s, x_s, linefmt="C1-", markerfmt="C1o", basefmt=" ", label=f"samples @ {fs} Hz")
    ax.set(xlabel="time (s)", ylabel="amplitude",
           title=f"2. Aliasing: {true_f} Hz sampled at {fs} Hz looks like {a} Hz")
    ax.legend(loc="upper right")
    save(fig, "02_aliasing.png")

    # the folding map: input frequency vs perceived frequency
    f_in = np.linspace(0, 2 * fs, 1000)
    f_out = np.array([alias_freq(f, fs) for f in f_in])
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(f_in, f_out, lw=2)
    ax.axvline(fs / 2, color="r", ls="--", label="Nyquist")
    ax.axvline(fs, color="k", ls=":", label="fs")
    ax.set(xlabel="true frequency (Hz)", ylabel="perceived frequency (Hz)",
           title="2b. The folding map -- frequencies fold back at Nyquist")
    ax.legend()
    save(fig, "02b_folding.png")


# ======================================================================== 3. FFT BASICS
def demo_fft():
    print("\n[3] FFT -- decomposing a signal into its frequencies")
    fs = 1000.0
    N = 1000                                       # 1 second -> bin width = fs/N = 1 Hz
    t = np.arange(N) / fs
    # three tones with known amplitudes
    comps = [(50, 1.0), (120, 0.5), (300, 0.25)]
    x = sum(a * np.sin(2 * np.pi * f * t) for f, a in comps)

    X = np.fft.rfft(x)                             # real FFT -> 0..Nyquist
    freqs = np.fft.rfftfreq(N, 1 / fs)
    mag = np.abs(X) / (N / 2)                       # scale so a pure sine reads its amplitude
    bin_hz = fs / N
    print(f"   fs={fs} Hz, N={N} samples -> frequency resolution = fs/N = {bin_hz} Hz per bin")
    print(f"   x-axis spans 0..{fs/2:.0f} Hz (Nyquist); {len(freqs)} bins total")
    # report detected peaks
    for f, a in comps:
        k = int(round(f / bin_hz))
        print(f"   tone {f:4.0f} Hz (amp {a:.2f})  -> bin {k:4d}, measured mag {mag[k]:.3f}")

    fig, axes = plt.subplots(2, 1, figsize=(9, 5))
    axes[0].plot(t[:200], x[:200])
    axes[0].set(xlabel="time (s)", ylabel="amp", title="3. Time domain (first 200 ms)")
    axes[1].plot(freqs, mag)
    for f, a in comps:
        axes[1].annotate(f"{f} Hz", (f, mag[int(round(f / bin_hz))]),
                         textcoords="offset points", xytext=(4, 4))
    axes[1].set(xlabel="frequency (Hz)", ylabel="magnitude",
                title="3b. Frequency domain (FFT magnitude) -- peaks at the tones")
    axes[1].set_xlim(0, 400)
    save(fig, "03_fft.png")


# ======================================================================== 4. LEAKAGE / WINDOWING
def demo_windowing():
    print("\n[4] SPECTRAL LEAKAGE -- why we apply a window before the FFT")
    fs = 1000.0
    N = 1000
    t = np.arange(N) / fs
    f = 50.5                                        # NON-integer # of cycles in the window -> leaks
    x = np.sin(2 * np.pi * f * t)
    freqs = np.fft.rfftfreq(N, 1 / fs)

    def spec_db(sig):
        X = np.abs(np.fft.rfft(sig))
        return 20 * np.log10(X / X.max() + 1e-12)

    rect = spec_db(x)
    hann = spec_db(x * np.hanning(N))
    print(f"   tone at {f} Hz does not land on a bin center ({fs/N} Hz grid)")
    print("   -> rectangular (no window): energy SMEARS across many bins")
    print("   -> Hann window: main lobe wider but side-lobes far lower (cleaner peak)")

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(freqs, rect, label="no window (rectangular)", alpha=0.8)
    ax.plot(freqs, hann, label="Hann window", alpha=0.9)
    ax.set(xlabel="frequency (Hz)", ylabel="magnitude (dB)",
           title="4. Spectral leakage: windowing tames the smear", xlim=(0, 150), ylim=(-120, 5))
    ax.legend()
    save(fig, "04_windowing.png")


# ======================================================================== 5. RECONSTRUCTION
def demo_reconstruction():
    print("\n[5] RECONSTRUCTION -- rebuilding the continuous wave from samples (sinc interpolation)")
    f = 3.0
    fs = 10.0                                       # above Nyquist (=6) -> reconstructable
    T = 1.0
    n = np.arange(0, int(fs * T))
    t_s = n / fs
    x_s = np.sin(2 * np.pi * f * t_s)
    t_fine = np.linspace(0, T, 1000)

    # Whittaker-Shannon interpolation: x(t) = sum_n x[n] * sinc((t - nTs)/Ts)
    Ts = 1 / fs
    recon = np.zeros_like(t_fine)
    for ni, xn in zip(n, x_s):
        recon += xn * np.sinc((t_fine - ni * Ts) / Ts)
    truth = np.sin(2 * np.pi * f * t_fine)
    err = np.sqrt(np.mean((recon - truth) ** 2))
    print(f"   tone {f} Hz, fs={fs} Hz (Nyquist {fs/2} Hz) -> below Nyquist, so EXACT recovery")
    print(f"   sinc-interpolation RMS error vs truth = {err:.4f} (near zero)")
    print("   KEY: this only works because the tone was below Nyquist. Content ABOVE Nyquist")
    print("        was destroyed at sampling time and CANNOT be sinc-reconstructed -- you must")
    print("        instead INFER/synthesize it (bandwidth extension), which is the IMU problem.")

    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(t_fine, truth, "C0", lw=1, alpha=0.4, label="true continuous")
    ax.plot(t_fine, recon, "C2--", lw=1.5, label="sinc-reconstructed")
    ax.stem(t_s, x_s, linefmt="C1-", markerfmt="C1o", basefmt=" ", label="samples")
    ax.set(xlabel="time (s)", ylabel="amp", title="5. Perfect reconstruction below Nyquist")
    ax.legend(loc="upper right")
    save(fig, "05_reconstruction.png")


# ======================================================================== 6. REAL .WAV
def demo_wav(path):
    print(f"\n[6] REAL AUDIO -- FFT and spectrogram of {os.path.basename(path)}")
    import soundfile as sf
    from scipy.signal import stft, butter, filtfilt, resample_poly
    x, fs = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float64)
    x = x / (np.max(np.abs(x)) + 1e-9)
    dur = len(x) / fs
    print(f"   sample rate fs={fs} Hz  ->  Nyquist={fs/2:.0f} Hz")
    print(f"   {len(x)} samples = {dur:.2f} s")

    # whole-clip FFT
    X = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    Xdb = 20 * np.log10(X / X.max() + 1e-12)
    peak_f = freqs[np.argmax(X)]
    # energy below vs above 1600 Hz (the IMU ceiling)
    band = freqs <= 1600
    e_low = np.sum(X[band] ** 2)
    e_high = np.sum(X[~band] ** 2)
    print(f"   dominant frequency in clip: {peak_f:.0f} Hz")
    print(f"   energy <=1600 Hz: {100*e_low/(e_low+e_high):.1f}%   "
          f">1600 Hz: {100*e_high/(e_low+e_high):.1f}%  (the band an IMU @3333 Hz would LOSE)")

    fig, axes = plt.subplots(3, 1, figsize=(9, 8))
    axes[0].plot(np.arange(len(x)) / fs, x, lw=0.4)
    axes[0].set(xlabel="time (s)", ylabel="amp", title="6a. Waveform (time domain)")

    axes[1].plot(freqs, Xdb, lw=0.5)
    axes[1].axvline(1600, color="r", ls="--", label="1.6 kHz (IMU Nyquist)")
    axes[1].set(xlabel="frequency (Hz)", ylabel="magnitude (dB)",
                title="6b. FFT magnitude spectrum (whole clip)", xlim=(0, fs / 2))
    axes[1].legend()

    f_, t_, Z = stft(x, fs=fs, nperseg=1024, noverlap=768)
    S = 20 * np.log10(np.abs(Z) + 1e-6)
    im = axes[2].pcolormesh(t_, f_, S, shading="auto", cmap="magma")
    axes[2].axhline(1600, color="cyan", ls="--", lw=1)
    axes[2].set(xlabel="time (s)", ylabel="frequency (Hz)",
                title="6c. Spectrogram (STFT) -- how the spectrum evolves over time")
    fig.colorbar(im, ax=axes[2], label="dB")
    save(fig, "06_wav_analysis.png")

    # band-limit demo: simulate the IMU's loss of the high band
    print("\n   --- simulating the bone-conduction / IMU bandwidth limit ---")
    ny = fs / 2
    b, a = butter(8, min(1600, ny * 0.99) / ny, btype="low")
    x_lp = filtfilt(b, a, x)
    Xlp = np.abs(np.fft.rfft(x_lp))
    Xlpdb = 20 * np.log10(Xlp / X.max() + 1e-12)
    print("   low-passed the clip at 1.6 kHz -> mimics what the 3333 Hz IMU physically captures.")
    print("   the consonant/fricative energy above 1.6 kHz is now GONE (see figure 07).")
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(freqs, Xdb, lw=0.5, alpha=0.5, label="full-band (mic)")
    ax.plot(freqs, Xlpdb, lw=0.6, label="band-limited to 1.6 kHz (IMU-like)")
    ax.axvline(1600, color="r", ls="--")
    ax.set(xlabel="frequency (Hz)", ylabel="dB", title="7. What the IMU bandwidth limit removes",
           xlim=(0, fs / 2), ylim=(-120, 5))
    ax.legend()
    save(fig, "07_bandlimit.png")


def main():
    print("=" * 72)
    print("  SIGNALS & SYSTEMS FROM SCRATCH -- runnable demo")
    print("=" * 72)
    demo_sampling()
    demo_aliasing()
    demo_fft()
    demo_windowing()
    demo_reconstruction()

    default = os.path.join(os.path.dirname(__file__), "..",
                           "src/mlx/mlx_whisper/assets/ls_test.flac")
    path = sys.argv[1] if len(sys.argv) > 1 else default
    if os.path.exists(path):
        demo_wav(path)
    else:
        print(f"\n[6] (skipped: audio file not found at {path})")
    print("\nDone. All figures are in ./figures/")


if __name__ == "__main__":
    main()
