"""Waveform preprocessing front-end (adapted from the user's Modal script).

Numpy/scipy operations applied per clip in the dataset:
  * high-pass Butterworth filter (default 25 Hz) to kill the gravity DC (~1 g) and
    low-frequency body motion that otherwise dominate accel-Z,
  * peak or RMS normalization (a single scalar gain; peak is the safe default,
    RMS can amplify noise on near-silent clips),
  * optional silence trimming (energy VAD),
  * ``crop_max_energy``: take the highest-energy fixed window (or zero-pad if the
    clip is shorter) -> no more diluting the signal with seconds of padding.

All functions are sample-rate aware and produce a fixed-length 1-D float32 array
of ``window_samples``. ``build_preproc(name, window_samples, sample_rate, ...)``
returns one of the named pipelines as a single ``x -> x`` callable.
"""

import numpy as np
from scipy.signal import butter, filtfilt

from .dataset import TARGET_SR

# Mel front-end defaults (used by features.py). Accelerometer/bone-conduction voice
# pickup has useful content well above 500 Hz but the high end is attenuated/noisy.
# Default to the (almost) full usable band 40-1600 Hz (Nyquist = 1666 Hz at 3.3 kHz);
# you can narrow f_max (e.g. 1000/1200) if the top turns out to be mostly noise.
FMIN = 40.0
FMAX = 1600.0
N_FFT = 512
HOP = 64           # ~19 ms @ 3333 Hz (STFT step)
WIN_LENGTH = 128   # ~38 ms @ 3333 Hz (STFT analysis window; << old implicit 154 ms)
# High-pass kills the gravity DC (~1 g) and low-frequency body motion (< ~20 Hz)
# that otherwise dominate accel-Z. Keep it just below FMIN.
HP_CUTOFF = 25.0


def _hp_coeffs(sample_rate, cutoff=HP_CUTOFF, order=4):
    return butter(order, cutoff / (sample_rate / 2.0), btype="high")


def apply_hp(x, sample_rate=TARGET_SR, cutoff=HP_CUTOFF, order=4):
    b, a = _hp_coeffs(sample_rate, cutoff, order)
    return filtfilt(b, a, x).astype(np.float32)


def norm_peak(x):
    p = np.abs(x).max()
    return (x / p).astype(np.float32) if p > 0 else x.astype(np.float32)


def norm_rms(x, target_rms=0.1):
    r = np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-9
    return (x * (target_rms / r)).astype(np.float32)


def trim_silence(wav, sample_rate=TARGET_SR, frame_ms=30.0, hop_ms=10.0,
                 energy_ratio=0.1, pad_ms=150.0):
    n_frame = max(1, int(sample_rate * frame_ms / 1000))
    n_hop = max(1, int(sample_rate * hop_ms / 1000))
    if len(wav) <= n_frame:
        return wav
    starts = np.arange(0, len(wav) - n_frame + 1, n_hop)
    rms = np.sqrt(np.array([(wav[s:s + n_frame] ** 2).mean() for s in starts]) + 1e-12)
    if rms.max() <= 0:
        return wav
    active = np.where(rms > rms.max() * energy_ratio)[0]
    if active.size == 0:
        return wav
    pad = int(sample_rate * pad_ms / 1000)
    first = max(0, int(starts[active[0]]) - pad)
    last = min(len(wav), int(starts[active[-1]]) + n_frame + pad)
    return wav[first:last]


def crop_max_energy(wav, window_samples, sample_rate=TARGET_SR, smooth_ms=25):
    """Return the highest-energy ``window_samples`` slice, or center-pad if shorter."""
    win = int(window_samples)
    if len(wav) <= win:
        pad = win - len(wav)
        return np.pad(wav, (pad // 2, pad - pad // 2)).astype(np.float32)
    k = max(1, int(smooth_ms / 1000 * sample_rate))
    e = np.convolve(wav.astype(np.float64) ** 2, np.ones(k) / k, mode="same")
    c = np.cumsum(np.insert(e, 0, 0.0))
    win_e = c[win:] - c[:-win]
    start = int(np.argmax(win_e))
    return wav[start:start + win].astype(np.float32)


def build_preproc(name, window_samples, sample_rate=TARGET_SR,
                  hp_cutoff=HP_CUTOFF, target_rms=0.1):
    """Return an ``x -> x`` pipeline that outputs a fixed ``window_samples`` clip."""

    def _hp(x):
        return apply_hp(x, sample_rate, hp_cutoff)

    def _crop(x):
        return crop_max_energy(x, window_samples, sample_rate)

    if name == "hp_peak_crop":
        def pp(x):
            return _crop(norm_peak(_hp(x)))
    elif name == "hp_peak_trim_crop":
        def pp(x):
            return _crop(trim_silence(norm_peak(_hp(x)), sample_rate))
    elif name == "no_hp":
        def pp(x):
            return _crop(norm_peak(x))
    elif name == "hp_rms_crop":
        def pp(x):
            return _crop(norm_rms(_hp(x), target_rms))
    else:
        raise ValueError(
            "unknown preproc '%s' (choices: hp_peak_crop, hp_peak_trim_crop, no_hp, hp_rms_crop)" % name)
    return pp


PREPROC_CHOICES = ("hp_peak_crop", "hp_peak_trim_crop", "no_hp", "hp_rms_crop")
