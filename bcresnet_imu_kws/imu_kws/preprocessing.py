"""Waveform preprocessing front-end (adapted from the user's Modal script).

Numpy/scipy operations applied per clip in the dataset:
  * high-pass Butterworth filter (default 25 Hz) to kill the gravity DC (~1 g) and
    low-frequency body motion that otherwise dominate accel-Z,
  * peak or RMS normalization (a single scalar gain; peak is the safe default,
    RMS can amplify noise on near-silent clips),
  * ``crop_max_energy``: take the highest-energy fixed window (or zero/repeat-pad
    if the clip is shorter) -> no more diluting the signal with seconds of padding.

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
HOP = 33            # ~10 ms @ 3333 Hz (STFT step)
WIN_LENGTH = 83    # ~25 ms @ 3333 Hz (STFT analysis window; standard speech frame)
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


def crop_max_energy(wav, window_samples, sample_rate=TARGET_SR, smooth_ms=25, pad_mode="zero"):
    """Return the highest-energy ``window_samples`` slice, or pad if shorter.

    ``pad_mode``:
      * ``"zero"`` (default) — center zero-pad (standard).
      * ``"repeat"`` — tile/loop the signal to fill the window (avoids silence
        dilution for very short clips; every frame sees real content).
    """
    win = int(window_samples)
    if len(wav) <= win:
        if pad_mode == "repeat" and len(wav) > 0:
            repeats = int(np.ceil(win / len(wav)))
            wav = np.tile(wav, repeats)[:win]
            return wav.astype(np.float32)
        # default: center zero-pad
        pad = win - len(wav)
        return np.pad(wav, (pad // 2, pad - pad // 2)).astype(np.float32)
    k = max(1, int(smooth_ms / 1000 * sample_rate))
    e = np.convolve(wav.astype(np.float64) ** 2, np.ones(k) / k, mode="same")
    c = np.cumsum(np.insert(e, 0, 0.0))
    win_e = c[win:] - c[:-win]
    start = int(np.argmax(win_e))
    return wav[start:start + win].astype(np.float32)


def build_preproc(name, window_samples, sample_rate=TARGET_SR,
                  hp_cutoff=HP_CUTOFF, target_rms=0.1, pad_mode="zero"):
    """Return an ``x -> x`` pipeline that outputs a fixed ``window_samples`` clip."""

    def _hp(x):
        return apply_hp(x, sample_rate, hp_cutoff)

    def _crop(x):
        return crop_max_energy(x, window_samples, sample_rate, pad_mode=pad_mode)

    if name == "hp_peak_crop":
        def pp(x):
            return _crop(norm_peak(_hp(x)))
    elif name == "no_hp":
        def pp(x):
            return _crop(norm_peak(x))
    elif name == "hp_rms_crop":
        def pp(x):
            return _crop(norm_rms(_hp(x), target_rms))
    else:
        raise ValueError(
            "unknown preproc '%s' (choices: hp_peak_crop, no_hp, hp_rms_crop)" % name)
    return pp


PREPROC_CHOICES = ("hp_peak_crop", "no_hp", "hp_rms_crop")
