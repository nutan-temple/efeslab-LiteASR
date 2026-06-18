"""GPU feature extractors (adapted from the user's Modal script).

Each is an ``nn.Module`` that maps a batch of waveforms ``[B, T]`` to a
4-D feature map ``[B, 1, n_mels, frames]`` ready for BC-ResNet, with per-utterance
mean/var normalization built in. The mel front-end is band-limited (``f_min``/
``f_max``) so the bins are concentrated on the informative 50-500 Hz IMU band.

BC-ResNet compatibility: the (unmodified) ``BCResNets`` needs exactly
``n_mels == 40`` and a single input channel. ``build_feature`` enforces this and
raises a clear error for the incompatible variants (logmel_30 / logmel_64 /
logmel_deltas), which would require changing the vendored model.
"""

import torch
import torch.nn as nn
import torchaudio

from .dataset import TARGET_SR
from .preprocessing import FMAX, FMIN, HOP, N_FFT

REQUIRED_N_MELS = 40


class LogMel(nn.Module):
    def __init__(self, sample_rate=TARGET_SR, n_fft=N_FFT, hop=HOP, n_mels=40,
                 f_min=FMIN, f_max=FMAX, deltas=False):
        super().__init__()
        self.deltas = deltas
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop,
            n_mels=n_mels, f_min=f_min, f_max=f_max, power=2.0)
        if deltas:
            self.delta_op = torchaudio.transforms.ComputeDeltas(win_length=5)
        self.out_channels = 3 if deltas else 1

    def forward(self, x):
        m = torch.log(self.mel(x) + 1e-6)
        m = (m - m.mean(dim=(-2, -1), keepdim=True)) / (m.std(dim=(-2, -1), keepdim=True) + 1e-5)
        if self.deltas:
            d1 = self.delta_op(m)
            d2 = self.delta_op(d1)
            return torch.stack([m, d1, d2], dim=1)
        return m.unsqueeze(1)


class MelLinear(nn.Module):
    """Mel power, mean-var normalized (no log)."""

    def __init__(self, sample_rate=TARGET_SR, n_fft=N_FFT, hop=HOP, n_mels=40,
                 f_min=FMIN, f_max=FMAX):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop,
            n_mels=n_mels, f_min=f_min, f_max=f_max, power=2.0)
        self.out_channels = 1

    def forward(self, x):
        m = self.mel(x)
        m = (m - m.mean(dim=(-2, -1), keepdim=True)) / (m.std(dim=(-2, -1), keepdim=True) + 1e-5)
        return m.unsqueeze(1)


class PCEN(nn.Module):
    """Per-channel energy normalization on the mel spectrogram (Wang et al., 2017)."""

    def __init__(self, sample_rate=TARGET_SR, n_fft=N_FFT, hop=HOP, n_mels=40,
                 f_min=FMIN, f_max=FMAX, alpha=0.98, delta=2.0, r=0.5, s=0.025, eps=1e-6):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop,
            n_mels=n_mels, f_min=f_min, f_max=f_max, power=2.0)
        self.alpha, self.delta, self.r, self.s, self.eps = alpha, delta, r, s, eps
        self.out_channels = 1

    def forward(self, x):
        E = self.mel(x)  # (B, n_mels, T)
        M = torch.zeros_like(E)
        m = E[..., 0]
        for t in range(E.shape[-1]):  # AR smoother over time
            m = (1 - self.s) * m + self.s * E[..., t]
            M[..., t] = m
        out = (E / (M + self.eps).pow(self.alpha) + self.delta).pow(self.r) - self.delta ** self.r
        out = (out - out.mean(dim=(-2, -1), keepdim=True)) / (out.std(dim=(-2, -1), keepdim=True) + 1e-5)
        return out.unsqueeze(1)


def _factory(name, sample_rate, n_fft, hop, f_min, f_max):
    builders = {
        "logmel_40": lambda: LogMel(sample_rate, n_fft, hop, 40, f_min, f_max, deltas=False),
        "mel_linear": lambda: MelLinear(sample_rate, n_fft, hop, 40, f_min, f_max),
        "pcen": lambda: PCEN(sample_rate, n_fft, hop, 40, f_min, f_max),
        # Incompatible with the unmodified BC-ResNet (see error below):
        "logmel_30": lambda: LogMel(sample_rate, n_fft, hop, 30, f_min, f_max),
        "logmel_64": lambda: LogMel(sample_rate, n_fft, hop, 64, f_min, f_max),
        "logmel_deltas": lambda: LogMel(sample_rate, n_fft, hop, 40, f_min, f_max, deltas=True),
    }
    if name not in builders:
        raise ValueError("unknown feature '%s' (choices: %s)" % (name, list(builders)))
    return builders[name]()


FEATURE_CHOICES = ("logmel_40", "mel_linear", "pcen")


def build_feature(name, sample_rate=TARGET_SR, n_fft=N_FFT, hop=HOP, f_min=FMIN, f_max=FMAX):
    """Build a feature extractor, enforcing BC-ResNet compatibility (40 mels, 1 ch)."""
    fe = _factory(name, sample_rate, n_fft, hop, f_min, f_max)
    n_mels = fe.mel.n_mels if hasattr(fe.mel, "n_mels") else None
    if getattr(fe, "out_channels", 1) != 1:
        raise ValueError(
            "feature '%s' produces %d channels; the unmodified BC-ResNet needs 1 input "
            "channel. Use logmel_40 / mel_linear / pcen." % (name, fe.out_channels))
    if n_mels is not None and n_mels != REQUIRED_N_MELS:
        raise ValueError(
            "feature '%s' uses n_mels=%d; the unmodified BC-ResNet requires n_mels=%d "
            "(freq must reduce 40->20->10->5->1). Use logmel_40 / mel_linear / pcen." % (
                name, n_mels, REQUIRED_N_MELS))
    return fe
