"""Dataset / data-indexing utilities for 3333 Hz IMU (accel-Z) keyword wavs.

We deliberately do NOT reuse the vendored ``Padding`` / ``SpeechCommand`` classes
because they are hardcoded to Google Speech Commands (16 kHz, directory-based
labels). Everything here is new code; the vendored BC-ResNet files stay untouched.
"""

import csv
import os
import random

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from .labels import CLASS_TO_IDX, filename_to_label

FILTERED_SUFFIX = "_50_500hz.wav"

# Single source of truth: this pipeline trains on 3.3 kHz audio ONLY.
# Any file that is not at this rate is resampled to it on load (or skipped in
# strict mode). The mel front-end in engine.py is tuned for exactly this rate.
TARGET_SR = 3333


def list_wav_files(wav_dir, use_filtered=True, manifest=None):
    """Collect the wav paths we want to train on.

    If ``manifest`` (a CSV with columns ``source_csv,wav_path,filtered_path``) is
    given and exists, we trust it as the source of truth. Otherwise we walk
    ``wav_dir`` and pick either the band-pass-filtered wavs (``*_50_500hz.wav``)
    or the raw wavs, never both.
    """
    files = []
    if manifest and os.path.isfile(manifest):
        col = "filtered_path" if use_filtered else "wav_path"
        with open(manifest, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel = (row.get(col) or row.get("wav_path") or "").strip()
                if rel:
                    files.append(os.path.join(wav_dir, rel))
    else:
        for root, _, fnames in os.walk(wav_dir):
            for fn in fnames:
                low = fn.lower()
                if not low.endswith(".wav"):
                    continue
                is_filtered = low.endswith(FILTERED_SUFFIX)
                if use_filtered and not is_filtered:
                    continue
                if not use_filtered and is_filtered:
                    continue
                files.append(os.path.join(root, fn))
    return sorted(set(files))


def read_manifest(manifest, wav_dir, use_filtered=True):
    """Return a list of ``(abs_path, label_str_or_None)`` from a manifest CSV.

    Uses the ``label`` column when present (the user-provided / generated labels);
    otherwise the label is left as ``None`` and the caller falls back to parsing the
    file name.
    """
    entries = []
    with open(manifest, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        path_col = "filtered_path" if (use_filtered and "filtered_path" in cols) else "wav_path"
        has_label = "label" in cols
        for row in reader:
            rel = (row.get(path_col) or row.get("wav_path") or "").strip()
            if not rel:
                continue
            lab = (row.get("label") or "").strip() if has_label else ""
            entries.append((os.path.join(wav_dir, rel), lab or None))
    return entries


def build_index(wav_dir, use_filtered=True, manifest=None, target_sr=TARGET_SR, strict_sr=False):
    """Return ``(paths, labels, skipped)`` keeping only readable, labeled files.

    Label source priority: the manifest ``label`` column (if it names a known class)
    -> otherwise parsed from the file name. Gracefully drops the stray malformed file
    and anything that fails to parse to a known class or fails to open. When
    ``strict_sr`` is True, files not at ``target_sr`` (3.3 kHz) are skipped too.
    """
    if manifest and os.path.isfile(manifest):
        entries = read_manifest(manifest, wav_dir, use_filtered)
    else:
        entries = [(p, None) for p in list_wav_files(wav_dir, use_filtered, None)]

    paths, labels, skipped = [], [], []
    for p, lab_str in entries:
        if lab_str and lab_str in CLASS_TO_IDX:
            lab = CLASS_TO_IDX[lab_str]
        else:
            lab = filename_to_label(p)  # fallback when label missing / "unknown"
        if lab is None:
            skipped.append((p, "no-label"))
            continue
        if not os.path.isfile(p):
            skipped.append((p, "missing"))
            continue
        try:
            info = torchaudio.info(p)
            if info.num_frames <= 0:
                skipped.append((p, "empty"))
                continue
        except Exception as exc:  # malformed / unreadable wav
            skipped.append((p, "unreadable: %s" % exc))
            continue
        if strict_sr and info.sample_rate != target_sr:
            skipped.append((p, "sample_rate=%d!=%d" % (info.sample_rate, target_sr)))
            continue
        paths.append(p)
        labels.append(lab)
    return paths, labels, skipped


def compute_fixed_len(paths, percentile=99.0, target_sr=TARGET_SR):
    """Pick a single fixed input length (in samples at ``target_sr``) from the data.

    Using a high percentile (default 99th) avoids guessing a duration and only
    crops a handful of unusually long outliers, while everything shorter is
    zero-padded. Lengths are scaled to ``target_sr`` so the value is correct even
    if a clip was stored at a different native rate (and thus gets resampled).
    """
    lengths = []
    for p in paths:
        try:
            info = torchaudio.info(p)
            n = info.num_frames
            if info.sample_rate != target_sr:
                n = int(round(n * target_sr / float(info.sample_rate)))
            lengths.append(n)
        except Exception:
            pass
    if not lengths:
        return 1
    return int(np.percentile(np.asarray(lengths, dtype=np.float64), percentile))


def load_wav_mono(path, target_sr=TARGET_SR, normalize=True):
    """Load a wav as a mono ``[1, L]`` float tensor at ``target_sr`` (3.3 kHz).

    Resamples to ``target_sr`` when the file was stored at a different rate so the
    model only ever sees 3.3 kHz audio.
    """
    wav, sr = torchaudio.load(path)  # [C, L]
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
    if normalize:
        peak = wav.abs().max()
        if peak > 0:
            wav = wav / (peak + 1e-8)
    return wav, target_sr


def load_wav_1d_np(path, target_sr=TARGET_SR):
    """Load a wav as a mono 1-D float32 numpy array at ``target_sr`` (3.3 kHz)."""
    wav, sr = torchaudio.load(path)  # [C, L]
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
    return wav.squeeze(0).numpy().astype(np.float32)


def fix_length(wav, target_len, train=False):
    """Pad (random/center) or crop (random/center) a ``[1, L]`` wav to ``target_len``."""
    length = wav.shape[-1]
    if length == target_len:
        return wav
    if length < target_len:
        pad = target_len - length
        left = random.randint(0, pad) if train else pad // 2
        return torch.nn.functional.pad(wav, (left, pad - left))
    # length > target_len -> crop
    start = random.randint(0, length - target_len) if train else (length - target_len) // 2
    return wav[..., start:start + target_len]


class IMUKeywordDataset(Dataset):
    """Returns ``([1, target_len] waveform, int label)``.

    Light, sample-rate-agnostic waveform augmentation (gain / circular time-shift /
    additive gaussian noise) is applied only for the training split. The mel /
    SpecAugment stage is handled later by the vendored ``Preprocess``.
    """

    def __init__(
        self,
        paths,
        labels,
        target_len,
        train=False,
        augment=False,
        normalize=True,
        target_sr=TARGET_SR,
        noise_std=0.005,
        gain_db=3.0,
        shift_frac=0.1,
        preproc_fn=None,
    ):
        self.paths = list(paths)
        self.labels = list(labels)
        self.target_len = int(target_len)
        self.train = train
        self.augment = bool(augment and train)
        self.normalize = normalize
        self.target_sr = int(target_sr)
        self.noise_std = noise_std
        self.gain_db = gain_db
        self.shift_frac = shift_frac
        # When set, a numpy ``x -> x`` pipeline (HP filter / norm / crop_max_energy)
        # produces a fixed-length 1-D clip and __getitem__ returns a [T] tensor.
        self.preproc_fn = preproc_fn

    def _augment_np(self, x):
        if self.gain_db and self.gain_db > 0:
            x = x * (10 ** (random.uniform(-self.gain_db, self.gain_db) / 20.0))
        if self.shift_frac and self.shift_frac > 0:
            shift = int(random.uniform(-self.shift_frac, self.shift_frac) * len(x))
            if shift != 0:
                x = np.roll(x, shift)
        if self.noise_std and self.noise_std > 0:
            x = x + np.random.randn(len(x)).astype(np.float32) * self.noise_std
        return x.astype(np.float32)

    def __len__(self):
        return len(self.paths)

    def _augment_waveform(self, wav):
        if self.gain_db and self.gain_db > 0:
            gain = 10 ** (random.uniform(-self.gain_db, self.gain_db) / 20.0)
            wav = wav * gain
        if self.shift_frac and self.shift_frac > 0:
            n = wav.shape[-1]
            shift = int(random.uniform(-self.shift_frac, self.shift_frac) * n)
            if shift != 0:
                wav = torch.roll(wav, shifts=shift, dims=-1)
        if self.noise_std and self.noise_std > 0:
            wav = wav + torch.randn_like(wav) * self.noise_std
        return wav

    def __getitem__(self, idx):
        if self.preproc_fn is not None:
            x = load_wav_1d_np(self.paths[idx], target_sr=self.target_sr)
            x = self.preproc_fn(x)  # fixed-length 1-D window
            if self.augment:
                x = self._augment_np(x)
            return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), self.labels[idx]
        wav, _sr = load_wav_mono(self.paths[idx], target_sr=self.target_sr, normalize=self.normalize)
        wav = fix_length(wav, self.target_len, train=self.train)
        if self.augment:
            wav = self._augment_waveform(wav)
        return wav, self.labels[idx]
