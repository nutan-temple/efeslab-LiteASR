"""
Calibration and evaluation data utilities for W8A8 Moonshine quantization.

Provides:
  - LibriSpeech dev-clean loading for GPTQ calibration
  - LibriSpeech test-clean/test-other loading for evaluation
  - Synthetic audio generation for testing
"""

import random

import numpy as np
import torch
from datasets import load_dataset, Audio


SAMPLE_RATE = 16000


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_librispeech_calibration(nsamples=128, audio_len=160000, seed=42):
    """
    Load real audio from LibriSpeech dev-clean for calibration.

    Uses the validation split of LibriSpeech clean as calibration data.
    Audio is padded/truncated to a fixed length for uniform batching
    during Hessian accumulation.

    Args:
        nsamples: Number of calibration samples to load
        audio_len: Fixed length (in samples at 16kHz) to pad/truncate to.
                   Default 160000 = 10 seconds.
        seed: Random seed for shuffling

    Returns:
        list of torch.Tensor: Each tensor has shape (1, audio_len)
    """
    print(f"Loading calibration data ({nsamples} samples from LibriSpeech dev-clean)...")
    ds = load_dataset("openslr/librispeech_asr", "clean", split="validation")
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    ds = ds.shuffle(seed=seed)

    audio_samples = []
    for i in range(min(nsamples, len(ds))):
        audio = ds[i]["audio"]["array"].astype(np.float32)
        # Pad or truncate to fixed length
        if len(audio) < audio_len:
            audio = np.pad(audio, (0, audio_len - len(audio)))
        else:
            audio = audio[:audio_len]
        audio_samples.append(torch.tensor(audio).unsqueeze(0))  # (1, audio_len)

    print(f"  Loaded {len(audio_samples)} calibration samples ({audio_len/SAMPLE_RATE:.1f}s each)")
    return audio_samples


def get_librispeech_eval(split="test", subset="clean", max_samples=None):
    """
    Load LibriSpeech evaluation data.

    Args:
        split: Dataset split (default "test")
        subset: "clean" or "other"
        max_samples: Maximum number of samples to load (None for all)

    Returns:
        dataset: HuggingFace dataset with 'audio' and 'text' columns
    """
    print(f"Loading LibriSpeech {subset} ({split})...")
    ds = load_dataset("openslr/librispeech_asr", subset, split=split)
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

    if max_samples and max_samples < len(ds):
        ds = ds.select(range(max_samples))

    print(f"  Loaded {len(ds)} eval samples")
    return ds


def get_synthetic_calibration(nsamples=128, audio_len=160000, seed=42):
    """
    Generate synthetic (random noise) audio for testing/debugging.

    This is useful for verifying the quantization pipeline works
    without requiring network access to download LibriSpeech.

    Args:
        nsamples: Number of synthetic samples
        audio_len: Length of each sample in audio frames (at 16kHz)
        seed: Random seed

    Returns:
        list of torch.Tensor: Each tensor has shape (1, audio_len)
    """
    set_seed(seed)
    print(f"Generating {nsamples} synthetic audio samples ({audio_len/SAMPLE_RATE:.1f}s each)...")
    data = []
    for _ in range(nsamples):
        audio = torch.randn(1, audio_len) * 0.1  # low amplitude noise
        data.append(audio)
    return data
