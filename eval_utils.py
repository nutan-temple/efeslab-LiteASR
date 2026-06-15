"""
Evaluation utilities for SpQR-quantized Moonshine models.

WER (Word Error Rate) evaluation on LibriSpeech test splits, replicating the
reference Moonshine recipe EXACTLY so numbers are comparable to the published
baseline (3.38% test-clean / 9.38% test-other):

  * model runs in float16 on GPU (float32 on CPU);
  * inputs are moved AND cast with `inputs.to(device, torch_dtype)`;
  * generation is token-limited: max_length = max(int(seq_lens * 6.5/16000), 10);
  * hypotheses AND references are normalized identically -- lowercased, stripped,
    and stripped of punctuation -- before WER is computed.

The last point is critical: Moonshine emits cased, punctuated text while the
LibriSpeech references are upper-cased and unpunctuated. Without identical
normalization the WER is hugely inflated (the source of the earlier ~16%).
"""

import string

import numpy as np
import torch
from tqdm import tqdm


SAMPLE_RATE = 16000
TOKEN_LIMIT_FACTOR = 6.5 / SAMPLE_RATE


def normalize_text(text):
    """Lowercase, strip, and remove punctuation (applied to ref AND hyp)."""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _load_wer_fn():
    """Return a callable wer(refs, preds) -> float, preferring `evaluate`."""
    try:
        import evaluate as hf_evaluate
        metric = hf_evaluate.load("wer")
        return lambda refs, preds: metric.compute(references=refs, predictions=preds)
    except Exception:
        import jiwer
        return lambda refs, preds: jiwer.wer(refs, preds)


@torch.no_grad()
def transcribe(model, processor, audio_array, device, torch_dtype):
    """Transcribe a single audio array, replicating the reference recipe."""
    inputs = processor(audio_array, return_tensors="pt", sampling_rate=SAMPLE_RATE)
    inputs = inputs.to(device, torch_dtype)
    seq_lens = inputs.attention_mask.sum(dim=-1)
    max_length = int((seq_lens * TOKEN_LIMIT_FACTOR).max().item())
    max_length = max(max_length, 10)
    generated_ids = model.generate(**inputs, max_length=max_length)
    return processor.decode(generated_ids[0], skip_special_tokens=True)


@torch.no_grad()
def evaluate_wer(model, processor, dataset, device, torch_dtype, desc="Eval"):
    """Compute WER (%) on a dataset with the reference normalization."""
    wer_fn = _load_wer_fn()
    predictions = []
    references = []

    for i in tqdm(range(len(dataset)), desc=f"  {desc}", leave=False):
        sample = dataset[i]
        audio = sample["audio"]["array"].astype(np.float32)
        reference = sample.get("text", "")
        if not reference or not reference.strip():
            continue
        hyp = transcribe(model, processor, audio, device, torch_dtype)
        predictions.append(normalize_text(hyp))
        references.append(normalize_text(reference))

    if not references:
        print("  WARNING: No valid references found in dataset")
        return 0.0
    return round(100.0 * wer_fn(references, predictions), 2)


@torch.no_grad()
def evaluate_model(model, processor, device, torch_dtype, max_eval_samples=None):
    """Full WER evaluation on LibriSpeech test-clean and test-other.

    The model is expected to already be on `device` in `torch_dtype`.
    Returns a dict with wer_clean, wer_other, wer_avg.
    """
    from datautils import get_librispeech_eval

    print("\n  Evaluating on LibriSpeech test-clean...")
    ds_clean = get_librispeech_eval(split="test", subset="clean", max_samples=max_eval_samples)
    wer_clean = evaluate_wer(model, processor, ds_clean, device, torch_dtype, desc="test-clean")
    print(f"  WER test-clean: {wer_clean}%   (baseline 3.38%)")

    print("\n  Evaluating on LibriSpeech test-other...")
    ds_other = get_librispeech_eval(split="test", subset="other", max_samples=max_eval_samples)
    wer_other = evaluate_wer(model, processor, ds_other, device, torch_dtype, desc="test-other")
    print(f"  WER test-other: {wer_other}%   (baseline 9.38%)")

    wer_avg = round((wer_clean + wer_other) / 2, 2)
    print(f"  WER average: {wer_avg}%")

    return {"wer_clean": wer_clean, "wer_other": wer_other, "wer_avg": wer_avg}
