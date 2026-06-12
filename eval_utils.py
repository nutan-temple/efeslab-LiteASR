"""
Evaluation utilities for W8A8 quantized Moonshine models.

Provides WER (Word Error Rate) evaluation on LibriSpeech test splits.
Uses AutoProcessor with max_length = max(int(seq_lens * 6.5/16000), 10)
as specified for Moonshine decoding.
"""

import numpy as np
import torch
from tqdm import tqdm


SAMPLE_RATE = 16000
TOKEN_LIMIT_FACTOR = 6.5 / SAMPLE_RATE  # ~0.000406 tokens per audio sample


@torch.no_grad()
def evaluate_wer(model, processor, dataset, device, desc="Eval"):
    """
    Evaluate Word Error Rate (WER) on a LibriSpeech dataset.

    Uses the Moonshine processor to prepare inputs and the model's generate()
    method for decoding. The max_length for generation is computed as:
        max_length = max(int(seq_lens * 6.5 / 16000), 10)

    This formula estimates the maximum number of tokens based on audio duration,
    providing a reasonable upper bound for Moonshine's output length.

    Args:
        model: MoonshineForConditionalGeneration (quantized or not)
        processor: AutoProcessor for Moonshine
        dataset: HuggingFace dataset with 'audio' and 'text' columns
        device: Device to run inference on
        desc: Description for progress bar

    Returns:
        wer: Word Error Rate as a percentage (0-100)
    """
    import evaluate as hf_evaluate

    wer_metric = hf_evaluate.load("wer")
    predictions = []
    references = []

    model = model.to(device).eval()

    for i in tqdm(range(len(dataset)), desc=f"  {desc}", leave=False):
        sample = dataset[i]
        audio = sample["audio"]["array"].astype(np.float32)
        ref_text = sample.get("text", "").strip()

        if not ref_text:
            continue

        # Prepare input
        inputs = processor(audio, return_tensors="pt", sampling_rate=SAMPLE_RATE).to(device)

        # Compute max_length based on audio duration
        seq_lens = inputs.attention_mask.sum(dim=-1)
        max_length = max(int((seq_lens * TOKEN_LIMIT_FACTOR).max().item()), 10)

        # Generate
        gen_ids = model.generate(**inputs, max_length=max_length)
        pred_text = processor.decode(gen_ids[0], skip_special_tokens=True)

        predictions.append(pred_text.strip().lower())
        references.append(ref_text.strip().lower())

    if not references:
        print(f"  WARNING: No valid references found in dataset")
        return 0.0

    wer = wer_metric.compute(references=references, predictions=predictions)
    return round(100.0 * wer, 2)


@torch.no_grad()
def evaluate_model(model, processor, device, max_eval_samples=None):
    """
    Run full WER evaluation on LibriSpeech test-clean and test-other.

    Args:
        model: MoonshineForConditionalGeneration
        processor: AutoProcessor
        device: Device for inference
        max_eval_samples: Maximum samples per split (None for full evaluation)

    Returns:
        dict with keys: wer_clean, wer_other, wer_avg
    """
    from datautils import get_librispeech_eval

    print("\n  Evaluating on LibriSpeech test-clean...")
    ds_clean = get_librispeech_eval(split="test", subset="clean", max_samples=max_eval_samples)
    wer_clean = evaluate_wer(model, processor, ds_clean, device, desc="test-clean")
    print(f"  WER test-clean: {wer_clean}%")

    print("\n  Evaluating on LibriSpeech test-other...")
    ds_other = get_librispeech_eval(split="test", subset="other", max_samples=max_eval_samples)
    wer_other = evaluate_wer(model, processor, ds_other, device, desc="test-other")
    print(f"  WER test-other: {wer_other}%")

    wer_avg = round((wer_clean + wer_other) / 2, 2)
    print(f"  WER average: {wer_avg}%")

    return {
        "wer_clean": wer_clean,
        "wer_other": wer_other,
        "wer_avg": wer_avg,
    }
