"""
Evaluate compressed Moonshine models on the AI4Bharat Svarah dataset.

Svarah is a gated dataset - you need to:
1. Go to https://huggingface.co/datasets/ai4bharat/Svarah
2. Accept the license agreement
3. Login: huggingface-cli login (paste your HF token)

Dataset structure (ai4bharat/Svarah):
  - audio: Audio column (wav/flac)
  - transcript: Ground truth transcription
  - language: Language label
  - speaker_id: Speaker identifier
  - duration: Audio duration in seconds

Usage:
    # Evaluate compressed model on Svarah:
    python src/eval_svarah.py \
        --compressed_weights lite-moonshine-moonshine-base_0.95:0.98.pth

    # Evaluate baseline (uncompressed):
    python src/eval_svarah.py --baseline

    # Compare multiple models:
    python src/eval_svarah.py \
        --compressed_weights \
            lite-moonshine-moonshine-base_0.99:0.999.pth \
            lite-moonshine-moonshine-base_0.95:0.98.pth \
        --baseline

    # Quick test (100 samples):
    python src/eval_svarah.py \
        --compressed_weights lite-moonshine-moonshine-base_0.95:0.98.pth \
        --max_samples 100

    # Filter by language (if Svarah has multiple):
    python src/eval_svarah.py \
        --compressed_weights lite-moonshine-moonshine-base_0.95:0.98.pth \
        --language english
"""

import argparse
import os
import sys
import json
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import evaluate
from tqdm import tqdm
from datasets import load_dataset, Audio
from transformers import (
    AutoTokenizer,
    AutoFeatureExtractor,
    MoonshineForConditionalGeneration,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalizer import EnglishTextNormalizer


class LinearLowRank(nn.Module):
    """Low-rank layer: forward(x) = (x @ weight1) @ weight2 + bias"""

    def __init__(self, weight1, weight2, bias):
        super().__init__()
        self.weight1 = nn.Parameter(weight1)
        self.weight2 = nn.Parameter(weight2)
        self.bias = nn.Parameter(bias)

    def forward(self, x):
        return (x @ self.weight1) @ self.weight2 + self.bias


def load_compressed_model(compressed_weights_path, model_name, device):
    """Load model with compressed encoder from .pth + original decoder from HuggingFace."""
    print(f"  Loading original model from: {model_name}")
    model = MoonshineForConditionalGeneration.from_pretrained(
        model_name, trust_remote_code=True
    )
    config = model.config

    print(f"  Loading compressed weights from: {compressed_weights_path}")
    state_dict = torch.load(compressed_weights_path, map_location="cpu", weights_only=False)

    # Extract encoder keys
    encoder_sd = {}
    for key, tensor in state_dict.items():
        if key.startswith("model.encoder."):
            encoder_sd[key[len("model."):]] = tensor

    if not encoder_sd:
        raise ValueError("No encoder keys found in .pth")

    # Replace compressed layers with LinearLowRank
    component_paths = [
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.fc1", "mlp.fc2",
    ]

    num_layers = config.encoder_num_hidden_layers
    replaced = 0

    for i in range(num_layers):
        layer = model.model.encoder.layers[i]
        for attr_path in component_paths:
            w1_key = f"encoder.layers.{i}.{attr_path}.weight1"
            w2_key = f"encoder.layers.{i}.{attr_path}.weight2"
            bias_key = f"encoder.layers.{i}.{attr_path}.bias"

            if w1_key in encoder_sd and w2_key in encoder_sd:
                w1 = encoder_sd.pop(w1_key)
                w2 = encoder_sd.pop(w2_key)
                bias = encoder_sd.pop(bias_key)

                low_rank_layer = LinearLowRank(w1, w2, bias)

                parts = attr_path.split(".")
                obj = layer
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], low_rank_layer)
                replaced += 1

                encoder_sd.pop(f"encoder.layers.{i}.{attr_path}.weight", None)

    print(f"  Replaced {replaced} layers with LinearLowRank")

    # Load remaining encoder weights
    model.model.load_state_dict(encoder_sd, strict=False)
    model = model.to(device)
    model.eval()
    return model


def load_baseline_model(model_name, device):
    """Load the original uncompressed model."""
    print(f"  Loading baseline model from: {model_name}")
    model = MoonshineForConditionalGeneration.from_pretrained(
        model_name, trust_remote_code=True
    )
    model = model.to(device)
    model.eval()
    return model


def detect_text_column(dataset):
    """Detect which column holds the transcription text."""
    cols = dataset.column_names
    # Priority order for text column detection
    candidates = ["transcript", "transcription", "text", "sentence", "normalized_text"]
    for candidate in candidates:
        if candidate in cols:
            return candidate
    raise ValueError(
        f"Could not detect text column. Available columns: {cols}\n"
        f"Expected one of: {candidates}"
    )


def load_svarah(args):
    """Load the Svarah dataset with proper authentication."""
    print("\nLoading Svarah dataset (ai4bharat/Svarah)...")
    print("  Note: This is a gated dataset. Make sure you have:")
    print("  1. Accepted the license at https://huggingface.co/datasets/ai4bharat/Svarah")
    print("  2. Run: huggingface-cli login")
    print()

    try:
        ds = load_dataset("ai4bharat/Svarah", split="test")
    except Exception as e:
        if "gated" in str(e).lower() or "authentication" in str(e).lower() or "401" in str(e):
            print("ERROR: Authentication required for Svarah dataset.")
            print("Run: huggingface-cli login")
            print("Then accept the license at: https://huggingface.co/datasets/ai4bharat/Svarah")
            sys.exit(1)
        else:
            raise

    print(f"  Dataset loaded: {len(ds)} samples")
    print(f"  Columns: {ds.column_names}")

    # Cast audio to 16kHz
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    # Detect text column
    text_key = detect_text_column(ds)
    print(f"  Using text column: '{text_key}'")

    # Filter by language if specified
    if args.language:
        lang_col = None
        for col in ["language", "lang", "locale"]:
            if col in ds.column_names:
                lang_col = col
                break

        if lang_col:
            # Check available languages
            unique_langs = set(ds[lang_col])
            print(f"  Available languages: {sorted(unique_langs)}")

            # Filter
            ds = ds.filter(lambda x: args.language.lower() in x[lang_col].lower())
            print(f"  Filtered to '{args.language}': {len(ds)} samples")
        else:
            print(f"  Warning: No language column found, using all samples")

    # Limit samples
    if args.max_samples and args.max_samples < len(ds):
        ds = ds.select(range(args.max_samples))
        print(f"  Limited to {args.max_samples} samples")

    return ds, text_key


def evaluate_model_on_svarah(model, dataset, text_key, feature_extractor, tokenizer, normalizer, device):
    """Run inference on Svarah and compute WER."""

    wer_metric = evaluate.load("wer")
    all_predictions = []
    all_references = []
    total_audio_duration = 0.0
    total_inference_time = 0.0
    errors = 0

    for i in tqdm(range(len(dataset)), desc="  Transcribing"):
        sample = dataset[i]

        try:
            audio = sample["audio"]["array"].astype(np.float32)
            sr = sample["audio"]["sampling_rate"]
        except (KeyError, TypeError) as e:
            errors += 1
            continue

        # Track audio duration
        total_audio_duration += len(audio) / sr

        # Get reference text
        ref_text = sample.get(text_key, "")
        if not ref_text or not str(ref_text).strip():
            errors += 1
            continue

        ref_text = str(ref_text)
        norm_ref = normalizer(ref_text)
        if not norm_ref.strip():
            errors += 1
            continue

        # Transcribe
        try:
            inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
            input_values = inputs["input_values"].to(device)

            start_time = time.time()
            with torch.no_grad():
                generated_ids = model.generate(input_values=input_values)
            inference_time = time.time() - start_time
            total_inference_time += inference_time

            pred_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            norm_pred = normalizer(pred_text)

            all_predictions.append(norm_pred)
            all_references.append(norm_ref)

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Warning: Error on sample {i}: {e}")
            continue

    if not all_references:
        return {"wer": None, "samples": 0, "errors": errors}

    wer = wer_metric.compute(references=all_references, predictions=all_predictions)
    wer_pct = round(100 * wer, 2)
    rtfx = round(total_audio_duration / total_inference_time, 2) if total_inference_time > 0 else None

    return {
        "wer": wer_pct,
        "samples": len(all_references),
        "errors": errors,
        "audio_hours": round(total_audio_duration / 3600, 3),
        "inference_time_s": round(total_inference_time, 1),
        "rtfx": rtfx,
    }


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load tokenizer and feature extractor
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model)
    normalizer = EnglishTextNormalizer()

    # Determine models to evaluate
    models_to_eval = []
    if args.baseline:
        models_to_eval.append(("baseline (uncompressed)", None))
    if args.compressed_weights:
        for path in args.compressed_weights:
            if not os.path.isfile(path):
                print(f"WARNING: File not found, skipping: {path}")
                continue
            models_to_eval.append((os.path.basename(path), path))

    if not models_to_eval:
        print("ERROR: No models to evaluate. Use --compressed_weights or --baseline")
        sys.exit(1)

    # Load Svarah dataset once
    dataset, text_key = load_svarah(args)

    # Evaluate each model
    all_results = {}

    for model_name, weights_path in models_to_eval:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")

        if weights_path is None:
            model = load_baseline_model(args.model, device)
        else:
            model = load_compressed_model(weights_path, args.model, device)

        # Print param counts
        encoder_params = sum(p.numel() for p in model.model.encoder.parameters())
        decoder_params = sum(p.numel() for p in model.model.decoder.parameters())
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Encoder: {encoder_params:,} | Decoder: {decoder_params:,} | Total: {total_params:,}")

        # Evaluate
        print(f"\n  Evaluating on Svarah ({len(dataset)} samples)...")
        result = evaluate_model_on_svarah(
            model, dataset, text_key, feature_extractor, tokenizer, normalizer, device
        )

        all_results[model_name] = {
            "encoder_params": encoder_params,
            "decoder_params": decoder_params,
            "total_params": total_params,
            **result,
        }

        if result["wer"] is not None:
            print(f"\n  WER: {result['wer']}%")
            print(f"  Samples: {result['samples']} (errors/skipped: {result['errors']})")
            if result["rtfx"]:
                print(f"  RTFx: {result['rtfx']} ({result['audio_hours']} hours audio in {result['inference_time_s']}s)")
        else:
            print(f"  No valid samples evaluated.")

        # Free memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Summary
    print(f"\n\n{'='*70}")
    print("SVARAH EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Model':<50s} {'Enc Params':<12s} {'WER':<8s} {'RTFx':<8s}")
    print(f"  {'-'*50} {'-'*12} {'-'*8} {'-'*8}")

    for model_name, result in all_results.items():
        wer_str = f"{result['wer']:.2f}%" if result.get('wer') is not None else "N/A"
        rtfx_str = f"{result['rtfx']:.1f}x" if result.get('rtfx') is not None else "N/A"
        print(f"  {model_name:<50s} {result['encoder_params']:>10,}  {wer_str:<8s} {rtfx_str:<8s}")

    print(f"{'='*70}")

    # Save results
    output_path = args.output or "eval_svarah_results.json"
    results_json = {
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "base_model": args.model,
        "dataset": "ai4bharat/Svarah",
        "language_filter": args.language,
        "max_samples": args.max_samples,
        "results": all_results,
    }
    with open(output_path, "w") as f:
        json.dump(results_json, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate compressed Moonshine models on AI4Bharat Svarah dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="usefulsensors/moonshine-base",
        help="Base model on HuggingFace (default: usefulsensors/moonshine-base)",
    )
    parser.add_argument(
        "--compressed_weights",
        type=str,
        nargs="+",
        default=None,
        help="Path(s) to .pth file(s). Can specify multiple for comparison.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Evaluate the original uncompressed model",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Filter Svarah by language (e.g., 'english', 'hindi'). Default: use all.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save JSON results (default: eval_svarah_results.json)",
    )

    args = parser.parse_args()
    main(args)
