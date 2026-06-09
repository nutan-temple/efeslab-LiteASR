"""
Evaluate compressed Moonshine models on multiple datasets.

Supports:
  - LibriSpeech test-clean (full)
  - LibriSpeech test-other (full)
  - SAVARA dataset (Indian English ASR)
  - Multiple .pth files in one run for comparison

Usage:
    # Single .pth on all datasets:
    python src/eval_moonshine_full.py \
        --compressed_weights lite-moonshine-moonshine-base_0.95:0.98.pth

    # Multiple .pth files compared:
    python src/eval_moonshine_full.py \
        --compressed_weights \
            lite-moonshine-moonshine-base_0.99:0.999.pth \
            lite-moonshine-moonshine-base_0.95:0.98.pth \
            lite-moonshine-moonshine-base_0.90:0.95.pth

    # Only specific datasets:
    python src/eval_moonshine_full.py \
        --compressed_weights lite-moonshine-moonshine-base_0.95:0.98.pth \
        --datasets librispeech_clean librispeech_other

    # Include SAVARA:
    python src/eval_moonshine_full.py \
        --compressed_weights lite-moonshine-moonshine-base_0.95:0.98.pth \
        --datasets librispeech_clean librispeech_other savara

    # Evaluate original uncompressed model as baseline:
    python src/eval_moonshine_full.py --baseline

    # Limit samples (for quick testing):
    python src/eval_moonshine_full.py \
        --compressed_weights lite-moonshine-moonshine-base_0.95:0.98.pth \
        --max_samples 50
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
    """
    Load model with compressed encoder from .pth + original decoder from HuggingFace.
    """
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


def get_dataset(dataset_name, max_samples=None):
    """Load a dataset by name. Returns (dataset, text_key)."""

    if dataset_name == "librispeech_clean":
        print(f"  Loading LibriSpeech test-clean...")
        ds = load_dataset("librispeech_asr", "clean", split="test")
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        text_key = "text"

    elif dataset_name == "librispeech_other":
        print(f"  Loading LibriSpeech test-other...")
        ds = load_dataset("librispeech_asr", "other", split="test")
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        text_key = "text"

    elif dataset_name == "savara":
        print(f"  Loading SAVARA dataset...")
        # SAVARA: Indian English ASR dataset
        # Try different possible HuggingFace locations
        try:
            ds = load_dataset("savara-ai/savara-asr", split="test", trust_remote_code=True)
        except Exception:
            try:
                ds = load_dataset("SAVARA/indian-english-asr", split="test", trust_remote_code=True)
            except Exception:
                try:
                    ds = load_dataset("ai4bharat/indicvoices", "english", split="test", trust_remote_code=True)
                except Exception as e:
                    print(f"  WARNING: Could not load SAVARA dataset: {e}")
                    print(f"  Trying alternative: google/fleurs (en_in)...")
                    try:
                        ds = load_dataset("google/fleurs", "en_in", split="test", trust_remote_code=True)
                        text_key = "transcription"
                        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
                        if max_samples and max_samples < len(ds):
                            ds = ds.select(range(max_samples))
                        print(f"  Loaded {len(ds)} samples (FLEURS Indian English)")
                        return ds, text_key
                    except Exception as e2:
                        print(f"  ERROR: Could not load any Indian English dataset: {e2}")
                        return None, None

        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        # Detect text key
        sample_keys = ds.column_names
        if "text" in sample_keys:
            text_key = "text"
        elif "sentence" in sample_keys:
            text_key = "sentence"
        elif "transcription" in sample_keys:
            text_key = "transcription"
        elif "transcript" in sample_keys:
            text_key = "transcript"
        else:
            text_key = "text"

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Use: librispeech_clean, librispeech_other, savara")

    if max_samples and max_samples < len(ds):
        ds = ds.select(range(max_samples))

    print(f"  Loaded {len(ds)} samples")
    return ds, text_key


def evaluate_model(model, dataset, text_key, feature_extractor, tokenizer, normalizer, device):
    """Run inference and compute WER on a dataset."""

    wer_metric = evaluate.load("wer")
    all_predictions = []
    all_references = []
    total_audio_duration = 0.0
    total_inference_time = 0.0

    for i in tqdm(range(len(dataset)), desc="  Transcribing", leave=False):
        sample = dataset[i]
        audio = sample["audio"]["array"].astype(np.float32)
        sr = sample["audio"]["sampling_rate"]

        # Track audio duration
        total_audio_duration += len(audio) / sr

        ref_text = sample.get(text_key, "")
        if not ref_text or not ref_text.strip():
            continue

        norm_ref = normalizer(ref_text)
        if not norm_ref.strip():
            continue

        # Transcribe
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

    if not all_references:
        return {"wer": None, "samples": 0, "rtfx": None}

    wer = wer_metric.compute(references=all_references, predictions=all_predictions)
    wer_pct = round(100 * wer, 2)

    rtfx = round(total_audio_duration / total_inference_time, 2) if total_inference_time > 0 else None

    return {
        "wer": wer_pct,
        "samples": len(all_references),
        "audio_hours": round(total_audio_duration / 3600, 2),
        "inference_time_s": round(total_inference_time, 1),
        "rtfx": rtfx,
    }


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Datasets: {args.datasets}")
    print()

    # Load tokenizer and feature extractor (shared across all models)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model)
    normalizer = EnglishTextNormalizer()

    # Determine which models to evaluate
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

    # Load datasets once (reuse across models)
    datasets_loaded = {}
    for ds_name in args.datasets:
        ds, text_key = get_dataset(ds_name, args.max_samples)
        if ds is not None:
            datasets_loaded[ds_name] = (ds, text_key)

    if not datasets_loaded:
        print("ERROR: No datasets loaded successfully.")
        sys.exit(1)

    # Run evaluation for each model on each dataset
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

        model_results = {
            "encoder_params": encoder_params,
            "decoder_params": decoder_params,
            "total_params": total_params,
            "datasets": {},
        }

        for ds_name, (ds, text_key) in datasets_loaded.items():
            print(f"\n  Evaluating on: {ds_name} ({len(ds)} samples)")
            result = evaluate_model(model, ds, text_key, feature_extractor, tokenizer, normalizer, device)
            model_results["datasets"][ds_name] = result

            if result["wer"] is not None:
                rtfx_str = f", RTFx={result['rtfx']}" if result["rtfx"] else ""
                print(f"  → WER: {result['wer']}% ({result['samples']} samples{rtfx_str})")
            else:
                print(f"  → No valid samples")

        all_results[model_name] = model_results

        # Free memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Print summary table
    print(f"\n\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")

    # Header
    ds_names = list(datasets_loaded.keys())
    header = f"  {'Model':<45s} {'Enc Params':<12s}"
    for ds in ds_names:
        header += f" {ds:<18s}"
    print(header)
    print(f"  {'-'*45} {'-'*12}" + f" {'-'*18}" * len(ds_names))

    # Rows
    for model_name, results in all_results.items():
        row = f"  {model_name:<45s} {results['encoder_params']:>10,}  "
        for ds in ds_names:
            ds_result = results['datasets'].get(ds, {})
            wer = ds_result.get('wer')
            if wer is not None:
                row += f" {wer:>6.2f}%           "
            else:
                row += f" {'N/A':<18s}"
        print(row)

    print(f"{'='*80}")

    # Save results to JSON
    output_path = args.output or "eval_results.json"
    results_json = {
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "base_model": args.model,
        "results": all_results,
    }
    with open(output_path, "w") as f:
        json.dump(results_json, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate compressed Moonshine models on multiple datasets"
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
        help="Also evaluate the original uncompressed model for comparison",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["librispeech_clean", "librispeech_other"],
        help="Datasets to evaluate on. Options: librispeech_clean, librispeech_other, savara (default: both librispeech)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples per dataset (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save JSON results (default: eval_results.json)",
    )

    args = parser.parse_args()
    main(args)
