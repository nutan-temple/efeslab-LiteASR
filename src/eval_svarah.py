"""
Evaluate compressed Moonshine models on the AI4Bharat Svarah dataset.

Based on the working reference script that achieves 16% WER baseline.

Usage:
    python src/eval_svarah.py --baseline --max_samples 100
    python src/eval_svarah.py --compressed_weights model1.pth model2.pth --baseline
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
from datasets import load_dataset
from transformers import MoonshineForConditionalGeneration, AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
TOKEN_LIMIT_FACTOR = 6.5 / SAMPLE_RATE


class LinearLowRank(nn.Module):
    def __init__(self, weight1, weight2, bias):
        super().__init__()
        self.weight1 = nn.Parameter(weight1)
        self.weight2 = nn.Parameter(weight2)
        self.bias = nn.Parameter(bias)

    def forward(self, x):
        return (x @ self.weight1) @ self.weight2 + self.bias


def load_compressed_model(compressed_weights_path, model_name, device, torch_dtype):
    """Load model with compressed encoder from .pth + original decoder from HuggingFace."""
    print(f"  Loading original model from: {model_name}")
    model = MoonshineForConditionalGeneration.from_pretrained(model_name)
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
    replaced = 0
    for i in range(config.encoder_num_hidden_layers):
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
    model.model.load_state_dict(encoder_sd, strict=False)
    model = model.to(device).to(torch_dtype)
    model.eval()
    return model


def load_baseline_model(model_name, device, torch_dtype):
    """Load the original uncompressed model."""
    print(f"  Loading baseline model from: {model_name}")
    model = MoonshineForConditionalGeneration.from_pretrained(model_name)
    model = model.to(device).to(torch_dtype)
    model.eval()
    return model


def transcribe_sample(model, processor, audio_array, device, torch_dtype):
    """Transcribe one audio sample. Matches reference script exactly."""
    inputs = processor(audio_array, return_tensors="pt", sampling_rate=SAMPLE_RATE)
    inputs = inputs.to(device, torch_dtype)

    # Dynamic max_length based on audio duration (prevents repetition loops)
    seq_lens = inputs.attention_mask.sum(dim=-1)
    max_length = int((seq_lens * TOKEN_LIMIT_FACTOR).max().item())
    max_length = max(max_length, 10)

    generated_ids = model.generate(**inputs, max_length=max_length)
    return processor.decode(generated_ids[0], skip_special_tokens=True)


def evaluate_model_on_svarah(model, samples, text_key, processor, device, torch_dtype, output_path=None):
    """Run inference on Svarah samples and compute WER."""
    wer_metric = evaluate.load("wer")
    all_predictions = []
    all_references = []
    all_results_per_sample = []
    total_audio_duration = 0.0
    total_inference_time = 0.0
    errors = 0

    for i in tqdm(range(len(samples)), desc="  Transcribing"):
        sample = samples[i]

        try:
            # Load audio (torchcodec decoder format)
            audio_decoder = sample["audio_filepath"]
            frames = audio_decoder.get_all_samples()
            audio_array = frames.data.squeeze(0).numpy()
            sr = frames.sample_rate

            # Resample if needed
            if sr != SAMPLE_RATE:
                import torchaudio
                audio_tensor = torch.tensor(audio_array).unsqueeze(0)
                audio_array = torchaudio.functional.resample(
                    audio_tensor, sr, SAMPLE_RATE
                ).squeeze().numpy()

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Warning [{i}]: Audio load error: {e}")
            continue

        # Get reference text
        ref_text = sample.get(text_key, "")
        if not ref_text or not str(ref_text).strip():
            errors += 1
            continue
        ref_text = str(ref_text).strip()

        # Track audio duration
        total_audio_duration += len(audio_array) / SAMPLE_RATE

        # Transcribe
        try:
            start_time = time.time()
            pred_text = transcribe_sample(model, processor, audio_array, device, torch_dtype)
            inference_time = time.time() - start_time
            total_inference_time += inference_time

            all_predictions.append(pred_text)
            all_references.append(ref_text)

            all_results_per_sample.append({
                "id": f"sample_{i}",
                "reference": ref_text,
                "prediction": pred_text,
                "duration": round(len(audio_array) / SAMPLE_RATE, 2),
                "inference_time": round(inference_time, 3),
            })

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Warning [{i}]: Inference error: {e}")
            continue

    if not all_references:
        return {"wer": None, "samples": 0, "errors": errors}

    wer = wer_metric.compute(references=all_references, predictions=all_predictions)
    wer_pct = round(100 * wer, 2)
    rtfx = round(total_audio_duration / total_inference_time, 2) if total_inference_time > 0 else None

    # Save per-sample predictions
    if output_path and all_results_per_sample:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results_per_sample, f, indent=2, ensure_ascii=False)
        print(f"  Predictions saved to: {output_path}")

    return {
        "wer": wer_pct,
        "samples": len(all_references),
        "errors": errors,
        "audio_hours": round(total_audio_duration / 3600, 3),
        "inference_time_s": round(total_inference_time, 1),
        "rtfx": rtfx,
    }


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Device: {device}, dtype: {torch_dtype}\n")

    # Load processor
    processor = AutoProcessor.from_pretrained(args.model)

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

    # Load Svarah dataset (streaming, matches reference script)
    print("Loading Svarah dataset (ai4bharat/Svarah, streaming)...")
    ds = load_dataset("ai4bharat/Svarah", split="test", streaming=True)

    samples = []
    for count, sample in enumerate(ds):
        if args.max_samples and count >= args.max_samples:
            break
        samples.append(sample)

    print(f"  Loaded {len(samples)} samples")
    if samples:
        print(f"  Keys: {list(samples[0].keys())}")
        print(f"  text[0]: '{str(samples[0].get('text', ''))[:80]}'")

    # Detect text column
    text_key = "text"
    if samples:
        for candidate in ["transcript", "transcription", "text", "sentence"]:
            if candidate in samples[0]:
                text_key = candidate
                break
    print(f"  Using text column: '{text_key}'")

    # Filter by language
    if args.language and samples:
        lang_col = None
        for col in ["primary_language", "language", "lang"]:
            if col in samples[0]:
                lang_col = col
                break
        if lang_col:
            samples = [s for s in samples if args.language.lower() in str(s.get(lang_col, "")).lower()]
            print(f"  Filtered to '{args.language}': {len(samples)} samples")

    # Evaluate each model
    all_results = {}

    for model_name, weights_path in models_to_eval:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")

        if weights_path is None:
            model = load_baseline_model(args.model, device, torch_dtype)
        else:
            model = load_compressed_model(weights_path, args.model, device, torch_dtype)

        encoder_params = sum(p.numel() for p in model.model.encoder.parameters())
        decoder_params = sum(p.numel() for p in model.model.decoder.parameters())
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Encoder: {encoder_params:,} | Decoder: {decoder_params:,} | Total: {total_params:,}")

        safe_name = model_name.replace("/", "_").replace(":", "_").replace(" ", "_")
        predictions_path = f"predictions_svarah_{safe_name}.json"

        print(f"\n  Evaluating on Svarah ({len(samples)} samples)...")
        result = evaluate_model_on_svarah(
            model, samples, text_key, processor, device, torch_dtype,
            output_path=predictions_path,
        )

        all_results[model_name] = {
            "encoder_params": encoder_params,
            "total_params": total_params,
            **result,
        }

        if result["wer"] is not None:
            print(f"\n  WER: {result['wer']}%")
            print(f"  Samples: {result['samples']} (errors: {result['errors']})")
            if result["rtfx"]:
                print(f"  RTFx: {result['rtfx']}x ({result['audio_hours']}h audio in {result['inference_time_s']}s)")
        else:
            print(f"  No valid samples evaluated.")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary table
    print(f"\n\n{'='*70}")
    print("SVARAH EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Model':<50s} {'Enc Params':<12s} {'WER':<8s} {'RTFx':<8s}")
    print(f"  {'-'*50} {'-'*12} {'-'*8} {'-'*8}")
    for name, r in all_results.items():
        wer_str = f"{r['wer']:.2f}%" if r.get('wer') else "N/A"
        rtfx_str = f"{r['rtfx']:.1f}x" if r.get('rtfx') else "N/A"
        print(f"  {name:<50s} {r['encoder_params']:>10,}  {wer_str:<8s} {rtfx_str:<8s}")
    print(f"{'='*70}")

    # Save results
    output_path = args.output or "eval_svarah_results.json"
    with open(output_path, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "results": all_results}, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Moonshine on Svarah")
    parser.add_argument("--model", type=str, default="usefulsensors/moonshine-base")
    parser.add_argument("--compressed_weights", type=str, nargs="+", default=None)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    main(args)
