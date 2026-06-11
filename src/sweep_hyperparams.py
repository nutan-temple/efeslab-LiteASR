"""
Hyperparameter sweep for LiteASR compression on Moonshine Base.

Sweeps attention and MLP variance thresholds, measures encoder parameter
reduction and WER on LibriSpeech test-clean and test-other.

Usage:
    python src/sweep_hyperparams.py --max_eval_samples 100
    python src/sweep_hyperparams.py --max_eval_samples 500
    python src/sweep_hyperparams.py  # full evaluation (slow)

Results are saved to sweep_results.json and printed as a formatted table.
"""

import argparse
import os
import sys
import json
import time
from datetime import datetime
from itertools import product

import numpy as np
import torch
import torch.nn as nn
import evaluate
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets, Audio
from transformers import MoonshineForConditionalGeneration, AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalizer import data_utils


# ── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
TOKEN_LIMIT_FACTOR = 6.5 / SAMPLE_RATE
ORIG_ENCODER_PARAMS = 20_153_120  # original moonshine-base encoder


class LinearLowRank(nn.Module):
    def __init__(self, weight1, weight2, bias):
        super().__init__()
        self.weight1 = nn.Parameter(weight1)
        self.weight2 = nn.Parameter(weight2)
        self.bias = nn.Parameter(bias)

    def forward(self, x):
        return (x @ self.weight1) @ self.weight2 + self.bias


def apply_low_rank(model, calibration_data, rank_threshold, processor, device):
    """
    Apply low-rank compression to encoder layers.
    Returns the compressed model and the encoder param count.

    rank_threshold: "attn_thresh:mlp_thresh" e.g. "0.98:0.995"
    """
    attn_thresh, mlp_thresh = [float(x) for x in rank_threshold.split(":")]
    base_dim = model.config.hidden_size  # 416
    num_layers = model.config.encoder_num_hidden_layers  # 8

    # ── Collect activations via hooks ──────────────────────────────────────
    calibration_activations = {
        i: {"q_proj": [], "k_proj": [], "v_proj": [], "o_proj": [], "fc1": [], "fc2": []}
        for i in range(num_layers)
    }

    hooks = []
    encoder_layers = model.model.encoder.layers

    for i_layer in range(num_layers):
        layer = encoder_layers[i_layer]

        def make_hook(layer_idx, comp_name):
            def hook_fn(module, inp, out):
                calibration_activations[layer_idx][comp_name].append(out.detach().cpu())
            return hook_fn

        hooks.append(layer.self_attn.q_proj.register_forward_hook(make_hook(i_layer, "q_proj")))
        hooks.append(layer.self_attn.k_proj.register_forward_hook(make_hook(i_layer, "k_proj")))
        hooks.append(layer.self_attn.v_proj.register_forward_hook(make_hook(i_layer, "v_proj")))
        hooks.append(layer.self_attn.o_proj.register_forward_hook(make_hook(i_layer, "o_proj")))
        hooks.append(layer.mlp.fc1.register_forward_hook(make_hook(i_layer, "fc1")))
        hooks.append(layer.mlp.fc2.register_forward_hook(make_hook(i_layer, "fc2")))

    # Run calibration
    model.eval()
    with torch.no_grad():
        for sample in calibration_data:
            audio = sample["audio"]["array"].astype(np.float32)
            inputs = processor(audio, return_tensors="pt", sampling_rate=SAMPLE_RATE)
            inputs = inputs.to(device)
            model.generate(**inputs, max_length=10)  # short generation just to run encoder

    for h in hooks:
        h.remove()

    # ── Apply SVD and replace layers ──────────────────────────────────────
    component_names = ["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"]
    replaced = 0

    for i_layer in range(num_layers):
        layer = encoder_layers[i_layer]

        for i, comp_name in enumerate(component_names):
            # Get the linear module
            if comp_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                parts = f"self_attn.{comp_name}".split(".")
            else:
                parts = f"mlp.{comp_name}".split(".")

            obj = layer
            for part in parts[:-1]:
                obj = getattr(obj, part)
            linear_module = getattr(obj, parts[-1])

            # Skip if already a LinearLowRank (shouldn't happen in fresh model)
            if isinstance(linear_module, LinearLowRank):
                continue

            # Flatten activations to 2D
            flattened = []
            for feat in calibration_activations[i_layer][comp_name]:
                flattened.append(feat.reshape(-1, feat.shape[-1]))
            features = torch.cat(flattened, dim=0).float()

            # Determine threshold
            thresh = attn_thresh if i <= 3 else mlp_thresh

            # Center + SVD
            Y_mean = features.mean(dim=0)
            features_centered = features - Y_mean
            U, S, Vt = torch.linalg.svd(features_centered, full_matrices=False)

            # Find rank k (multiple of 16)
            S_F = S ** 2
            k = -1
            for j in range(16, len(S_F) + 1, 16):
                if S_F[:j].sum() / S_F.sum() > thresh:
                    k = j
                    break

            # Skip if rank too high (no benefit)
            if i <= 3:
                if k > 0.5 * base_dim or k == -1:
                    continue
            else:
                if k > 0.8 * features.shape[-1] or k == -1:
                    continue

            V = Vt.T
            V_k = V[:, :k]

            W = linear_module.weight.T.float()
            w1 = W @ V_k
            w2 = V_k.T

            if linear_module.bias is None:
                bias = Y_mean - Y_mean @ V_k @ V_k.T
            else:
                original_bias = linear_module.bias.float()
                bias = Y_mean + (original_bias - Y_mean) @ V_k @ V_k.T

            new_layer = LinearLowRank(
                w1.to(linear_module.weight.dtype),
                w2.to(linear_module.weight.dtype),
                bias.to(linear_module.weight.dtype),
            )
            setattr(obj, parts[-1], new_layer)
            replaced += 1

    encoder_params = sum(p.numel() for p in model.model.encoder.parameters())
    return model, encoder_params, replaced


def evaluate_wer(model, processor, dataset, device, max_samples=None):
    """Evaluate WER on a dataset."""
    wer_metric = evaluate.load("wer")
    all_preds = []
    all_refs = []

    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))

    for i in tqdm(range(len(dataset)), desc="    Eval", leave=False):
        sample = dataset[i]
        audio = sample["audio"]["array"].astype(np.float32)
        ref_text = sample.get("text", "").strip()
        if not ref_text:
            continue

        inputs = processor(audio, return_tensors="pt", sampling_rate=SAMPLE_RATE)
        inputs = inputs.to(device)

        seq_lens = inputs.attention_mask.sum(dim=-1)
        max_length = max(int((seq_lens * TOKEN_LIMIT_FACTOR).max().item()), 10)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_length=max_length)

        pred_text = processor.decode(generated_ids[0], skip_special_tokens=True)

        all_preds.append(pred_text.strip().lower())
        all_refs.append(ref_text.strip().lower())

    if not all_refs:
        return None

    wer = wer_metric.compute(references=all_refs, predictions=all_preds)
    return round(100 * wer, 2)


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float32  # keep float32 for compression accuracy
    print(f"Device: {device}")
    print(f"Calibration samples: {args.num_calibration_samples}")
    print(f"Eval samples per dataset: {args.max_eval_samples or 'all'}\n")

    # Load processor
    processor = AutoProcessor.from_pretrained(args.model)

    # ── Load calibration data (LibriSpeech validation) ────────────────────
    print("Loading calibration data (LibriSpeech dev-clean + dev-other)...")
    cal_clean = load_dataset("librispeech_asr", "clean", split="validation")
    cal_other = load_dataset("librispeech_asr", "other", split="validation")
    cal_clean = cal_clean.cast_column("audio", Audio(sampling_rate=16000))
    cal_other = cal_other.cast_column("audio", Audio(sampling_rate=16000))

    n_per_source = args.num_calibration_samples // 2
    cal_data = concatenate_datasets([
        cal_clean.shuffle(seed=42).select(range(min(n_per_source, len(cal_clean)))),
        cal_other.shuffle(seed=42).select(range(min(n_per_source, len(cal_other)))),
    ]).shuffle(seed=42)
    print(f"  Calibration: {len(cal_data)} samples\n")

    # ── Load evaluation data ──────────────────────────────────────────────
    print("Loading evaluation data...")
    eval_clean = load_dataset("librispeech_asr", "clean", split="test")
    eval_other = load_dataset("librispeech_asr", "other", split="test")
    eval_clean = eval_clean.cast_column("audio", Audio(sampling_rate=16000))
    eval_other = eval_other.cast_column("audio", Audio(sampling_rate=16000))
    print(f"  test-clean: {len(eval_clean)} samples")
    print(f"  test-other: {len(eval_other)} samples\n")

    # ── Define sweep grid ─────────────────────────────────────────────────
    # Phase 1: Fix attention at 0.99, sweep MLP
    # Phase 2: Fix attention at 0.98, sweep MLP
    # Phase 3: Fix attention at 0.97, sweep MLP
    attn_thresholds = [0.99, 0.98, 0.97]
    mlp_thresholds = [0.999, 0.995, 0.99, 0.985, 0.98, 0.975, 0.97]

    sweep_configs = []
    for attn_t in attn_thresholds:
        for mlp_t in mlp_thresholds:
            sweep_configs.append((attn_t, mlp_t))

    print(f"Total configurations to sweep: {len(sweep_configs)}")
    print(f"{'='*80}\n")

    # ── Run sweep ─────────────────────────────────────────────────────────
    results = []

    for idx, (attn_t, mlp_t) in enumerate(sweep_configs):
        threshold_str = f"{attn_t}:{mlp_t}"
        print(f"\n[{idx+1}/{len(sweep_configs)}] Threshold: attn={attn_t}, mlp={mlp_t}")
        print(f"  {'─'*50}")

        # Load fresh model each time
        model = MoonshineForConditionalGeneration.from_pretrained(args.model)
        model = model.to(device)
        model.eval()

        # Compress
        start_time = time.time()
        model, enc_params, n_replaced = apply_low_rank(
            model, cal_data, threshold_str, processor, device
        )
        compress_time = time.time() - start_time

        enc_reduction = round(100 * (1 - enc_params / ORIG_ENCODER_PARAMS), 1)
        print(f"  Encoder params: {enc_params:,} (reduction: {enc_reduction}%)")
        print(f"  Layers replaced: {n_replaced}")
        print(f"  Compression time: {compress_time:.1f}s")

        # Evaluate on test-clean
        print(f"  Evaluating test-clean...")
        wer_clean = evaluate_wer(model, processor, eval_clean, device, args.max_eval_samples)

        # Evaluate on test-other
        print(f"  Evaluating test-other...")
        wer_other = evaluate_wer(model, processor, eval_other, device, args.max_eval_samples)

        print(f"  WER test-clean: {wer_clean}%")
        print(f"  WER test-other: {wer_other}%")

        result = {
            "attn_threshold": attn_t,
            "mlp_threshold": mlp_t,
            "threshold_str": threshold_str,
            "encoder_params": enc_params,
            "encoder_reduction_pct": enc_reduction,
            "layers_replaced": n_replaced,
            "wer_clean": wer_clean,
            "wer_other": wer_other,
            "wer_avg": round((wer_clean + wer_other) / 2, 2) if wer_clean and wer_other else None,
            "compress_time_s": round(compress_time, 1),
        }
        results.append(result)

        # Free memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Print results table ───────────────────────────────────────────────
    print(f"\n\n{'='*100}")
    print("HYPERPARAMETER SWEEP RESULTS")
    print(f"{'='*100}")
    print(f"  {'Attn':<6s} {'MLP':<6s} {'Enc Params':<12s} {'Reduction':<10s} {'Replaced':<9s} {'Clean WER':<10s} {'Other WER':<10s} {'Avg WER':<10s}")
    print(f"  {'-'*6} {'-'*6} {'-'*12} {'-'*10} {'-'*9} {'-'*10} {'-'*10} {'-'*10}")

    for r in results:
        clean_str = f"{r['wer_clean']:.2f}%" if r['wer_clean'] else "N/A"
        other_str = f"{r['wer_other']:.2f}%" if r['wer_other'] else "N/A"
        avg_str = f"{r['wer_avg']:.2f}%" if r['wer_avg'] else "N/A"
        print(f"  {r['attn_threshold']:<6.3f} {r['mlp_threshold']:<6.3f} {r['encoder_params']:>10,}  {r['encoder_reduction_pct']:>7.1f}%  {r['layers_replaced']:>7d}   {clean_str:<10s} {other_str:<10s} {avg_str:<10s}")

    print(f"{'='*100}")

    # ── Find best configurations ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("BEST CONFIGURATIONS")
    print(f"{'='*60}")

    # Best: lowest avg WER with >5% reduction
    valid_results = [r for r in results if r['wer_avg'] is not None and r['encoder_reduction_pct'] > 5]
    if valid_results:
        best_wer = min(valid_results, key=lambda x: x['wer_avg'])
        print(f"\n  Best WER (with >5% reduction):")
        print(f"    Threshold: attn={best_wer['attn_threshold']}, mlp={best_wer['mlp_threshold']}")
        print(f"    Avg WER: {best_wer['wer_avg']}%")
        print(f"    Encoder reduction: {best_wer['encoder_reduction_pct']}%")

    # Best: highest reduction with <1% WER increase over baseline
    baseline_results = [r for r in results if r['attn_threshold'] == 0.99 and r['mlp_threshold'] == 0.999]
    if baseline_results and baseline_results[0]['wer_avg']:
        baseline_wer = baseline_results[0]['wer_avg']
        near_lossless = [r for r in results if r['wer_avg'] and r['wer_avg'] <= baseline_wer + 1.0]
        if near_lossless:
            best_compression = max(near_lossless, key=lambda x: x['encoder_reduction_pct'])
            print(f"\n  Best compression (within 1% WER of baseline {baseline_wer}%):")
            print(f"    Threshold: attn={best_compression['attn_threshold']}, mlp={best_compression['mlp_threshold']}")
            print(f"    Avg WER: {best_compression['wer_avg']}%")
            print(f"    Encoder reduction: {best_compression['encoder_reduction_pct']}%")

    # Best tradeoff: highest (reduction / WER_increase)
    if baseline_results and baseline_results[0]['wer_avg']:
        tradeoff = []
        for r in results:
            if r['wer_avg'] and r['encoder_reduction_pct'] > 0:
                wer_increase = max(r['wer_avg'] - baseline_wer, 0.01)  # avoid div by 0
                score = r['encoder_reduction_pct'] / wer_increase
                tradeoff.append((score, r))
        if tradeoff:
            best_tradeoff = max(tradeoff, key=lambda x: x[0])[1]
            print(f"\n  Best tradeoff (max reduction per WER point):")
            print(f"    Threshold: attn={best_tradeoff['attn_threshold']}, mlp={best_tradeoff['mlp_threshold']}")
            print(f"    Avg WER: {best_tradeoff['wer_avg']}%")
            print(f"    Encoder reduction: {best_tradeoff['encoder_reduction_pct']}%")

    print(f"{'='*60}")

    # ── Save results ──────────────────────────────────────────────────────
    output_path = args.output or "sweep_results.json"
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "device": str(device),
        "calibration_samples": args.num_calibration_samples,
        "max_eval_samples": args.max_eval_samples,
        "original_encoder_params": ORIG_ENCODER_PARAMS,
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for LiteASR compression")
    parser.add_argument("--model", type=str, default="usefulsensors/moonshine-base")
    parser.add_argument("--num_calibration_samples", type=int, default=100)
    parser.add_argument("--max_eval_samples", type=int, default=None,
                        help="Max samples per eval dataset (default: all). Use 100-200 for fast sweep.")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    main(args)
