"""
Run LiteASR hyperparameter sweep on Modal (cloud GPU).

Setup:
    1. pip install modal
    2. modal token new  (authenticate)
    3. python src/sweep_modal.py

This will:
    - Spin up a GPU instance on Modal (A10G by default)
    - Run the full sweep (21 configs)
    - Download results to local machine

To change GPU type:
    python src/sweep_modal.py --gpu a100
    python src/sweep_modal.py --gpu t4
    python src/sweep_modal.py --gpu a10g
"""

import modal

# ── Modal App Setup ───────────────────────────────────────────────────────────
app = modal.App("liteasr-sweep")

# Docker image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers>=4.49.0",
        "datasets",
        "evaluate",
        "numpy",
        "tqdm",
        "sentencepiece",
        "jiwer",
        "librosa",
        "soundfile",
    )
)

# Persistent volume to cache models/datasets across runs
volume = modal.Volume.from_name("liteasr-cache", create_if_missing=True)


# ── The actual sweep function that runs on GPU ────────────────────────────────
@app.function(
    image=image,
    gpu="a10g",  # Options: "t4", "a10g", "a100", "h100"
    timeout=7200,  # 2 hours max
    volumes={"/cache": volume},
    secrets=[modal.Secret.from_name("huggingface-secret", required=False)],
)
def run_sweep(
    model_name: str = "usefulsensors/moonshine-base",
    num_calibration_samples: int = 100,
    max_eval_samples: int = None,
):
    """Run the full hyperparameter sweep on a cloud GPU."""
    import os
    os.environ["HF_HOME"] = "/cache/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "/cache/huggingface"

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

    # ── Config ────────────────────────────────────────────────────────────
    SAMPLE_RATE = 16000
    TOKEN_LIMIT_FACTOR = 6.5 / SAMPLE_RATE
    ORIG_ENCODER_PARAMS = 20_153_120

    device = torch.device("cuda:0")
    torch_dtype = torch.float32  # float32 for compression accuracy
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB\n")

    # ── LinearLowRank ─────────────────────────────────────────────────────
    class LinearLowRank(nn.Module):
        def __init__(self, weight1, weight2, bias):
            super().__init__()
            self.weight1 = nn.Parameter(weight1)
            self.weight2 = nn.Parameter(weight2)
            self.bias = nn.Parameter(bias)

        def forward(self, x):
            return (x @ self.weight1) @ self.weight2 + self.bias

    # ── Compression function ──────────────────────────────────────────────
    def apply_low_rank(model, calibration_data, rank_threshold, processor):
        attn_thresh, mlp_thresh = [float(x) for x in rank_threshold.split(":")]
        base_dim = model.config.hidden_size
        num_layers = model.config.encoder_num_hidden_layers

        # Collect activations
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

        model.eval()
        with torch.no_grad():
            for sample in calibration_data:
                audio = sample["audio"]["array"].astype(np.float32)
                inputs = processor(audio, return_tensors="pt", sampling_rate=SAMPLE_RATE)
                inputs = inputs.to(device)
                model.generate(**inputs, max_length=10)

        for h in hooks:
            h.remove()

        # SVD + replace
        component_names = ["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"]
        replaced = 0

        for i_layer in range(num_layers):
            layer = encoder_layers[i_layer]
            for i, comp_name in enumerate(component_names):
                if comp_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    parts = f"self_attn.{comp_name}".split(".")
                else:
                    parts = f"mlp.{comp_name}".split(".")

                obj = layer
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                linear_module = getattr(obj, parts[-1])

                if isinstance(linear_module, LinearLowRank):
                    continue

                flattened = []
                for feat in calibration_activations[i_layer][comp_name]:
                    flattened.append(feat.reshape(-1, feat.shape[-1]))
                features = torch.cat(flattened, dim=0).float()

                thresh = attn_thresh if i <= 3 else mlp_thresh

                Y_mean = features.mean(dim=0)
                features_centered = features - Y_mean
                U, S, Vt = torch.linalg.svd(features_centered, full_matrices=False)

                S_F = S ** 2
                k = -1
                for j in range(16, len(S_F) + 1, 16):
                    if S_F[:j].sum() / S_F.sum() > thresh:
                        k = j
                        break

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

        enc_params = sum(p.numel() for p in model.model.encoder.parameters())
        return model, enc_params, replaced

    # ── Evaluation function ───────────────────────────────────────────────
    def evaluate_wer(model, processor, dataset, max_samples=None):
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

    # ── Load data ─────────────────────────────────────────────────────────
    processor = AutoProcessor.from_pretrained(model_name)

    print("Loading calibration data...")
    cal_clean = load_dataset("librispeech_asr", "clean", split="validation")
    cal_other = load_dataset("librispeech_asr", "other", split="validation")
    cal_clean = cal_clean.cast_column("audio", Audio(sampling_rate=16000))
    cal_other = cal_other.cast_column("audio", Audio(sampling_rate=16000))

    n_per = num_calibration_samples // 2
    cal_data = concatenate_datasets([
        cal_clean.shuffle(seed=42).select(range(min(n_per, len(cal_clean)))),
        cal_other.shuffle(seed=42).select(range(min(n_per, len(cal_other)))),
    ]).shuffle(seed=42)
    print(f"  Calibration: {len(cal_data)} samples")

    print("Loading eval data...")
    eval_clean = load_dataset("librispeech_asr", "clean", split="test")
    eval_other = load_dataset("librispeech_asr", "other", split="test")
    eval_clean = eval_clean.cast_column("audio", Audio(sampling_rate=16000))
    eval_other = eval_other.cast_column("audio", Audio(sampling_rate=16000))
    print(f"  test-clean: {len(eval_clean)}, test-other: {len(eval_other)}\n")

    # ── Sweep grid ────────────────────────────────────────────────────────
    attn_thresholds = [0.99, 0.98, 0.97]
    mlp_thresholds = [0.999, 0.995, 0.99, 0.985, 0.98, 0.975, 0.97]

    sweep_configs = [(a, m) for a in attn_thresholds for m in mlp_thresholds]
    print(f"Sweep: {len(sweep_configs)} configurations\n")
    print("=" * 100)

    # ── Run sweep ─────────────────────────────────────────────────────────
    results = []

    for idx, (attn_t, mlp_t) in enumerate(sweep_configs):
        threshold_str = f"{attn_t}:{mlp_t}"
        print(f"\n[{idx+1}/{len(sweep_configs)}] attn={attn_t}, mlp={mlp_t}")

        model = MoonshineForConditionalGeneration.from_pretrained(model_name).to(device)
        model.eval()

        start = time.time()
        model, enc_params, n_replaced = apply_low_rank(model, cal_data, threshold_str, processor)
        compress_time = time.time() - start

        enc_reduction = round(100 * (1 - enc_params / ORIG_ENCODER_PARAMS), 1)
        print(f"  Enc params: {enc_params:,} ({enc_reduction}% reduction), {n_replaced} layers replaced")

        wer_clean = evaluate_wer(model, processor, eval_clean, max_eval_samples)
        wer_other = evaluate_wer(model, processor, eval_other, max_eval_samples)
        wer_avg = round((wer_clean + wer_other) / 2, 2) if wer_clean and wer_other else None

        print(f"  WER: clean={wer_clean}%, other={wer_other}%, avg={wer_avg}%")

        results.append({
            "attn_threshold": attn_t,
            "mlp_threshold": mlp_t,
            "encoder_params": enc_params,
            "encoder_reduction_pct": enc_reduction,
            "layers_replaced": n_replaced,
            "wer_clean": wer_clean,
            "wer_other": wer_other,
            "wer_avg": wer_avg,
            "compress_time_s": round(compress_time, 1),
        })

        del model
        torch.cuda.empty_cache()

    # ── Print table ───────────────────────────────────────────────────────
    print(f"\n\n{'='*100}")
    print("RESULTS TABLE")
    print(f"{'='*100}")
    print(f"  {'Attn':<6} {'MLP':<6} {'Enc Params':<12} {'Reduction':<10} {'Clean':<8} {'Other':<8} {'Avg':<8}")
    print(f"  {'-'*6} {'-'*6} {'-'*12} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
    for r in results:
        print(f"  {r['attn_threshold']:<6.3f} {r['mlp_threshold']:<6.3f} {r['encoder_params']:>10,}  {r['encoder_reduction_pct']:>7.1f}%  {r['wer_clean'] or 'N/A':<8} {r['wer_other'] or 'N/A':<8} {r['wer_avg'] or 'N/A':<8}")
    print(f"{'='*100}")

    # ── Best configs ──────────────────────────────────────────────────────
    baseline = next((r for r in results if r['attn_threshold'] == 0.99 and r['mlp_threshold'] == 0.999), None)
    if baseline and baseline['wer_avg']:
        bwer = baseline['wer_avg']
        near_lossless = [r for r in results if r['wer_avg'] and r['wer_avg'] <= bwer + 1.0 and r['encoder_reduction_pct'] > 5]
        if near_lossless:
            best = max(near_lossless, key=lambda x: x['encoder_reduction_pct'])
            print(f"\n  BEST (within 1% WER of baseline {bwer}%):")
            print(f"    attn={best['attn_threshold']}, mlp={best['mlp_threshold']}")
            print(f"    Reduction: {best['encoder_reduction_pct']}%, Avg WER: {best['wer_avg']}%")

    # ── Save ──────────────────────────────────────────────────────────────
    output = {
        "timestamp": datetime.now().isoformat(),
        "gpu": torch.cuda.get_device_name(0),
        "model": model_name,
        "calibration_samples": num_calibration_samples,
        "max_eval_samples": max_eval_samples,
        "results": results,
    }

    # Save to volume for persistence
    with open("/cache/sweep_results.json", "w") as f:
        json.dump(output, f, indent=2)

    return output


# ── Local entrypoint ──────────────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="usefulsensors/moonshine-base")
    parser.add_argument("--num_calibration_samples", type=int, default=100)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--gpu", type=str, default="a10g", choices=["t4", "a10g", "a100", "h100"])
    parser.add_argument("--output", type=str, default="sweep_results.json")
    args = parser.parse_args()

    print(f"Launching sweep on Modal ({args.gpu} GPU)...")
    print(f"  Model: {args.model}")
    print(f"  Calibration: {args.num_calibration_samples} samples")
    print(f"  Eval: {args.max_eval_samples or 'all'} samples per dataset")
    print()

    # Run on Modal
    result = run_sweep.remote(
        model_name=args.model,
        num_calibration_samples=args.num_calibration_samples,
        max_eval_samples=args.max_eval_samples,
    )

    # Save locally
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n\nResults saved to: {args.output}")

    # Print summary
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"  {'Attn':<6} {'MLP':<6} {'Reduction':<10} {'Clean WER':<10} {'Other WER':<10} {'Avg WER':<10}")
    print(f"  {'-'*6} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for r in result["results"]:
        print(f"  {r['attn_threshold']:<6.3f} {r['mlp_threshold']:<6.3f} {r['encoder_reduction_pct']:>7.1f}%   {r['wer_clean'] or 'N/A':<10} {r['wer_other'] or 'N/A':<10} {r['wer_avg'] or 'N/A':<10}")
    print(f"{'='*80}")
