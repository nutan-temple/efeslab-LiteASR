"""
Apply SpQR quantization ON TOP of a LiteASR-compressed Moonshine .pth checkpoint.

Pipeline:
  1. Load original Moonshine model from HuggingFace
  2. Replace encoder layers with LinearLowRank from .pth (LiteASR compression)
  3. Run SpQR (GPTQ + outlier detection) on ALL linear layers
  4. Evaluate WER on LibriSpeech test-clean and test-other
  5. Report: params, model size, WER before/after quantization

Usage:
    python quantize_liteasr_pth.py \
        --pth_path /path/to/lite-moonshine-moonshine-base_0.98:0.99.pth \
        --wbits 4 \
        --groupsize 16 \
        --perchannel \
        --nsamples 128 \
        --max_eval_samples 200

    # 3-bit (more aggressive):
    python quantize_liteasr_pth.py \
        --pth_path /path/to/lite-moonshine-moonshine-base_0.98:0.99.pth \
        --wbits 3 \
        --groupsize 16 \
        --perchannel \
        --outlier_threshold 0.2 \
        --nsamples 128

    # 8-bit (safest):
    python quantize_liteasr_pth.py \
        --pth_path /path/to/lite-moonshine-moonshine-base_0.98:0.99.pth \
        --wbits 8 \
        --nsamples 64
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
from tqdm import tqdm, trange
from datasets import load_dataset, Audio
from transformers import MoonshineForConditionalGeneration, AutoProcessor

from spqr_engine import SPQRUtil, Quantizer, quantize
from modelutils import find_sublayers, get_sequential_groups, get_decoder_sequential_groups


# ── Constants ─────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
TOKEN_LIMIT_FACTOR = 6.5 / SAMPLE_RATE


class LinearLowRank(nn.Module):
    """Low-rank layer from LiteASR."""
    def __init__(self, weight1, weight2, bias):
        super().__init__()
        self.weight1 = nn.Parameter(weight1)
        self.weight2 = nn.Parameter(weight2)
        self.bias = nn.Parameter(bias)

    def forward(self, x):
        return (x @ self.weight1) @ self.weight2 + self.bias


# ── Step 1: Load LiteASR model ────────────────────────────────────────────────
def load_liteasr_model(pth_path, model_name):
    """Load Moonshine with LiteASR-compressed encoder from .pth."""
    print(f"Loading base model: {model_name}")
    model = MoonshineForConditionalGeneration.from_pretrained(model_name)
    config = model.config

    print(f"Loading LiteASR weights: {pth_path}")
    state_dict = torch.load(pth_path, map_location="cpu", weights_only=False)

    encoder_sd = {}
    for key, tensor in state_dict.items():
        if key.startswith("model.encoder."):
            encoder_sd[key[len("model."):]] = tensor

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
                setattr_nested(layer, attr_path, LinearLowRank(w1, w2, bias))
                replaced += 1
                encoder_sd.pop(f"encoder.layers.{i}.{attr_path}.weight", None)

    model.model.load_state_dict(encoder_sd, strict=False)
    print(f"  Replaced {replaced} encoder layers with LinearLowRank")
    return model


def setattr_nested(obj, path, value):
    """Set a nested attribute: setattr_nested(layer, 'self_attn.q_proj', module)"""
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


# ── Step 2: Prepare calibration data ─────────────────────────────────────────
def get_calibration_audio(nsamples, audio_len=160000, seed=42):
    """Load real audio from LibriSpeech for calibration."""
    print(f"Loading calibration data ({nsamples} samples from LibriSpeech dev)...")
    ds = load_dataset("openslr/librispeech_asr", "clean", split="validation")
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    ds = ds.shuffle(seed=seed)

    audio_samples = []
    for i in range(min(nsamples, len(ds))):
        audio = ds[i]["audio"]["array"].astype(np.float32)
        # Pad or truncate to fixed length for uniform batching
        if len(audio) < audio_len:
            audio = np.pad(audio, (0, audio_len - len(audio)))
        else:
            audio = audio[:audio_len]
        audio_samples.append(torch.tensor(audio).unsqueeze(0))  # (1, audio_len)

    print(f"  Loaded {len(audio_samples)} audio samples")
    return audio_samples


# ── Step 3: SpQR quantization of encoder ──────────────────────────────────────
@torch.no_grad()
def run_conv_frontend(model, audio, device):
    """Run Moonshine encoder conv frontend."""
    x = audio.unsqueeze(1).to(device)  # (1, 1, audio_len)
    x = torch.nn.functional.tanh(model.model.encoder.conv1(x))
    x = model.model.encoder.groupnorm(x)
    x = torch.nn.functional.gelu(model.model.encoder.conv2(x))
    x = torch.nn.functional.gelu(model.model.encoder.conv3(x))
    x = x.permute(0, 2, 1)  # (1, seq_len, hidden_size)
    return x


@torch.no_grad()
def quantize_encoder_spqr(model, audio_samples, args, device):
    """Quantize encoder transformer layers using SpQR."""
    print("\n" + "=" * 60)
    print("QUANTIZING ENCODER (SpQR)")
    print("=" * 60)

    nsamples = len(audio_samples)

    # Get encoder layer inputs via conv frontend
    print("  Running conv frontend...")
    hidden_list = []
    for audio in audio_samples:
        hs = run_conv_frontend(model, audio, device)
        hidden_list.append(hs)

    # Pad to same length
    max_seq = max(h.shape[1] for h in hidden_list)
    hidden_size = hidden_list[0].shape[2]
    dtype = hidden_list[0].dtype
    inps = torch.zeros((nsamples, max_seq, hidden_size), dtype=dtype, device=device)
    for i, hs in enumerate(hidden_list):
        inps[i, :hs.shape[1], :] = hs[0]

    outs = torch.zeros_like(inps)

    # Position embeddings
    position_ids = torch.arange(max_seq, device=device).unsqueeze(0)
    position_embeddings = model.model.encoder.rotary_emb(inps[:1], position_ids=position_ids)
    forward_args = {"position_embeddings": position_embeddings}

    layers = model.model.encoder.layers
    total_outliers = 0
    total_weights = 0

    for i in range(len(layers)):
        print(f"\n  Encoder Layer {i}/{len(layers)}")
        layer = layers[i].to(device)

        # Prepare forward args for this layer
        layer_fwd = {}
        for k, v in forward_args.items():
            if isinstance(v, tuple):
                layer_fwd[k] = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in v)
            elif isinstance(v, torch.Tensor):
                layer_fwd[k] = v.to(device)
            else:
                layer_fwd[k] = v

        all_sublayers = find_sublayers(layer)
        sequential = get_sequential_groups(model)

        for names in sequential:
            subset = {n: all_sublayers[n] for n in names if n in all_sublayers}
            if not subset:
                continue

            # Collect Hessian
            spqr_handlers = {}
            for name in subset:
                spqr_handlers[name] = SPQRUtil(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    spqr_handlers[name].add_batch(inp[0].data)
                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            for j in range(nsamples):
                layer_out = layer(inps[j].unsqueeze(0), **layer_fwd)
                outs[j] = layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out

            for h in handles:
                h.remove()

            # Quantize each sublayer
            for name in subset:
                print(f"    Quantizing {name}...", end=" ")
                quantized = spqr_handlers[name].quantize(
                    percdamp=args.percdamp,
                    bits=args.wbits,
                    groupsize=args.groupsize,
                    sym=args.sym,
                    perchannel=args.perchannel,
                    qq_groupsize=args.qq_groupsize,
                    round_zero=args.round_zero,
                    qq_scale_bits=args.qq_scale_bits,
                    qq_zero_bits=args.qq_zero_bits,
                    qq_zero_sym=False,
                    outlier_relative_threshold=args.outlier_threshold,
                    permutation_order=args.permutation_order,
                    simplified_outliers=args.simplified_outliers,
                    save_quantization=False,
                )
                # Replace weight with dequantized version
                spqr_handlers[name].layer.weight.data = quantized.weight.to(
                    spqr_handlers[name].layer.weight.data.dtype
                )
                n_outliers = quantized.unstructured_outlier_mask.sum().item()
                n_weights = quantized.weight.numel()
                total_outliers += n_outliers
                total_weights += n_weights
                print(f"outliers: {100*n_outliers/n_weights:.2f}%")

        # Recompute outs after quantization
        for j in range(nsamples):
            layer_out = layer(inps[j].unsqueeze(0), **layer_fwd)
            outs[j] = layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out

        layers[i] = layer.cpu()
        inps, outs = outs, inps

    # Apply final layer norm
    ln = model.model.encoder.layer_norm.to(device)
    encoder_outputs = torch.zeros_like(inps)
    for j in range(nsamples):
        encoder_outputs[j] = ln(inps[j].unsqueeze(0))
    model.model.encoder.layer_norm.cpu()

    print(f"\n  Encoder quantization complete.")
    print(f"  Total outlier share: {100*total_outliers/max(total_weights,1):.3f}%")
    return encoder_outputs


@torch.no_grad()
def quantize_decoder_spqr(model, encoder_outputs, args, device):
    """Quantize decoder transformer layers using SpQR."""
    print("\n" + "=" * 60)
    print("QUANTIZING DECODER (SpQR)")
    print("=" * 60)

    nsamples = encoder_outputs.shape[0]

    # Prepare decoder inputs (BOS token embeddings)
    bos_id = getattr(model.config, "decoder_start_token_id", 1) or 1
    dec_seq_len = 4
    embed = model.model.decoder.embed_tokens.to(device)

    torch.manual_seed(0)
    hidden_size = embed.weight.shape[1]
    dtype = embed.weight.dtype

    inps = torch.zeros((nsamples, dec_seq_len, hidden_size), dtype=dtype, device=device)
    for i in range(nsamples):
        tokens = torch.cat([
            torch.full((1, 1), bos_id, dtype=torch.long),
            torch.randint(0, model.config.vocab_size, (1, dec_seq_len - 1))
        ], dim=1).to(device)
        inps[i] = embed(tokens)[0]
    model.model.decoder.embed_tokens.cpu()

    outs = torch.zeros_like(inps)

    # Position embeddings
    dec_pos_ids = torch.arange(dec_seq_len, device=device).unsqueeze(0)
    position_embeddings = model.model.decoder.rotary_emb(inps[:1], position_ids=dec_pos_ids)

    enc_seq_len = encoder_outputs.shape[1]
    enc_pos_ids = torch.arange(enc_seq_len, device=device).unsqueeze(0)
    encoder_position_embeddings = model.model.encoder.rotary_emb(
        encoder_outputs[:1].to(device), position_ids=enc_pos_ids
    )

    forward_args = {
        "encoder_hidden_states": encoder_outputs,
        "position_embeddings": position_embeddings,
        "encoder_position_embeddings": encoder_position_embeddings,
    }

    layers = model.model.decoder.layers

    for i in range(len(layers)):
        print(f"\n  Decoder Layer {i}/{len(layers)}")
        layer = layers[i].to(device)

        layer_fwd = {}
        for k, v in forward_args.items():
            if isinstance(v, tuple):
                layer_fwd[k] = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in v)
            elif isinstance(v, torch.Tensor):
                layer_fwd[k] = v.to(device)
            else:
                layer_fwd[k] = v

        all_sublayers = find_sublayers(layer)
        sequential = get_decoder_sequential_groups(model)

        for names in sequential:
            subset = {n: all_sublayers[n] for n in names if n in all_sublayers}
            if not subset:
                continue

            spqr_handlers = {}
            for name in subset:
                spqr_handlers[name] = SPQRUtil(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    spqr_handlers[name].add_batch(inp[0].data)
                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            for j in range(nsamples):
                sample_fwd = dict(layer_fwd)
                if "encoder_hidden_states" in sample_fwd:
                    sample_fwd["encoder_hidden_states"] = encoder_outputs[j].unsqueeze(0).to(device)
                layer_out = layer(inps[j].unsqueeze(0), **sample_fwd)
                outs[j] = layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out

            for h in handles:
                h.remove()

            for name in subset:
                print(f"    Quantizing {name}...", end=" ")
                quantized = spqr_handlers[name].quantize(
                    percdamp=args.percdamp,
                    bits=args.wbits,
                    groupsize=args.groupsize,
                    sym=args.sym,
                    perchannel=args.perchannel,
                    qq_groupsize=args.qq_groupsize,
                    round_zero=args.round_zero,
                    qq_scale_bits=args.qq_scale_bits,
                    qq_zero_bits=args.qq_zero_bits,
                    qq_zero_sym=False,
                    outlier_relative_threshold=args.outlier_threshold,
                    permutation_order=args.permutation_order,
                    simplified_outliers=args.simplified_outliers,
                    save_quantization=False,
                )
                spqr_handlers[name].layer.weight.data = quantized.weight.to(
                    spqr_handlers[name].layer.weight.data.dtype
                )
                n_outliers = quantized.unstructured_outlier_mask.sum().item()
                n_weights = quantized.weight.numel()
                print(f"outliers: {100*n_outliers/n_weights:.2f}%")

        for j in range(nsamples):
            sample_fwd = dict(layer_fwd)
            if "encoder_hidden_states" in sample_fwd:
                sample_fwd["encoder_hidden_states"] = encoder_outputs[j].unsqueeze(0).to(device)
            layer_out = layer(inps[j].unsqueeze(0), **sample_fwd)
            outs[j] = layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out

        layers[i] = layer.cpu()
        inps, outs = outs, inps

    print("\n  Decoder quantization complete.")


# ── Step 4: Evaluate WER ──────────────────────────────────────────────────────
def evaluate_wer(model, processor, dataset_name, split, device, max_samples=None):
    """Evaluate WER on LibriSpeech."""
    ds = load_dataset("openslr/librispeech_asr", dataset_name, split=split)
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

    if max_samples and max_samples < len(ds):
        ds = ds.select(range(max_samples))

    wer_metric = evaluate.load("wer")
    preds, refs = [], []

    model = model.to(device).eval()

    for i in tqdm(range(len(ds)), desc=f"  Eval {dataset_name}-{split}"):
        audio = ds[i]["audio"]["array"].astype(np.float32)
        ref = ds[i].get("text", "").strip()
        if not ref:
            continue

        inputs = processor(audio, return_tensors="pt", sampling_rate=SAMPLE_RATE).to(device)
        seq_lens = inputs.attention_mask.sum(dim=-1)
        max_length = max(int((seq_lens * TOKEN_LIMIT_FACTOR).max().item()), 10)

        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_length=max_length)
        pred = processor.decode(gen_ids[0], skip_special_tokens=True)

        preds.append(pred.strip().lower())
        refs.append(ref.strip().lower())

    wer = wer_metric.compute(references=refs, predictions=preds)
    return round(100 * wer, 2)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SpQR quantization on top of LiteASR .pth")
    parser.add_argument("--pth_path", type=str, required=True, help="Path to LiteASR .pth")
    parser.add_argument("--model", type=str, default="usefulsensors/moonshine-base")
    parser.add_argument("--wbits", type=int, default=4, help="Weight bits (3, 4, or 8)")
    parser.add_argument("--groupsize", type=int, default=16)
    parser.add_argument("--perchannel", action="store_true")
    parser.add_argument("--sym", action="store_true")
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--permutation_order", type=str, default="identity")
    parser.add_argument("--outlier_threshold", type=float, default=float("inf"))
    parser.add_argument("--simplified_outliers", action="store_true")
    parser.add_argument("--qq_scale_bits", type=int, default=None)
    parser.add_argument("--qq_zero_bits", type=int, default=None)
    parser.add_argument("--qq_groupsize", type=int, default=16)
    parser.add_argument("--round_zero", type=int, default=None)
    parser.add_argument("--part", type=str, default="both", choices=["encoder", "decoder", "both"])
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"LiteASR checkpoint: {args.pth_path}")
    print(f"Quantization: W{args.wbits}, groupsize={args.groupsize}, outlier_thresh={args.outlier_threshold}")
    print()

    # Load processor
    processor = AutoProcessor.from_pretrained(args.model)

    # Load LiteASR-compressed model
    model = load_liteasr_model(args.pth_path, args.model)

    # Print pre-quantization stats
    enc_params = sum(p.numel() for p in model.model.encoder.parameters())
    dec_params = sum(p.numel() for p in model.model.decoder.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"\nPre-quantization:")
    print(f"  Encoder: {enc_params:,} | Decoder: {dec_params:,} | Total: {total_params:,}")
    print(f"  Model size: {size_bytes/1e6:.1f} MB")

    # Evaluate BEFORE quantization
    print(f"\n{'='*60}")
    print("EVALUATING BEFORE QUANTIZATION")
    print(f"{'='*60}")
    model_on_device = model.to(device)
    wer_clean_before = evaluate_wer(model_on_device, processor, "clean", "test", device, args.max_eval_samples)
    wer_other_before = evaluate_wer(model_on_device, processor, "other", "test", device, args.max_eval_samples)
    print(f"  WER clean: {wer_clean_before}% | WER other: {wer_other_before}%")
    model = model.cpu()

    # Prepare calibration audio
    audio_samples = get_calibration_audio(args.nsamples)

    # Quantize
    encoder_outputs = None
    if args.part in ("encoder", "both"):
        encoder_outputs = quantize_encoder_spqr(model, audio_samples, args, device)

    if args.part in ("decoder", "both"):
        if encoder_outputs is None:
            # Need encoder outputs for decoder calibration
            print("Running encoder to get outputs for decoder...")
            encoder_outputs = quantize_encoder_spqr(model, audio_samples, args, device)
        quantize_decoder_spqr(model, encoder_outputs, args, device)

    # Evaluate AFTER quantization
    print(f"\n{'='*60}")
    print("EVALUATING AFTER QUANTIZATION")
    print(f"{'='*60}")
    model_on_device = model.to(device)
    wer_clean_after = evaluate_wer(model_on_device, processor, "clean", "test", device, args.max_eval_samples)
    wer_other_after = evaluate_wer(model_on_device, processor, "other", "test", device, args.max_eval_samples)
    print(f"  WER clean: {wer_clean_after}% | WER other: {wer_other_after}%")

    # Summary
    print(f"\n\n{'='*70}")
    print("FINAL RESULTS: LiteASR + SpQR W{} Quantization".format(args.wbits))
    print(f"{'='*70}")
    print(f"  {'Metric':<25s} {'Before SpQR':<20s} {'After SpQR W{}':<20s}".format(args.wbits))
    print(f"  {'-'*25} {'-'*20} {'-'*20}")
    print(f"  {'Parameters':<25s} {total_params:>15,}   {total_params:>15,}")
    print(f"  {'Model size':<25s} {size_bytes/1e6:>14.1f} MB  {'~'+str(round(size_bytes/1e6 * args.wbits/32, 1))+' MB':<20s}")
    print(f"  {'Effective bits/weight':<25s} {'32':>15s}   {str(args.wbits):>15s}")
    print(f"  {'WER test-clean':<25s} {wer_clean_before:>14}%   {wer_clean_after:>14}%")
    print(f"  {'WER test-other':<25s} {wer_other_before:>14}%   {wer_other_after:>14}%")
    avg_before = round((wer_clean_before + wer_other_before) / 2, 2)
    avg_after = round((wer_clean_after + wer_other_after) / 2, 2)
    print(f"  {'WER average':<25s} {avg_before:>14}%   {avg_after:>14}%")
    print(f"  {'WER degradation':<25s} {'(baseline)':<20s} {'+' + str(round(avg_after - avg_before, 2)) + '%':<20s}")
    print(f"{'='*70}")

    # Save results
    output_path = args.output or f"spqr_w{args.wbits}_on_liteasr_results.json"
    results = {
        "timestamp": datetime.now().isoformat(),
        "pth_path": args.pth_path,
        "wbits": args.wbits,
        "groupsize": args.groupsize,
        "outlier_threshold": args.outlier_threshold,
        "nsamples": args.nsamples,
        "part": args.part,
        "total_params": total_params,
        "encoder_params": enc_params,
        "size_before_mb": round(size_bytes / 1e6, 1),
        "effective_size_mb": round(size_bytes / 1e6 * args.wbits / 32, 1),
        "wer_clean_before": wer_clean_before,
        "wer_other_before": wer_other_before,
        "wer_clean_after": wer_clean_after,
        "wer_other_after": wer_other_after,
        "wer_avg_before": avg_before,
        "wer_avg_after": avg_after,
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    if args.save:
        print(f"Saving quantized model to: {args.save}")
        torch.save(model.state_dict(), args.save)


if __name__ == "__main__":
    main()
