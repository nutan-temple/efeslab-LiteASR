"""
Encoder-only SpQR quantization for LiteASR-compressed Moonshine models.

This applies the REAL SpQR algorithm -- GPTQ Hessian-aware error propagation +
unstructured outlier detection + (act_order) column permutation + meta-
quantization of the per-group scales/zeros -- to the ENCODER transformer layers
of a Moonshine model. The decoder is left in full precision.

The quantization engine is used VERBATIM from the upstream SpQR repository
(https://github.com/Vahe1994/SpQR):

  * spqr_engine.py        -- SPQRUtil: GPTQ + outlier detection
  * quant_groups.py       -- Quantizer (per-group scale/zero, meta-quant)
  * weight_permutation.py -- get_permutation_order (act_order / spearman)

These three files are copied byte-for-byte and are NOT modified. This is a true
SpQR/GPTQ quantizer, not a naive round-to-int8 scheme.

Pipeline
--------
1. Load `usefulsensors/moonshine-base` via MoonshineForConditionalGeneration.
2. (optional) Load a LiteASR .pth and swap encoder sublayers for LinearLowRank.
3. Quantize ONLY `model.model.encoder.layers` with the verbatim SPQRUtil,
   processing sublayers in GPTQ-sequential groups so later projections are
   calibrated against the already-quantized activations of earlier ones.
4. Evaluate WER on LibriSpeech test-clean + test-other with the reference recipe
   (fp16 on GPU, token-limited generation, punctuation-stripped normalization).
   Baseline: 3.38% test-clean / 9.38% test-other.
5. Report compression for the ENCODER ONLY (decoder stays fp32).

Examples
--------
    # 8-bit encoder-only SpQR with outlier detection, full eval:
    python quantize_encoder_spqr.py \
        --pth_path lite-moonshine-moonshine-base_0.99:0.999.pth \
        --wbits 8 --groupsize 16 --perchannel \
        --qq_scale_bits 3 --qq_zero_bits 3 \
        --outlier_threshold 0.2 --permutation_order act_order \
        --nsamples 128

    # Dense base encoder (no .pth), quick check on a few eval samples:
    python quantize_encoder_spqr.py --wbits 8 --max_eval_samples 50

    # True dynamic W8A8: SpQR int8 weights + dynamic int8 activations:
    python quantize_encoder_spqr.py \
        --pth_path lite-moonshine-moonshine-base_0.99:0.999.pth \
        --wbits 8 --groupsize 16 --perchannel \
        --quantize_activations --act_granularity per_token \
        --nsamples 128

    # CPU smoke test (no eval, one layer):
    python quantize_encoder_spqr.py --nsamples 2 --skip_eval --max_layers 1
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

# Make the verbatim SpQR engine importable regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spqr_engine import SPQRUtil  # noqa: E402  (verbatim upstream)
from modelutils import (  # noqa: E402
    LinearLowRank,
    load_moonshine_model,
    load_liteasr_pth,
    get_processor,
    find_sublayers,
    get_encoder_sequential_groups,
    count_parameters,
)


SAMPLE_RATE = 16000


class _WeightProxy(nn.Module):
    """Minimal module exposing a single `.weight` attribute for SPQRUtil.

    SPQRUtil only needs `layer.weight` (an [out, in] tensor) to build the
    Hessian and run GPTQ. We wrap one factor of a LinearLowRank layer so the
    verbatim engine can quantize it unchanged.
    """

    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight.detach().clone(), requires_grad=False)


def qcfg_for(out_features, base_qcfg):
    """Adapt the meta-quant group size to the matrix's output dimension.

    SpQR meta-quantizes the per-output-channel scales/zeros in groups of
    `qq_groupsize`, which requires `out_features % qq_groupsize == 0`. Real
    LiteASR ranks are multiples of 16 so the default just works, but to stay
    robust on arbitrary dimensions we shrink the group size to a divisor (gcd).
    If no useful grouping exists we keep scales/zeros in fp16.
    """
    cfg = dict(base_qcfg)
    if cfg.get("qq_scale_bits") is None and cfg.get("qq_zero_bits") is None:
        return cfg
    g = int(cfg.get("qq_groupsize", 16))
    if g <= 0 or out_features % g != 0:
        g = math.gcd(out_features, g)
    if g <= 1:
        cfg["qq_scale_bits"] = None
        cfg["qq_zero_bits"] = None
    else:
        cfg["qq_groupsize"] = g
    return cfg


@torch.no_grad()
def run_conv_frontend(model, audio, device):
    """Run the Moonshine encoder conv frontend -> (1, seq_len, hidden)."""
    enc = model.model.encoder
    x = audio.unsqueeze(1).to(device)  # (1, 1, audio_len)
    x = torch.nn.functional.tanh(enc.conv1(x))
    x = enc.groupnorm(x)
    x = torch.nn.functional.gelu(enc.conv2(x))
    x = torch.nn.functional.gelu(enc.conv3(x))
    x = x.permute(0, 2, 1)  # (1, seq_len, hidden)
    return x


@torch.no_grad()
def quantize_encoder_spqr(model, audio_samples, qcfg, device, max_layers=None):
    """Quantize encoder transformer layers with the verbatim SPQRUtil."""
    print("\n" + "=" * 70)
    print("ENCODER SpQR QUANTIZATION (verbatim GPTQ + outlier detection)")
    print("=" * 70)

    encoder = model.model.encoder
    rotary = encoder.rotary_emb
    layers = encoder.layers
    n_layers = len(layers) if max_layers is None else min(max_layers, len(layers))
    groups = get_encoder_sequential_groups()

    # Per-sample encoder-layer inputs (variable sequence lengths -> keep a list,
    # which avoids polluting the Hessian with zero-padding).
    print(f"  Running conv frontend on {len(audio_samples)} calibration clips...")
    inps = [run_conv_frontend(model, a, device) for a in audio_samples]

    stats = {"quantized_params": 0, "total_outliers": 0}

    for li in range(n_layers):
        layer = layers[li].to(device)
        print(f"\n  Encoder layer {li}/{n_layers - 1}")

        # Rotary position embeddings depend on (variable) sequence length.
        pos_embs = []
        for x in inps:
            seq = x.shape[1]
            pos_ids = torch.arange(seq, device=device).unsqueeze(0)
            pos_embs.append(rotary(x, position_ids=pos_ids))

        all_sublayers = find_sublayers(layer)

        for names in groups:
            subset = {n: all_sublayers[n] for n in names if n in all_sublayers}
            if not subset:
                continue

            handlers = {}
            hooks = []
            for name, module in subset.items():
                if isinstance(module, LinearLowRank):
                    # W1 (in, rank) and W2 (rank, out). As nn.Linear-style
                    # [out, in] weights these are W1.t() and W2.t().
                    h1 = SPQRUtil(_WeightProxy(module.weight1.t()).to(device))
                    h2 = SPQRUtil(_WeightProxy(module.weight2.t()).to(device))
                    handlers[name] = {"kind": "lowrank", "module": module, "w1": h1, "w2": h2}

                    def make_lowrank_hook(mod, hh1, hh2):
                        def hook(_m, inp, _out):
                            x = inp[0].data
                            hh1.add_batch(x)                       # input to W1 is x
                            hh2.add_batch((x @ mod.weight1).data)  # input to W2 is x @ W1
                        return hook

                    hooks.append(module.register_forward_hook(make_lowrank_hook(module, h1, h2)))
                else:  # nn.Linear
                    h = SPQRUtil(module)
                    handlers[name] = {"kind": "linear", "module": module, "h": h}

                    def make_linear_hook(hh):
                        def hook(_m, inp, _out):
                            hh.add_batch(inp[0].data)
                        return hook

                    hooks.append(module.register_forward_hook(make_linear_hook(h)))

            # Accumulate Hessians by forwarding the (partially quantized) layer.
            for x, pe in zip(inps, pos_embs):
                layer(x, position_embeddings=pe)
            for h in hooks:
                h.remove()

            # Quantize each sublayer in the group with the verbatim engine.
            for name, hd in handlers.items():
                if hd["kind"] == "linear":
                    module = hd["module"]
                    res = hd["h"].quantize(**qcfg_for(module.weight.shape[0], qcfg))
                    module.weight.data = res.weight.to(module.weight.dtype)
                    n_out = int(res.unstructured_outlier_mask.sum().item())
                    n_w = res.weight.numel()
                    print(f"    {name:<22s} [linear  {tuple(res.weight.shape)}]  "
                          f"outliers={100 * n_out / max(n_w, 1):.2f}%")
                else:
                    module = hd["module"]
                    res1 = hd["w1"].quantize(**qcfg_for(module.weight1.shape[1], qcfg))
                    res2 = hd["w2"].quantize(**qcfg_for(module.weight2.shape[1], qcfg))
                    module.weight1.data = res1.weight.t().to(module.weight1.dtype)
                    module.weight2.data = res2.weight.t().to(module.weight2.dtype)
                    n_out = int(res1.unstructured_outlier_mask.sum().item()
                                + res2.unstructured_outlier_mask.sum().item())
                    n_w = res1.weight.numel() + res2.weight.numel()
                    print(f"    {name:<22s} [lowrank W1{tuple(res1.weight.shape)} "
                          f"W2{tuple(res2.weight.shape)}]  "
                          f"outliers={100 * n_out / max(n_w, 1):.2f}%")
                stats["quantized_params"] += n_w
                stats["total_outliers"] += n_out

        # Recompute layer outputs (now fully quantized) -> inputs for next layer.
        new_inps = []
        for x, pe in zip(inps, pos_embs):
            out = layer(x, position_embeddings=pe)
            out = out[0] if isinstance(out, (tuple, list)) else out
            new_inps.append(out)
        inps = new_inps
        layers[li] = layer

    frac = stats["total_outliers"] / max(stats["quantized_params"], 1)
    stats["outlier_fraction"] = frac
    print(f"\n  Encoder quantization complete.")
    print(f"  Quantized encoder weights : {stats['quantized_params']:,}")
    print(f"  Outlier share (kept fp16) : {100 * frac:.3f}%")
    return stats


def report_compression(model, args, outlier_fraction, quantized_params):
    """Honest compression accounting: only the encoder is quantized."""
    enc_params, dec_params, total_params = count_parameters(model)

    # Effective bits per quantized weight: base wbits + meta-quant of scales/
    # zeros + a small fp16 budget for outliers (value + column index).
    if args.qq_scale_bits and args.qq_zero_bits:
        meta_bits = (args.qq_scale_bits + args.qq_zero_bits) / float(args.groupsize)
    else:
        meta_bits = (16 + 16) / float(args.groupsize)  # raw fp16 scale + zero
    outlier_bits = outlier_fraction * 32.0  # ~fp16 value (16) + index (16)
    eff_bits = args.wbits + meta_bits + outlier_bits

    fp32 = 4
    total_fp32_bytes = total_params * fp32
    quant_bytes = quantized_params * eff_bits / 8.0
    enc_unquant_bytes = (enc_params - quantized_params) * fp32  # conv/norm/bias stay fp32
    enc_compressed_bytes = quant_bytes + enc_unquant_bytes
    enc_fp32_bytes = enc_params * fp32
    other_bytes = (total_params - enc_params) * fp32  # decoder + heads stay fp32
    compressed_total_bytes = enc_compressed_bytes + other_bytes

    report = {
        "encoder_params": enc_params,
        "decoder_params": dec_params,
        "total_params": total_params,
        "quantized_params": quantized_params,
        "effective_bits_per_quant_weight": round(eff_bits, 3),
        "outlier_fraction": round(outlier_fraction, 5),
        "encoder_fp32_mb": round(enc_fp32_bytes / 1e6, 2),
        "encoder_compressed_mb": round(enc_compressed_bytes / 1e6, 2),
        "encoder_compression_x": round(enc_fp32_bytes / max(enc_compressed_bytes, 1), 3),
        "decoder_fp32_mb": round(dec_params * fp32 / 1e6, 2),
        "model_fp32_mb": round(total_fp32_bytes / 1e6, 2),
        "model_compressed_mb": round(compressed_total_bytes / 1e6, 2),
        "model_compression_x": round(total_fp32_bytes / max(compressed_total_bytes, 1), 3),
    }

    print("\n" + "=" * 70)
    print("COMPRESSION (encoder quantized, decoder fp32)")
    print("=" * 70)
    print(f"  Encoder params              : {enc_params:,}")
    print(f"  Decoder params (fp32)       : {dec_params:,}")
    print(f"  Quantized encoder weights   : {quantized_params:,}")
    print(f"  Effective bits / qweight    : {eff_bits:.2f}")
    print(f"  Encoder size                : {report['encoder_fp32_mb']} MB"
          f" -> {report['encoder_compressed_mb']} MB ({report['encoder_compression_x']}x)")
    print(f"  Whole model size            : {report['model_fp32_mb']} MB"
          f" -> {report['model_compressed_mb']} MB ({report['model_compression_x']}x)")
    print("  NOTE: only the encoder is quantized, so the whole model shrinks far")
    print("        less than the ~4x of the encoder alone -- the fp32 decoder")
    print("        dominates the remaining size. No flat '4x total' claim.")
    return report


def get_calibration_audio(nsamples, audio_len, seed=42):
    """Load real calibration speech (LibriSpeech validation-clean), truncate only."""
    from datasets import load_dataset, Audio

    print(f"\nLoading {nsamples} calibration clips (LibriSpeech validation-clean)...")
    try:
        ds = load_dataset("openslr/librispeech_asr", "clean", split="validation")
    except Exception:
        ds = load_dataset("librispeech_asr", "clean", split="validation")
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    ds = ds.shuffle(seed=seed)

    samples = []
    for i in range(min(nsamples, len(ds))):
        a = ds[i]["audio"]["array"].astype(np.float32)
        if audio_len and len(a) > audio_len:
            a = a[:audio_len]
        samples.append(torch.tensor(a).unsqueeze(0))
    print(f"  Loaded {len(samples)} clips.")
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Encoder-only SpQR quantization for Moonshine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", type=str, default="usefulsensors/moonshine-base")
    parser.add_argument("--pth_path", type=str, default=None,
                        help="Optional LiteASR .pth checkpoint (encoder LinearLowRank).")
    # SpQR hyperparameters (defaults follow the upstream SpQR W8 recipe).
    parser.add_argument("--wbits", type=int, default=8)
    parser.add_argument("--groupsize", type=int, default=16)
    parser.add_argument("--perchannel", action="store_true", default=True)
    parser.add_argument("--no_perchannel", dest="perchannel", action="store_false")
    parser.add_argument("--sym", action="store_true")
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--qq_scale_bits", type=int, default=3)
    parser.add_argument("--qq_zero_bits", type=int, default=3)
    parser.add_argument("--qq_groupsize", type=int, default=16)
    parser.add_argument("--outlier_threshold", type=float, default=0.2,
                        help="outlier_relative_threshold; use inf to disable outliers.")
    parser.add_argument("--permutation_order", type=str, default="act_order",
                        choices=["identity", "act_order", "spearman"])
    parser.add_argument("--simplified_outliers", action="store_true")
    # dynamic activation quantization (true W8A8)
    parser.add_argument("--quantize_activations", action="store_true",
                        help="Enable dynamic INT8 activation quantization on top "
                             "of the SpQR int8 weights (true W8A8). Without this "
                             "flag the encoder is weight-only quantized (W8A16).")
    parser.add_argument("--act_granularity", type=str, default="per_token",
                        choices=["per_token", "per_tensor"],
                        help="Dynamic activation scale granularity (default per_token).")
    # calibration / eval
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--audio_len", type=int, default=160000)
    parser.add_argument("--max_eval_samples", type=int, default=None,
                        help="Default None = full test sets (2620 clean / 2939 other).")
    parser.add_argument("--max_layers", type=int, default=None,
                        help="Quantize only the first N encoder layers (smoke testing).")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Device: {device}  | eval dtype: {torch_dtype}")
    print(f"SpQR: W{args.wbits} groupsize={args.groupsize} perchannel={args.perchannel} "
          f"qq=({args.qq_scale_bits},{args.qq_zero_bits}) "
          f"outlier_thr={args.outlier_threshold} perm={args.permutation_order}")
    if args.quantize_activations:
        print(f"Activations: dynamic INT8 ({args.act_granularity}) "
              f"-> mode W{args.wbits}A8 (true dynamic W8A8)")
    else:
        print(f"Activations: fp16 (weight-only) -> mode W{args.wbits}A16")

    start = time.time()

    processor = get_processor(args.model)
    model = load_moonshine_model(args.model)
    if args.pth_path:
        model, n_replaced = load_liteasr_pth(model, args.pth_path)
        print(f"  LiteASR LinearLowRank sublayers: {n_replaced}")

    # SpQR quantization config (passed verbatim to SPQRUtil.quantize()).
    qcfg = dict(
        bits=args.wbits,
        groupsize=args.groupsize,
        percdamp=args.percdamp,
        sym=args.sym,
        perchannel=args.perchannel,
        qq_scale_bits=args.qq_scale_bits,
        qq_zero_bits=args.qq_zero_bits,
        qq_groupsize=args.qq_groupsize,
        qq_zero_sym=False,
        outlier_relative_threshold=args.outlier_threshold,
        permutation_order=args.permutation_order,
        simplified_outliers=args.simplified_outliers,
        save_quantization=False,
        verbose=False,
    )

    audio_samples = get_calibration_audio(args.nsamples, args.audio_len, args.seed)

    # Quantize the encoder in float32 for accuracy.
    model = model.to(device).to(torch.float32)
    stats = quantize_encoder_spqr(model, audio_samples, qcfg, device, max_layers=args.max_layers)

    comp = report_compression(model, args, stats["outlier_fraction"], stats["quantized_params"])

    # Optionally add dynamic INT8 activation quantization (true W8A8) on top of
    # the SpQR int8 weights. This wraps each quantized encoder sublayer so that
    # the activation entering every matmul is quantized to int8 at runtime.
    n_wrapped = 0
    if args.quantize_activations:
        from act_quant import wrap_encoder_activations
        n_wrapped = wrap_encoder_activations(
            model, granularity=args.act_granularity, max_layers=args.max_layers)
        print("\n" + "=" * 70)
        print("DYNAMIC ACTIVATION QUANTIZATION (true W8A8)")
        print("=" * 70)
        print(f"  Wrapped {n_wrapped} encoder sublayers with ActQuantWrapper")
        print(f"  Activation scheme : dynamic INT8, {args.act_granularity}, symmetric")
        print(f"  Mode              : W{args.wbits}A8 (dynamic activations, "
              f"{args.act_granularity})")

    mode = (f"W{args.wbits}A8 (dynamic activations, {args.act_granularity})"
            if args.quantize_activations else f"W{args.wbits}A16")

    results = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "pth_path": args.pth_path,
        "method": "encoder-only SpQR (verbatim SPQRUtil: GPTQ + outlier detection)",
        "mode": mode,
        "wbits": args.wbits,
        "activation_quant": bool(args.quantize_activations),
        "act_granularity": args.act_granularity if args.quantize_activations else None,
        "activation_wrapped_sublayers": n_wrapped,
        "groupsize": args.groupsize,
        "outlier_threshold": args.outlier_threshold,
        "permutation_order": args.permutation_order,
        "qq_scale_bits": args.qq_scale_bits,
        "qq_zero_bits": args.qq_zero_bits,
        "nsamples": args.nsamples,
        "compression": comp,
    }

    if not args.skip_eval:
        from eval_utils import evaluate_model
        model = model.to(device).to(torch_dtype).eval()
        print("\n" + "=" * 70)
        print(f"WER EVALUATION ({mode} encoder-quantized model)")
        print("=" * 70)
        wer = evaluate_model(model, processor, device, torch_dtype, args.max_eval_samples)
        results["wer"] = wer

    elapsed = time.time() - start
    results["time_seconds"] = round(elapsed, 1)
    print(f"\n  Total time: {elapsed:.1f}s")

    if args.save:
        print(f"Saving quantized model to: {args.save}")
        torch.save(model.state_dict(), args.save)

    out_path = args.output or f"spqr_encoder_w{args.wbits}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
