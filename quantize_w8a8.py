"""
W8A8 Quantization Pipeline for LiteASR-compressed Moonshine Models.

This script applies W8A8 (8-bit weight, 8-bit activation) quantization with
GPTQ-style calibration to a LiteASR .pth Moonshine checkpoint.

Pipeline:
  1. Load original Moonshine model from HuggingFace
  2. Replace encoder layers with LinearLowRank from .pth (LiteASR compression)
  3. Run GPTQ-calibrated W8A8 quantization on ALL linear layers
  4. Evaluate WER on LibriSpeech test-clean and test-other
  5. Report: params, model size, WER before/after quantization

What W8A8 means:
  - W8: Weights quantized to INT8 per-channel symmetric. Each output channel
    (row of the weight matrix) gets its own scale factor: scale = max(|row|)/127
  - A8: Activations quantized to INT8 at runtime (dynamic per-tensor symmetric).
    At inference time, each activation tensor is scaled to fit [-127, 127].

The GPTQ calibration uses a Hessian-aware approach:
  - Collects H = X^T X / n from calibration data (LibriSpeech dev)
  - Uses inverse Hessian to optimally propagate quantization errors
  - This produces better INT8 rounding than naive min/max quantization

Usage:
    # Basic W8A8 quantization with GPTQ calibration:
    python quantize_w8a8.py \\
        --pth_path /path/to/lite-moonshine-moonshine-base_0.98:0.99.pth \\
        --nsamples 128 \\
        --max_eval_samples 200

    # Quick test with synthetic data (no network required):
    python quantize_w8a8.py \\
        --pth_path /path/to/model.pth \\
        --use_synthetic \\
        --nsamples 16 \\
        --skip_eval

    # Encoder-only quantization:
    python quantize_w8a8.py \\
        --pth_path /path/to/model.pth \\
        --part encoder \\
        --nsamples 64

    # Full quantization with result saving:
    python quantize_w8a8.py \\
        --pth_path /path/to/model.pth \\
        --nsamples 128 \\
        --save quantized_model.pth \\
        --output results.json
"""

import argparse
import json
import time
from datetime import datetime

import torch

from modelutils import (
    load_moonshine_model,
    load_liteasr_pth,
    get_processor,
    count_parameters,
    model_size_bytes,
)
from quant_engine import quantize_encoder_w8a8, quantize_decoder_w8a8
from eval_utils import evaluate_model


def main():
    parser = argparse.ArgumentParser(
        description="W8A8 Quantization for LiteASR Moonshine .pth files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model arguments
    parser.add_argument(
        "--pth_path", type=str, required=True,
        help="Path to the LiteASR .pth checkpoint file"
    )
    parser.add_argument(
        "--model", type=str, default="usefulsensors/moonshine-base",
        help="HuggingFace model name for Moonshine base (default: usefulsensors/moonshine-base)"
    )

    # Quantization arguments
    parser.add_argument(
        "--part", type=str, default="both", choices=["encoder", "decoder", "both"],
        help="Which parts to quantize: encoder, decoder, or both (default: both)"
    )
    parser.add_argument(
        "--blocksize", type=int, default=128,
        help="GPTQ block size for column processing (default: 128)"
    )
    parser.add_argument(
        "--percdamp", type=float, default=0.01,
        help="GPTQ damping factor for Hessian regularization (default: 0.01)"
    )

    # Data arguments
    parser.add_argument(
        "--nsamples", type=int, default=128,
        help="Number of calibration samples from LibriSpeech dev (default: 128)"
    )
    parser.add_argument(
        "--audio_len", type=int, default=160000,
        help="Audio length in samples for calibration (default: 160000 = 10s at 16kHz)"
    )
    parser.add_argument(
        "--use_synthetic", action="store_true",
        help="Use synthetic random audio instead of LibriSpeech for calibration"
    )

    # Evaluation arguments
    parser.add_argument(
        "--skip_eval", action="store_true",
        help="Skip WER evaluation (useful for quick testing)"
    )
    parser.add_argument(
        "--max_eval_samples", type=int, default=None,
        help="Maximum samples per eval split (None for full LibriSpeech test)"
    )

    # Output arguments
    parser.add_argument(
        "--save", type=str, default=None,
        help="Path to save the quantized model state_dict"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save results JSON (default: w8a8_results.json)"
    )

    args = parser.parse_args()

    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"Device: {device} ({gpu_name}, {gpu_mem:.1f} GB)")
    else:
        print(f"Device: {device}")

    print(f"LiteASR checkpoint: {args.pth_path}")
    print(f"Quantization: W8A8 (INT8 weights + INT8 dynamic activations)")
    print(f"GPTQ blocksize: {args.blocksize}, percdamp: {args.percdamp}")
    print(f"Parts to quantize: {args.part}")
    print()

    start_time = time.time()

    # ── Load model ────────────────────────────────────────────────────────────
    processor = get_processor(args.model)
    model = load_moonshine_model(args.model)
    model, n_replaced = load_liteasr_pth(model, args.pth_path)

    # Print pre-quantization stats
    enc_params, dec_params, total_params = count_parameters(model)
    size_bytes = model_size_bytes(model)
    print(f"\nPre-quantization model stats:")
    print(f"  Encoder params:  {enc_params:>12,}")
    print(f"  Decoder params:  {dec_params:>12,}")
    print(f"  Total params:    {total_params:>12,}")
    print(f"  Model size:      {size_bytes/1e6:>12.1f} MB (FP32)")
    print(f"  W8A8 est. size:  {size_bytes/4/1e6:>12.1f} MB (INT8)")
    print(f"  LinearLowRank layers: {n_replaced}")

    # ── Pre-quantization evaluation ──────────────────────────────────────────
    wer_before = None
    if not args.skip_eval:
        print(f"\n{'='*70}")
        print("EVALUATING BEFORE QUANTIZATION")
        print(f"{'='*70}")
        wer_before = evaluate_model(model, processor, device, args.max_eval_samples)
        model = model.cpu()

    # ── Load calibration data ─────────────────────────────────────────────────
    if args.use_synthetic:
        from datautils import get_synthetic_calibration
        audio_samples = get_synthetic_calibration(args.nsamples, args.audio_len)
    else:
        from datautils import get_librispeech_calibration
        audio_samples = get_librispeech_calibration(args.nsamples, args.audio_len)

    # ── Run W8A8 quantization ─────────────────────────────────────────────────
    encoder_outputs = None

    if args.part in ("encoder", "both"):
        encoder_outputs = quantize_encoder_w8a8(
            model, audio_samples, device,
            blocksize=args.blocksize, percdamp=args.percdamp
        )

    if args.part in ("decoder", "both"):
        if encoder_outputs is None:
            # Need encoder outputs for decoder calibration even if not quantizing encoder
            print("\nRunning encoder forward for decoder calibration...")
            encoder_outputs = quantize_encoder_w8a8(
                model, audio_samples, device,
                blocksize=args.blocksize, percdamp=args.percdamp
            )
        quantize_decoder_w8a8(
            model, encoder_outputs, device,
            blocksize=args.blocksize, percdamp=args.percdamp
        )

    elapsed = time.time() - start_time
    print(f"\n  Total quantization time: {elapsed:.1f}s")

    # ── Post-quantization evaluation ──────────────────────────────────────────
    wer_after = None
    if not args.skip_eval:
        print(f"\n{'='*70}")
        print("EVALUATING AFTER W8A8 QUANTIZATION")
        print(f"{'='*70}")
        wer_after = evaluate_model(model, processor, device, args.max_eval_samples)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("FINAL RESULTS: LiteASR + W8A8 Quantization")
    print(f"{'='*70}")
    print(f"  Model:           {args.model}")
    print(f"  LiteASR .pth:    {args.pth_path}")
    print(f"  Quantization:    W8A8 (GPTQ-calibrated)")
    print(f"  Calibration:     {args.nsamples} samples")
    print(f"  Parts quantized: {args.part}")
    print(f"  {'─'*50}")
    print(f"  Parameters:      {total_params:,}")
    print(f"  FP32 size:       {size_bytes/1e6:.1f} MB")
    print(f"  INT8 size (est): {size_bytes/4/1e6:.1f} MB")
    print(f"  Compression:     ~4x")

    if wer_before and wer_after:
        print(f"  {'─'*50}")
        print(f"  {'Metric':<20s} {'Before W8A8':<15s} {'After W8A8':<15s} {'Delta':<10s}")
        print(f"  {'─'*20} {'─'*15} {'─'*15} {'─'*10}")
        for key, label in [("wer_clean", "WER test-clean"), ("wer_other", "WER test-other"), ("wer_avg", "WER average")]:
            before = wer_before[key]
            after = wer_after[key]
            delta = round(after - before, 2)
            sign = "+" if delta >= 0 else ""
            print(f"  {label:<20s} {before:>10}%     {after:>10}%     {sign}{delta}%")

    print(f"  {'─'*50}")
    print(f"  Time elapsed:    {elapsed:.1f}s")
    print(f"{'='*70}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    if args.save:
        print(f"\nSaving quantized model to: {args.save}")
        torch.save(model.state_dict(), args.save)

    output_path = args.output or "w8a8_results.json"
    results = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "pth_path": args.pth_path,
        "quantization": "W8A8",
        "method": "GPTQ-calibrated INT8 per-channel symmetric",
        "blocksize": args.blocksize,
        "percdamp": args.percdamp,
        "nsamples": args.nsamples,
        "part": args.part,
        "total_params": total_params,
        "encoder_params": enc_params,
        "decoder_params": dec_params,
        "liteasr_lowrank_layers": n_replaced,
        "fp32_size_mb": round(size_bytes / 1e6, 1),
        "int8_size_mb": round(size_bytes / 4 / 1e6, 1),
        "compression_ratio": "4x",
        "time_seconds": round(elapsed, 1),
    }
    if wer_before:
        results["wer_before"] = wer_before
    if wer_after:
        results["wer_after"] = wer_after
        if wer_before:
            results["wer_degradation"] = {
                "clean": round(wer_after["wer_clean"] - wer_before["wer_clean"], 2),
                "other": round(wer_after["wer_other"] - wer_before["wer_other"], 2),
                "avg": round(wer_after["wer_avg"] - wer_before["wer_avg"], 2),
            }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
