"""
Entry point for quantizing Moonshine models (encoder-decoder ASR) using SpQR.

This script handles the encoder and decoder separately:
- Encoder: processes raw audio waveforms through the conv frontend, then quantizes encoder transformer layers.
- Decoder: uses encoder outputs + BOS-token embeddings to quantize decoder transformer layers.

Usage:
    python3 main_moonshine.py usefulsensors/moonshine-base audio_random --wbits 4 --groupsize 16 --nsamples 2 --perchannel
"""

import os
import time

import torch
import torch.nn as nn
from tqdm import trange

from datautils import get_loaders
from modelutils import (
    find_sublayers,
    get_decoder_layers,
    get_decoder_sequential_groups,
    get_layers,
    get_model,
    get_sequential_groups,
)
from spqr_engine import Quantizer, SPQRUtil, quantize


@torch.no_grad()
def run_conv_frontend(model, audio):
    """
    Run the Moonshine encoder conv frontend on raw audio waveform.

    Args:
        model: MoonshineForConditionalGeneration model
        audio: tensor of shape (1, audio_len)

    Returns:
        hidden_states: tensor of shape (1, seq_len, hidden_size)
    """
    # TODO: This manually reimplements the encoder's pre-transformer computation.
    # If the HuggingFace MoonshineEncoder class changes its conv/activation sequence,
    # this will silently diverge. Consider using the model's own forward method for the
    # encoder prefix if a standalone encoder-conv forward becomes available.
    x = audio.unsqueeze(1)  # (1, 1, audio_len)
    x = torch.nn.functional.tanh(model.model.encoder.conv1(x))
    x = model.model.encoder.groupnorm(x)
    x = torch.nn.functional.gelu(model.model.encoder.conv2(x))
    x = torch.nn.functional.gelu(model.model.encoder.conv3(x))
    x = x.permute(0, 2, 1)  # (1, seq_len, hidden_size)
    return x


@torch.no_grad()
def get_encoder_inps(model, data_iterable, dev, nsamples):
    """
    Get inputs to the first encoder transformer layer by running audio through the conv frontend.

    Args:
        model: MoonshineForConditionalGeneration model
        data_iterable: list of audio tensors, each of shape (1, audio_len)
        dev: device string
        nsamples: number of samples to process

    Returns:
        inps: tensor of shape (nsamples, seq_len, hidden_size) - all padded to same seq_len
        forward_args: dict with position_embeddings for encoder layers
    """
    print("Running conv frontend to get encoder layer inputs...", flush=True)

    # Process all audio samples through conv frontend
    hidden_states_list = []
    for i in range(min(nsamples, len(data_iterable))):
        audio = data_iterable[i].to(dev)
        hs = run_conv_frontend(model, audio)
        hidden_states_list.append(hs)

    # All audio is padded to same length, so seq_lens should be identical
    # But handle variable lengths just in case by padding to max
    max_seq_len = max(hs.shape[1] for hs in hidden_states_list)
    hidden_size = hidden_states_list[0].shape[2]
    dtype = hidden_states_list[0].dtype

    inps = torch.zeros((nsamples, max_seq_len, hidden_size), dtype=dtype, device=dev)
    for i, hs in enumerate(hidden_states_list):
        inps[i, : hs.shape[1], :] = hs[0]

    # Compute position embeddings for encoder layers
    position_ids = torch.arange(max_seq_len, device=dev).unsqueeze(0)
    position_embeddings = model.model.encoder.rotary_emb(inps[:1], position_ids=position_ids)

    forward_args = {
        "position_embeddings": position_embeddings,
    }

    return inps, forward_args


@torch.no_grad()
def quantize_encoder(model, dataloader, args, device):
    """
    Quantize encoder transformer layers using SpQR.

    Args:
        model: MoonshineForConditionalGeneration model
        dataloader: list of audio tensors
        args: argparse namespace with quantization params
        device: device string

    Returns:
        encoder_outputs: tensor of shape (nsamples, seq_len, hidden_size) - outputs of quantized encoder
    """
    print("\n============ Quantizing Encoder ============")

    inps, forward_args = get_encoder_inps(
        model, dataloader, dev=device, nsamples=args.nsamples
    )
    outs = torch.zeros_like(inps)

    layers = get_layers(model)
    print(f"Encoder has {len(layers)} layers")

    for i in range(len(layers)):
        print(f"\n--- Encoder Layer {i} of {len(layers)} ---")
        start_time = time.time()

        layer_dev_original = next(layers[i].parameters()).device
        if layer_dev_original.type != "cuda":
            layer = layers[i].to(device)
        else:
            layer = layers[i]
        layer_dev = next(layers[i].parameters()).device

        all_sublayers = find_sublayers(layer)

        # Move forward_args tensors to layer device
        layer_forward_args = {}
        for k, v in forward_args.items():
            if isinstance(v, torch.Tensor):
                layer_forward_args[k] = v.to(layer_dev)
            elif isinstance(v, tuple):
                layer_forward_args[k] = tuple(t.to(layer_dev) if isinstance(t, torch.Tensor) else t for t in v)
            else:
                layer_forward_args[k] = v

        if args.true_sequential:
            sequential = get_sequential_groups(model)
        else:
            sequential = [list(all_sublayers.keys())]

        for names in sequential:
            subset = {n: all_sublayers[n] for n in names if n in all_sublayers}
            if not subset:
                continue

            spqr_handlers = {}
            for sublayer_name in subset:
                spqr_handlers[sublayer_name] = SPQRUtil(subset[sublayer_name])

            def add_batch(name):
                def tmp(_, inp, out):
                    spqr_handlers[name].add_batch(inp[0].data)
                return tmp

            handles = []
            for sublayer_name in subset:
                handles.append(subset[sublayer_name].register_forward_hook(add_batch(sublayer_name)))

            for j in range(args.nsamples):
                layer_out = layer(inps[j].unsqueeze(0).to(layer_dev), **layer_forward_args)
                outs[j] = layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out
            for h in handles:
                h.remove()

            for sublayer_name in subset:
                print(f"  Quantizing {sublayer_name}")
                quantized = spqr_handlers[sublayer_name].quantize(
                    percdamp=args.percdamp,
                    bits=args.wbits,
                    groupsize=args.groupsize,
                    sym=args.sym,
                    perchannel=args.perchannel,
                    qq_groupsize=args.qq_groupsize,
                    round_zero=args.round_zero,
                    qq_scale_bits=args.qq_scale_bits,
                    qq_zero_bits=args.qq_zero_bits,
                    qq_zero_sym=args.qq_zero_sym,
                    outlier_relative_threshold=args.outlier_threshold,
                    permutation_order=args.permutation_order,
                    simplified_outliers=args.simplified_outliers,
                    save_quantization=False,
                )
                spqr_handlers[sublayer_name].layer.weight.data = quantized.weight.to(
                    spqr_handlers[sublayer_name].layer.weight.data.dtype
                )

        # Recompute outputs after quantization
        for j in range(args.nsamples):
            layer_out = layer(inps[j].unsqueeze(0).to(layer_dev), **layer_forward_args)
            outs[j] = layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out

        layers[i] = layer.to(layer_dev_original)
        del layer, spqr_handlers

        inps, outs = outs, inps

        print(f"  Layer {i} done in {time.time() - start_time:.1f}s")

    # After all encoder layers, apply layer_norm to get final encoder outputs
    encoder_outputs = torch.zeros_like(inps)
    layer_norm = model.model.encoder.layer_norm.to(device)
    for j in range(args.nsamples):
        encoder_outputs[j] = layer_norm(inps[j].unsqueeze(0).to(device))
    model.model.encoder.layer_norm.to("cpu")

    return encoder_outputs


@torch.no_grad()
def get_decoder_inps(model, encoder_outputs, dev, nsamples):
    """
    Get inputs to the first decoder transformer layer.

    Uses BOS token embedding as decoder input and provides encoder_hidden_states
    from the quantized encoder.

    Args:
        model: MoonshineForConditionalGeneration model
        encoder_outputs: tensor of shape (nsamples, enc_seq_len, hidden_size)
        dev: device string
        nsamples: number of samples

    Returns:
        inps: tensor of shape (nsamples, dec_seq_len, hidden_size)
        forward_args: dict with encoder_hidden_states, position_embeddings, encoder_position_embeddings
    """
    print("Preparing decoder layer inputs...", flush=True)

    # Use BOS token as decoder input (token id 1 is typical, but check config)
    bos_token_id = getattr(model.config, "decoder_start_token_id", 1) or 1
    vocab_size = model.config.vocab_size

    # Generate varied short token sequences per sample so that self-attention
    # projections see diverse activations (avoiding a degenerate rank-1 Hessian).
    dec_seq_len = 4
    embed_tokens = model.model.decoder.embed_tokens.to(dev)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)

    hidden_size = embed_tokens.weight.shape[1]
    dtype = embed_tokens.weight.dtype

    inps = torch.zeros((nsamples, dec_seq_len, hidden_size), dtype=dtype, device=dev)
    for i in range(nsamples):
        # BOS token followed by random tokens from the vocabulary
        random_tokens = torch.randint(0, vocab_size, (1, dec_seq_len - 1), generator=generator)
        dec_input_ids = torch.cat(
            [torch.full((1, 1), bos_token_id, dtype=torch.long), random_tokens], dim=1
        ).to(dev)
        dec_emb = embed_tokens(dec_input_ids)  # (1, dec_seq_len, hidden_size)
        inps[i] = dec_emb[0]

    model.model.decoder.embed_tokens.to("cpu")

    # Decoder position embeddings
    dec_position_ids = torch.arange(dec_seq_len, device=dev).unsqueeze(0)
    position_embeddings = model.model.decoder.rotary_emb(inps[:1], position_ids=dec_position_ids)

    # Encoder position embeddings (needed for cross-attention rotary)
    enc_seq_len = encoder_outputs.shape[1]
    enc_position_ids = torch.arange(enc_seq_len, device=dev).unsqueeze(0)
    encoder_position_embeddings = model.model.encoder.rotary_emb(
        encoder_outputs[:1].to(dev), position_ids=enc_position_ids
    )

    forward_args = {
        "encoder_hidden_states": encoder_outputs,
        "position_embeddings": position_embeddings,
        "encoder_position_embeddings": encoder_position_embeddings,
    }

    return inps, forward_args


@torch.no_grad()
def quantize_decoder(model, encoder_outputs, args, device):
    """
    Quantize decoder transformer layers using SpQR.

    Args:
        model: MoonshineForConditionalGeneration model
        encoder_outputs: tensor of shape (nsamples, enc_seq_len, hidden_size) from quantized encoder
        args: argparse namespace with quantization params
        device: device string
    """
    print("\n============ Quantizing Decoder ============")

    inps, forward_args = get_decoder_inps(
        model, encoder_outputs, dev=device, nsamples=args.nsamples
    )
    outs = torch.zeros_like(inps)

    layers = get_decoder_layers(model)
    print(f"Decoder has {len(layers)} layers")

    for i in range(len(layers)):
        print(f"\n--- Decoder Layer {i} of {len(layers)} ---")
        start_time = time.time()

        layer_dev_original = next(layers[i].parameters()).device
        if layer_dev_original.type != "cuda":
            layer = layers[i].to(device)
        else:
            layer = layers[i]
        layer_dev = next(layers[i].parameters()).device

        all_sublayers = find_sublayers(layer)

        # Move forward_args tensors to layer device
        layer_forward_args = {}
        for k, v in forward_args.items():
            if isinstance(v, torch.Tensor):
                layer_forward_args[k] = v.to(layer_dev)
            elif isinstance(v, tuple):
                layer_forward_args[k] = tuple(t.to(layer_dev) if isinstance(t, torch.Tensor) else t for t in v)
            else:
                layer_forward_args[k] = v

        if args.true_sequential:
            sequential = get_decoder_sequential_groups(model)
        else:
            sequential = [list(all_sublayers.keys())]

        for names in sequential:
            subset = {n: all_sublayers[n] for n in names if n in all_sublayers}
            if not subset:
                continue

            spqr_handlers = {}
            for sublayer_name in subset:
                spqr_handlers[sublayer_name] = SPQRUtil(subset[sublayer_name])

            def add_batch(name):
                def tmp(_, inp, out):
                    spqr_handlers[name].add_batch(inp[0].data)
                return tmp

            handles = []
            for sublayer_name in subset:
                handles.append(subset[sublayer_name].register_forward_hook(add_batch(sublayer_name)))

            for j in range(args.nsamples):
                # For decoder, encoder_hidden_states is per-sample
                sample_forward_args = dict(layer_forward_args)
                if "encoder_hidden_states" in sample_forward_args:
                    sample_forward_args["encoder_hidden_states"] = (
                        forward_args["encoder_hidden_states"][j].unsqueeze(0).to(layer_dev)
                    )
                layer_out = layer(inps[j].unsqueeze(0).to(layer_dev), **sample_forward_args)
                outs[j] = layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out

            for h in handles:
                h.remove()

            for sublayer_name in subset:
                print(f"  Quantizing {sublayer_name}")
                quantized = spqr_handlers[sublayer_name].quantize(
                    percdamp=args.percdamp,
                    bits=args.wbits,
                    groupsize=args.groupsize,
                    sym=args.sym,
                    perchannel=args.perchannel,
                    qq_groupsize=args.qq_groupsize,
                    round_zero=args.round_zero,
                    qq_scale_bits=args.qq_scale_bits,
                    qq_zero_bits=args.qq_zero_bits,
                    qq_zero_sym=args.qq_zero_sym,
                    outlier_relative_threshold=args.outlier_threshold,
                    permutation_order=args.permutation_order,
                    simplified_outliers=args.simplified_outliers,
                    save_quantization=False,
                )
                spqr_handlers[sublayer_name].layer.weight.data = quantized.weight.to(
                    spqr_handlers[sublayer_name].layer.weight.data.dtype
                )

        # Recompute outputs after quantization
        for j in range(args.nsamples):
            sample_forward_args = dict(layer_forward_args)
            if "encoder_hidden_states" in sample_forward_args:
                sample_forward_args["encoder_hidden_states"] = (
                    forward_args["encoder_hidden_states"][j].unsqueeze(0).to(layer_dev)
                )
            layer_out = layer(inps[j].unsqueeze(0).to(layer_dev), **sample_forward_args)
            outs[j] = layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out

        layers[i] = layer.to(layer_dev_original)
        del layer, spqr_handlers

        inps, outs = outs, inps

        print(f"  Layer {i} done in {time.time() - start_time:.1f}s")

    print("\nDecoder quantization complete.")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="SpQR quantization for Moonshine encoder-decoder models"
    )

    parser.add_argument(
        "model_path",
        type=str,
        help="Path or HuggingFace model name for Moonshine model",
    )
    parser.add_argument(
        "dataset",
        type=str,
        default="audio_random",
        help="Dataset name. Use 'audio_random' for synthetic audio calibration data.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration samples.")
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--wbits",
        type=int,
        default=4,
        help="Number of bits for quantization.",
    )
    parser.add_argument(
        "--groupsize",
        type=int,
        default=16,
        help="Quantization group size.",
    )
    parser.add_argument(
        "--permutation_order",
        type=str,
        default="identity",
        help="Weights permutation order; options: identity(default), spearman, act_order",
    )
    parser.add_argument(
        "--true-sequential",
        action="store_true",
        help="Whether to run in true sequential mode.",
    )
    parser.add_argument("--sym", action="store_true", help="Symmetric quantization")
    parser.add_argument(
        "--perchannel",
        action="store_true",
        help="Fit a unique quantizer to each output dim.",
    )
    parser.add_argument(
        "--qq_scale_bits",
        type=int,
        default=None,
        help="Quantize quantization scale with this many bits.",
    )
    parser.add_argument(
        "--round_zero",
        type=int,
        default=None,
        help="Whether to allow non-integer zero when quantizing weights.",
    )
    parser.add_argument(
        "--qq_zero_bits",
        type=int,
        default=None,
        help="Quantize quantization zero with this many bits.",
    )
    parser.add_argument(
        "--qq_zero_sym",
        action="store_true",
        help="Enable sym=True in meta-quantization for groupwise zero.",
    )
    parser.add_argument(
        "--qq_groupsize",
        type=int,
        default=16,
        help="Quantize quantization scale in groups of this many scales.",
    )
    parser.add_argument(
        "--outlier_threshold",
        type=float,
        default=float("inf"),
        help="Relative threshold for outliers.",
    )
    parser.add_argument(
        "--simplified_outliers",
        action="store_true",
        help="Do not perform leave-one-out evaluation when detecting outliers.",
    )
    parser.add_argument(
        "--skip_out_loss",
        action="store_true",
        help="Whether to skip computation of output loss.",
    )
    parser.add_argument(
        "--part",
        type=str,
        default="both",
        choices=["encoder", "decoder", "both"],
        help="Which part to quantize: encoder, decoder, or both.",
    )
    parser.add_argument(
        "--audio_len",
        type=int,
        default=160000,
        help="Audio length in samples (default 160000 = 10s at 16kHz).",
    )
    parser.add_argument("--save", type=str, default=None, help="Path to save quantized model.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32"],
        help="dtype to load the model.",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print("\n============ Loading model ============")
    model = get_model(args.model_path, None, args.dtype).train(False)

    # Load calibration data
    print("\n============ Loading calibration data ============")
    dataloader = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=args.audio_len,
        model_path=args.model_path,
    )

    tick = time.time()

    encoder_outputs = None
    if args.part in ("encoder", "both"):
        encoder_outputs = quantize_encoder(model, dataloader, args, device)

    if args.part in ("decoder", "both"):
        if encoder_outputs is None:
            # If only quantizing decoder, still need encoder outputs
            print("Running encoder forward pass to get encoder outputs for decoder...")
            inps, _ = get_encoder_inps(model, dataloader, dev=device, nsamples=args.nsamples)
            layers = get_layers(model)
            # Run through all encoder layers
            position_ids = torch.arange(inps.shape[1], device=device).unsqueeze(0)
            position_embeddings = model.model.encoder.rotary_emb(inps[:1], position_ids=position_ids)
            fwd_args = {"position_embeddings": position_embeddings}
            for i in range(len(layers)):
                layer = layers[i].to(device)
                layer_fwd_args = {}
                for k, v in fwd_args.items():
                    if isinstance(v, tuple):
                        layer_fwd_args[k] = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in v)
                    elif isinstance(v, torch.Tensor):
                        layer_fwd_args[k] = v.to(device)
                    else:
                        layer_fwd_args[k] = v
                for j in range(args.nsamples):
                    layer_out = layer(inps[j].unsqueeze(0).to(device), **layer_fwd_args)
                    inps[j] = layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out
                layers[i] = layer.cpu()
            # Apply layer norm
            layer_norm = model.model.encoder.layer_norm.to(device)
            encoder_outputs = torch.zeros_like(inps)
            for j in range(args.nsamples):
                encoder_outputs[j] = layer_norm(inps[j].unsqueeze(0).to(device))
            model.model.encoder.layer_norm.to("cpu")

        quantize_decoder(model, encoder_outputs, args, device)

    print(f"\nTotal quantization time: {time.time() - tick:.1f}s")

    # Evaluation placeholder
    print("\n============ Evaluation ============")
    print("Evaluation skipped - WER evaluation requires a proper test set and decoder generation pipeline.")

    if args.save:
        # TODO: The save format here (raw state_dict) is incompatible with load_quantized_model()
        # in modelutils.py, which expects per-layer quantization dictionaries. Additionally,
        # load_quantized_model only restores encoder layers for moonshine (via get_layers), so
        # decoder weights would never be restored. To support saving/loading quantized models,
        # either match the per-sublayer save format from main.py or implement a corresponding
        # load path that handles both encoder and decoder layers.
        print(f"\nSaving quantized model to {args.save}")
        os.makedirs(args.save, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.save, "model.pt"))
        torch.save(vars(args), os.path.join(args.save, "args.pt"))
        print("Save complete.")

    print("\nDone.")


if __name__ == "__main__":
    main()
