"""
Inspect a compressed .pth checkpoint to verify compression actually happened.

Usage:
    python src/check_pth.py lite-moonshine-moonshine-base_0.99:0.999.pth
"""

import sys
import torch


def main(pth_path):
    print(f"Loading: {pth_path}\n")
    state_dict = torch.load(pth_path, map_location="cpu", weights_only=False)

    if not isinstance(state_dict, dict):
        if hasattr(state_dict, "state_dict"):
            state_dict = state_dict.state_dict()
        else:
            print(f"ERROR: Unexpected type: {type(state_dict)}")
            return

    # Count parameters - use STRICT prefix matching
    total_params = 0
    encoder_params = 0
    decoder_params = 0
    other_params = 0

    # Track which encoder layers are compressed
    compressed_layers = {}

    encoder_keys = []
    decoder_keys = []
    other_keys = []

    for key, tensor in sorted(state_dict.items()):
        numel = tensor.numel()
        total_params += numel

        # Strict classification: encoder vs decoder vs other
        # "model.encoder." = encoder weights
        # "model.decoder." or "proj_out." = decoder weights
        # "encoder_attn" is part of DECODER (cross-attention lives in decoder)
        if key.startswith("model.encoder.") or key.startswith("encoder."):
            encoder_params += numel
            encoder_keys.append((key, tensor.shape, numel))
        elif key.startswith("model.decoder.") or key.startswith("decoder.") or key.startswith("proj_out"):
            decoder_params += numel
            decoder_keys.append((key, tensor.shape, numel))
        else:
            other_params += numel
            other_keys.append((key, tensor.shape, numel))

        # Detect low-rank layers (only in encoder)
        if "weight1" in key and ("model.encoder." in key or key.startswith("encoder.")):
            rank = tensor.shape[1] if tensor.dim() == 2 else tensor.shape[0]
            compressed_layers[key.replace(".weight1", "")] = {
                "rank": rank,
                "shape": tuple(tensor.shape),
            }

    # Print encoder keys
    print("=" * 70)
    print("ENCODER KEYS:")
    print("=" * 70)
    for key, shape, numel in encoder_keys:
        marker = " *** LOW-RANK ***" if "weight1" in key or "weight2" in key else ""
        print(f"  {key:60s} {str(shape):20s} {numel:>10,}{marker}")

    # Print decoder keys (summary)
    print(f"\n{'=' * 70}")
    print(f"DECODER KEYS ({len(decoder_keys)} total, showing first 10):")
    print("=" * 70)
    for key, shape, numel in decoder_keys[:10]:
        print(f"  {key:60s} {str(shape):20s} {numel:>10,}")
    if len(decoder_keys) > 10:
        print(f"  ... and {len(decoder_keys) - 10} more decoder keys")

    # Print other keys
    if other_keys:
        print(f"\n{'=' * 70}")
        print(f"OTHER KEYS ({len(other_keys)}):")
        print("=" * 70)
        for key, shape, numel in other_keys:
            print(f"  {key:60s} {str(shape):20s} {numel:>10,}")

    # Print summary
    print(f"\n{'=' * 70}")
    print("PARAMETER SUMMARY:")
    print("=" * 70)
    print(f"  Encoder parameters:  {encoder_params:>12,}")
    print(f"  Decoder parameters:  {decoder_params:>12,}")
    print(f"  Other parameters:    {other_params:>12,}")
    print(f"  TOTAL parameters:    {total_params:>12,}")

    # Print compression info
    print(f"\n{'=' * 70}")
    print("COMPRESSION ANALYSIS:")
    print("=" * 70)

    if not compressed_layers:
        print("\n  *** WARNING: NO COMPRESSED LAYERS FOUND! ***")
        print("  The checkpoint has NO weight1/weight2 keys in the encoder.")
        print("  This means compression DID NOT HAPPEN.")
    else:
        print(f"\n  Found {len(compressed_layers)} compressed encoder layers:\n")

        # Group by layer
        layers_summary = {}
        for layer_path, info in sorted(compressed_layers.items()):
            # Extract layer number
            parts = layer_path.split("layers.")
            if len(parts) > 1:
                layer_num = parts[1].split(".")[0]
                if layer_num not in layers_summary:
                    layers_summary[layer_num] = {}
                # Extract component name
                comp = layer_path.split(f"layers.{layer_num}.")[-1]
                layers_summary[layer_num][comp] = info["rank"]

        print(f"  {'Layer':<8} {'q_proj':<8} {'k_proj':<8} {'v_proj':<8} {'o_proj':<8} {'fc1':<8} {'fc2':<8}")
        print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for layer_num in sorted(layers_summary.keys(), key=int):
            comps = layers_summary[layer_num]
            q = comps.get("self_attn.q_proj", "-")
            k = comps.get("self_attn.k_proj", "-")
            v = comps.get("self_attn.v_proj", "-")
            o = comps.get("self_attn.o_proj", "-")
            fc1 = comps.get("mlp.fc1", "-")
            fc2 = comps.get("mlp.fc2", "-")
            print(f"  {layer_num:<8} {str(q):<8} {str(k):<8} {str(v):<8} {str(o):<8} {str(fc1):<8} {str(fc2):<8}")

        print(f"\n  Original dim: 416 (attention) / 1664 (fc1 out) / 416 (fc2 out)")
        print(f"  '-' means layer was NOT compressed (rank too high, skipped)")

    # Compare to original Moonshine Base sizes
    # Original Moonshine Base encoder (from HF): conv frontend + 8 transformer layers + layer_norm
    # Conv: conv1(52832) + conv2(2422784+832) + conv3(1038336+416) + groupnorm(832) + layer_norm(416) = ~3,516,448
    # Per transformer layer (uncompressed): 
    #   self_attn: q(173056) + k(173056) + v(173056) + o(173056) = 692,224
    #   mlp: fc1(416*1664 + 1664 = 693888) + fc2(1664*416 + 416 = 692640) = 1,386,528
    #   norms: 2 * 416 = 832
    #   Total per layer: ~2,079,584
    # 8 layers: 16,636,672
    # Total encoder: ~20,153,120
    original_encoder = 20_153_120
    original_decoder = 29_400_000  # approximate (includes encoder_attn cross-attention)
    
    print(f"\n{'=' * 70}")
    print("COMPARISON TO ORIGINAL MOONSHINE BASE:")
    print("=" * 70)
    print(f"  Original encoder (uncompressed):  {original_encoder:>12,}")
    print(f"  Your encoder (from .pth):         {encoder_params:>12,}")
    if encoder_params < original_encoder:
        print(f"  Encoder reduction:                {100*(1 - encoder_params/original_encoder):>11.1f}%")
    else:
        print(f"  Encoder INCREASE:                 {100*(encoder_params/original_encoder - 1):>11.1f}% (bias terms added by compression)")
    print()
    print(f"  Original decoder:                ~{original_decoder:>12,}")
    print(f"  Your decoder (from .pth):         {decoder_params:>12,}")
    print()
    print(f"  Note: Decoder in .pth should be IDENTICAL to original (not compressed)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/check_pth.py <path_to_pth_file>")
        sys.exit(1)
    main(sys.argv[1])
