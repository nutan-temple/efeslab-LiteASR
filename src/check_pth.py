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

    # Count parameters
    total_params = 0
    encoder_params = 0
    decoder_params = 0
    other_params = 0

    # Track which encoder layers are compressed
    compressed_layers = {}  # "layer.component" -> rank
    normal_layers = []

    print("=" * 70)
    print("ALL KEYS IN CHECKPOINT:")
    print("=" * 70)

    encoder_keys = []
    decoder_keys = []

    for key, tensor in sorted(state_dict.items()):
        numel = tensor.numel()
        total_params += numel

        if "encoder" in key:
            encoder_params += numel
            encoder_keys.append((key, tensor.shape, numel))
        elif "decoder" in key or "embed_tokens" in key:
            decoder_params += numel
            decoder_keys.append((key, tensor.shape, numel))
        else:
            other_params += numel

        # Detect low-rank layers
        if "weight1" in key and "encoder" in key:
            # Extract layer number and component name
            rank = tensor.shape[1] if tensor.dim() == 2 else tensor.shape[0]
            compressed_layers[key.replace(".weight1", "")] = {
                "rank": rank,
                "shape": tuple(tensor.shape),
            }

    # Print encoder keys
    print("\n" + "=" * 70)
    print("ENCODER KEYS:")
    print("=" * 70)
    for key, shape, numel in encoder_keys:
        marker = " *** LOW-RANK ***" if "weight1" in key or "weight2" in key else ""
        print(f"  {key:60s} {str(shape):20s} {numel:>10,}{marker}")

    # Print summary
    print("\n" + "=" * 70)
    print("PARAMETER SUMMARY:")
    print("=" * 70)
    print(f"  Encoder parameters:  {encoder_params:>12,}")
    print(f"  Decoder parameters:  {decoder_params:>12,}")
    print(f"  Other parameters:    {other_params:>12,}")
    print(f"  TOTAL parameters:    {total_params:>12,}")

    # Print compression info
    print("\n" + "=" * 70)
    print("COMPRESSION ANALYSIS:")
    print("=" * 70)

    if not compressed_layers:
        print("\n  *** WARNING: NO COMPRESSED LAYERS FOUND! ***")
        print("  The checkpoint has NO weight1/weight2 keys.")
        print("  This means compression DID NOT HAPPEN or was saved differently.")
        print("\n  Looking for standard weight keys in encoder:")
        for key, shape, numel in encoder_keys:
            if "weight" in key and "norm" not in key and "conv" not in key:
                print(f"    {key}: {shape}")
    else:
        print(f"\n  Found {len(compressed_layers)} compressed layers:\n")
        for layer_path, info in sorted(compressed_layers.items()):
            print(f"    {layer_path}")
            print(f"      rank = {info['rank']}, weight1 shape = {info['shape']}")

    # Compare to original Moonshine Base sizes
    print("\n" + "=" * 70)
    print("COMPARISON TO ORIGINAL MOONSHINE BASE:")
    print("=" * 70)
    original_encoder = 29_000_000  # approximate
    original_decoder = 29_000_000  # approximate
    original_total = 58_000_000

    print(f"  Original encoder:  ~{original_encoder:>12,}")
    print(f"  Your encoder:       {encoder_params:>12,}")
    print(f"  Encoder reduction:  {100*(1 - encoder_params/original_encoder):>11.1f}%")
    print()
    print(f"  Original decoder:  ~{original_decoder:>12,}")
    print(f"  Your decoder:       {decoder_params:>12,}")
    print()
    print(f"  Original total:    ~{original_total:>12,}")
    print(f"  Your total:         {total_params:>12,}")
    print(f"  Total reduction:    {100*(1 - total_params/original_total):>11.1f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/check_pth.py <path_to_pth_file>")
        sys.exit(1)
    main(sys.argv[1])
