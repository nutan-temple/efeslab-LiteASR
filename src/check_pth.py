"""
Inspect a compressed .pth checkpoint with CORRECT parameter counting.

Accounts for weight tying (proj_out.weight == embed_tokens.weight).

Usage:
    python src/check_pth.py lite-moonshine-moonshine-base_0.99:0.999.pth
"""
import sys
import torch


def main(pth_path):
    print(f"Loading: {pth_path}\n")
    sd = torch.load(pth_path, map_location="cpu", weights_only=False)

    encoder_params = 0
    decoder_params = 0
    proj_out_params = 0
    compressed = []

    for key, tensor in sorted(sd.items()):
        n = tensor.numel()

        if key.startswith("model.encoder."):
            encoder_params += n
            marker = ""
            if "weight1" in key:
                rank = tensor.shape[1]
                compressed.append((key.replace(".weight1", ""), rank, tuple(tensor.shape)))
                marker = f"  *** RANK={rank} ***"
            elif "weight2" in key:
                marker = "  *** LOW-RANK ***"
            print(f"  [ENC] {key:65s} {str(tuple(tensor.shape)):20s} {n:>10,}{marker}")

        elif key.startswith("model.decoder."):
            decoder_params += n

        elif key.startswith("proj_out."):
            proj_out_params += n

    # proj_out is TIED with embed_tokens - same tensor, stored twice in state_dict
    # Real unique params = encoder + decoder (already includes embed_tokens) + 0 for proj_out
    unique_total = encoder_params + decoder_params
    state_dict_total = encoder_params + decoder_params + proj_out_params

    # Original Moonshine Base numbers (from model.parameters()):
    # Encoder: 20,153,120 | Decoder: 41,360,800 | Total unique: 61,513,920
    ORIG_ENCODER = 20_153_120
    ORIG_DECODER = 41_360_800
    ORIG_TOTAL = 61_513_920

    print(f"\n{'='*60}")
    print("PARAMETER COUNTS (accounting for weight tying)")
    print(f"{'='*60}")
    print(f"  Encoder (compressed):     {encoder_params:>12,}")
    print(f"  Decoder (unchanged):      {decoder_params:>12,}")
    print(f"  proj_out (TIED, not counted): {proj_out_params:>8,} (duplicate of embed_tokens)")
    print(f"  ─────────────────────────────────────────")
    print(f"  Unique total:             {unique_total:>12,}")
    print(f"  State dict total:         {state_dict_total:>12,} (includes tied duplicate)")

    print(f"\n{'='*60}")
    print("COMPARISON TO ORIGINAL MOONSHINE BASE")
    print(f"{'='*60}")
    print(f"  Original encoder:         {ORIG_ENCODER:>12,}")
    print(f"  Compressed encoder:       {encoder_params:>12,}")
    enc_reduction = 100 * (1 - encoder_params / ORIG_ENCODER)
    print(f"  Encoder reduction:        {enc_reduction:>11.1f}%")
    print()
    print(f"  Original decoder:         {ORIG_DECODER:>12,}")
    print(f"  Your decoder:             {decoder_params:>12,}")
    print(f"  Decoder changed:          {'NO' if decoder_params == ORIG_DECODER else 'YES (unexpected!)'}")
    print()
    print(f"  Original total (unique):  {ORIG_TOTAL:>12,}")
    print(f"  Your total (unique):      {unique_total:>12,}")
    total_reduction = 100 * (1 - unique_total / ORIG_TOTAL)
    print(f"  Total reduction:          {total_reduction:>11.1f}%")

    print(f"\n{'='*60}")
    print(f"COMPRESSED LAYERS ({len(compressed)})")
    print(f"{'='*60}")
    if not compressed:
        print("  *** NONE - no compression happened! ***")
    else:
        print(f"  {'Layer':<50s} {'Rank':<6} {'Original':<10} {'Savings'}")
        print(f"  {'-'*50} {'-'*6} {'-'*10} {'-'*10}")
        for path, rank, shape in compressed:
            short = path.replace("model.encoder.layers.", "L").replace("self_attn.", "").replace("mlp.", "")
            in_feat = shape[0]  # 416
            # Original size for this layer
            if "fc1" in path:
                orig_size = 416 * 1664  # fc1: 416 -> 1664
                compressed_size = 416 * rank + rank * 1664
            elif "fc2" in path:
                orig_size = 1664 * 416  # fc2: 1664 -> 416
                compressed_size = 1664 * rank + rank * 416
            else:
                orig_size = 416 * 416  # attention: 416 -> 416
                compressed_size = 416 * rank + rank * 416
            saving = 100 * (1 - compressed_size / orig_size)
            print(f"  {short:<50s} {rank:<6} {orig_size:>10,} {saving:>8.1f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/check_pth.py <path_to_pth_file>")
        sys.exit(1)
    main(sys.argv[1])
