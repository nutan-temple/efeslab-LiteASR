import argparse
import numpy as np
import torch
import torch.nn as nn
import evaluate
from normalizer import data_utils
import tqdm
from transformers import MoonshineForConditionalGeneration, AutoTokenizer, AutoFeatureExtractor
from datasets import load_dataset, concatenate_datasets

wer_metric = evaluate.load("wer")
torch.set_float32_matmul_precision('high')


class LinearLowRank(torch.nn.Module):
    def __init__(self, weight1: torch.Tensor, weight2: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.weight1 = nn.Parameter(weight1)
        self.weight2 = nn.Parameter(weight2)
        self.bias = nn.Parameter(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.weight1) @ self.weight2 + self.bias


def transcribe(model, tokenizer, audio, feature_extractor):
    """Transcribe audio using Moonshine model.generate()."""
    inputs = feature_extractor(
        audio, sampling_rate=16000, return_tensors="pt"
    )
    input_values = inputs["input_values"].to(model.device)
    generated_ids = model.generate(input_values=input_values)
    text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text


def apply_low_rank(model, dataset, rank_threshold, tokenizer, feature_extractor, device):
    """Apply low-rank approximation to Moonshine encoder layers using PCA/SVD.

    Uses forward hooks to collect activations from encoder linear layers during
    calibration, then decomposes each layer using SVD and replaces with LinearLowRank.
    """
    base_dim = model.config.hidden_size  # 416 for moonshine-base

    # Storage for collected activations
    # calibration_data[layer_idx][component_name] = list of tensors
    num_layers = model.config.num_hidden_layers
    calibration_data = {
        i: {"q_proj": [], "k_proj": [], "v_proj": [], "o_proj": [], "fc1": [], "fc2": []}
        for i in range(num_layers)
    }

    # Register forward hooks on encoder linear layers to capture activations
    hooks = []
    encoder_layers = model.model.encoder.layers

    for i_layer in range(num_layers):
        layer = encoder_layers[i_layer]

        def make_hook(layer_idx, component_name):
            def hook_fn(module, input, output):
                # Capture output activation (detached, moved to cpu to save memory)
                calibration_data[layer_idx][component_name].append(
                    output.detach().cpu()
                )
            return hook_fn

        hooks.append(layer.self_attn.q_proj.register_forward_hook(
            make_hook(i_layer, "q_proj")))
        hooks.append(layer.self_attn.k_proj.register_forward_hook(
            make_hook(i_layer, "k_proj")))
        hooks.append(layer.self_attn.v_proj.register_forward_hook(
            make_hook(i_layer, "v_proj")))
        hooks.append(layer.self_attn.o_proj.register_forward_hook(
            make_hook(i_layer, "o_proj")))
        hooks.append(layer.mlp.fc1.register_forward_hook(
            make_hook(i_layer, "fc1")))
        hooks.append(layer.mlp.fc2.register_forward_hook(
            make_hook(i_layer, "fc2")))

    # Run calibration: process each sample to collect activations
    model.eval()
    with torch.no_grad():
        for i in tqdm.tqdm(range(len(dataset)), desc="Calibration"):
            audio = dataset[i]["audio"]["array"].astype(np.float32)
            transcribe(model, tokenizer, audio, feature_extractor)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Apply SVD decomposition and replace layers
    component_names = ["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"]

    for i_layer in tqdm.tqdm(range(num_layers), desc="Compressing layers"):
        layer = encoder_layers[i_layer]

        for i, comp_name in enumerate(component_names):
            # Get the linear module
            if comp_name == "q_proj":
                linear_module = layer.self_attn.q_proj
            elif comp_name == "k_proj":
                linear_module = layer.self_attn.k_proj
            elif comp_name == "v_proj":
                linear_module = layer.self_attn.v_proj
            elif comp_name == "o_proj":
                linear_module = layer.self_attn.o_proj
            elif comp_name == "fc1":
                linear_module = layer.mlp.fc1
            elif comp_name == "fc2":
                linear_module = layer.mlp.fc2

            # Concatenate collected activations
            # Each tensor may have different seq_len due to variable-length audio,
            # so flatten each to 2D (tokens, features) before concatenating.
            flattened = []
            for feat in calibration_data[i_layer][comp_name]:
                # feat shape: (batch, seq_len, out_features) or (seq_len, out_features)
                flattened.append(feat.reshape(-1, feat.shape[-1]))
            features = torch.cat(flattened, dim=0)

            # Determine threshold
            if ":" in rank_threshold:
                # Different thresholds for self-attention and MLP layers
                thresh = float(rank_threshold.split(":")[0]) if i <= 3 else float(rank_threshold.split(":")[1])
            else:
                thresh = float(rank_threshold)

            features = features.to(device).float()
            Y_mean = features.mean(dim=0, keepdim=False).to(device)

            # Center the data
            features_centered = features - Y_mean

            # Compute SVD (X_centered = U * S * Vt)
            U, S, Vt = torch.linalg.svd(features_centered, full_matrices=False)

            # Determine k: smallest multiple of 16 that satisfies accuracy constraint
            S_F = S ** 2

            k = -1
            for j in range(0, len(S_F), 16):
                if S_F[:j].sum() / S_F.sum() > thresh:
                    k = j
                    break
            print(f"Layer {i_layer}, Component {comp_name}, k = {k} / {len(S_F)}")

            # If k is too big, no benefit in using low rank approximation
            if i <= 3:
                if k > 0.5 * base_dim or k == -1:
                    continue
            else:
                if k > 0.8 * base_dim or k == -1:
                    continue

            V = Vt.T
            V_k = V[:, :k]  # top k principal components

            V_k = V_k.to(device)
            W = linear_module.weight.T.to(device).float()
            w1 = W @ V_k
            w2 = V_k.T

            # Handle bias based on whether the original layer has bias
            if linear_module.bias is None:
                # No bias in original (attention projections)
                bias = Y_mean - Y_mean @ V_k @ V_k.T
            else:
                # Has bias (MLP fc1, fc2)
                original_bias = linear_module.bias.to(device).float()
                bias = Y_mean + (original_bias - Y_mean) @ V_k @ V_k.T

            # Create low-rank replacement
            new_layer = LinearLowRank(
                w1.to(linear_module.weight.dtype).to(device),
                w2.to(linear_module.weight.dtype).to(device),
                bias.to(linear_module.weight.dtype).to(device),
            )

            # Replace the linear module
            if comp_name == "q_proj":
                layer.self_attn.q_proj = new_layer
            elif comp_name == "k_proj":
                layer.self_attn.k_proj = new_layer
            elif comp_name == "v_proj":
                layer.self_attn.v_proj = new_layer
            elif comp_name == "o_proj":
                layer.self_attn.o_proj = new_layer
            elif comp_name == "fc1":
                layer.mlp.fc1 = new_layer
            elif comp_name == "fc2":
                layer.mlp.fc2 = new_layer

            # Free memory
            features = features.cpu()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MoonshineForConditionalGeneration.from_pretrained(args.base_model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.base_model)

    # ─────────────────────────────────────────────────────────────────────
    # CALIBRATION DATA
    # Use dev/validation splits (NOT test) to avoid contaminating evaluation.
    # Sources:
    #   - LibriSpeech dev-clean (2703 samples, read English speech)
    #   - LibriSpeech dev-other (2864 samples, noisier read English speech)
    #   - Common Voice (validation split, diverse accents/recording conditions)
    #
    # Literature guidance:
    #   - LiteASR paper (2502.20583): uses 100 samples from ESB benchmarks
    #   - SpQR (2306.03078): uses 128 samples for Hessian estimation
    #   - GPTQ: uses 128 samples from training distribution
    #   - AWQ: uses 128 samples
    #   - General consensus: 100-256 diverse samples is sufficient for
    #     calibration-based compression. More samples help stability but
    #     with diminishing returns beyond ~256.
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Loading calibration data ({args.num_calibration_samples} samples)")
    print(f"{'='*60}")

    calibrate_datasets = []

    # LibriSpeech dev-clean
    print("Loading librispeech_asr (clean, validation)...")
    ls_clean = load_dataset("librispeech_asr", "clean", split="validation")
    ls_clean = ls_clean.shuffle(seed=42)
    calibrate_datasets.append(ls_clean)
    print(f"  → {len(ls_clean)} samples available")

    # LibriSpeech dev-other
    print("Loading librispeech_asr (other, validation)...")
    ls_other = load_dataset("librispeech_asr", "other", split="validation")
    ls_other = ls_other.shuffle(seed=42)
    calibrate_datasets.append(ls_other)
    print(f"  → {len(ls_other)} samples available")

    # Common Voice (English, validation split)
    if args.use_commonvoice:
        print("Loading mozilla-foundation/common_voice_17_0 (en, validation)...")
        try:
            cv = load_dataset(
                "mozilla-foundation/common_voice_17_0",
                "en",
                split="validation",
                trust_remote_code=True,
            )
            cv = cv.shuffle(seed=42)
            calibrate_datasets.append(cv)
            print(f"  → {len(cv)} samples available")
        except Exception as e:
            print(f"  → WARNING: Could not load Common Voice: {e}")
            print(f"  → Continuing without Common Voice...")

    # Merge and select calibration samples
    # Take equal portions from each source for diversity
    samples_per_source = args.num_calibration_samples // len(calibrate_datasets)
    remainder = args.num_calibration_samples % len(calibrate_datasets)

    selected = []
    for i, ds in enumerate(calibrate_datasets):
        n = samples_per_source + (1 if i < remainder else 0)
        n = min(n, len(ds))
        selected.append(ds.select(range(n)))

    calibrate_dataset = concatenate_datasets(selected).shuffle(seed=42)
    print(f"\nTotal calibration samples: {len(calibrate_dataset)}")
    print(f"  Sources: LibriSpeech dev-clean, dev-other{', Common Voice' if args.use_commonvoice else ''}")
    print(f"{'='*60}\n")

    if args.low_rank:
        model = apply_low_rank(
            model, calibrate_dataset, args.rank_threshold,
            tokenizer, feature_extractor, device,
        )

        if args.save_weight:
            save_path = f"lite-moonshine-{args.base_model.split('/')[-1]}_{args.rank_threshold}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"\nSaved compressed weights to: {save_path}")

    # Parameter counts (correct, accounting for weight tying)
    encoder_params = sum(p.numel() for p in model.model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.model.decoder.parameters())
    total_unique = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"COMPRESSION RESULTS")
    print(f"{'='*60}")
    print(f"  Encoder parameters: {encoder_params:,} (original: 20,153,120)")
    print(f"  Decoder parameters: {decoder_params:,} (unchanged)")
    print(f"  Total (unique):     {total_unique:,} (original: 61,513,920)")
    print(f"  Encoder reduction:  {100*(1 - encoder_params/20_153_120):.1f}%")
    print(f"  Total reduction:    {100*(1 - total_unique/61_513_920):.1f}%")
    print(f"{'='*60}")

    if not args.do_eval:
        return

    # Evaluation on LibriSpeech test splits (separate from calibration data!)
    print(f"\n{'='*60}")
    print(f"EVALUATION (LibriSpeech test-clean & test-other)")
    print(f"{'='*60}")

    eval_splits = [
        ("librispeech_asr", "clean", "test"),
        ("librispeech_asr", "other", "test"),
    ]

    total_sum_wer = 0
    wer_metric = evaluate.load("wer")

    for dataset_name, config, split in eval_splits:
        print(f"\nEvaluating on {dataset_name} ({config}, {split})...")
        eval_dataset = load_dataset(dataset_name, config, split=split)

        if args.max_eval_samples is not None and args.max_eval_samples < len(eval_dataset):
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))

        sum_wer = 0
        for i in tqdm.tqdm(range(len(eval_dataset)), desc=f"{config}-{split}"):
            audio = eval_dataset[i]["audio"]["array"].astype(np.float32)
            ref_text = eval_dataset[i]["text"]

            pred = transcribe(model, tokenizer, audio, feature_extractor)

            wer = wer_metric.compute(
                references=[data_utils.normalizer(ref_text)],
                predictions=[data_utils.normalizer(pred)]
            )
            sum_wer += round(100 * wer, 2)

        avg_wer = sum_wer / len(eval_dataset)
        total_sum_wer += avg_wer
        print(f"  {config}-{split} WER: {avg_wer:.2f}%")

        with open('output.txt', 'a') as f:
            f.write(f"encoder_params: {encoder_params} | {config}-{split} WER: {avg_wer:.2f}%\n")

    total_avg_wer = total_sum_wer / len(eval_splits)
    print(f"\n  Average WER: {total_avg_wer:.2f}%")

    with open('output.txt', 'a') as f:
        f.write(f"Total avg WER: {total_avg_wer:.2f}% | threshold: {args.rank_threshold} | encoder_params: {encoder_params}\n")
        f.write("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_model",
        type=str,
        default="usefulsensors/moonshine-base",
        help="Base model name on HuggingFace. E.g. 'usefulsensors/moonshine-base'.",
    )
    parser.add_argument(
        "--low_rank",
        action="store_true",
        help="Whether to apply low-rank approximation for the activations",
    )
    parser.add_argument(
        "--rank_threshold",
        type=str,
        default="0.99:0.999",
        help="The threshold for the rank approximation. If two values separated by ':', the first is for self-attention layers and the second is for MLP layers",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=1000,
        help="Number of samples to be evaluated. Put a lower number e.g. 64 for testing this script.",
    )
    parser.add_argument(
        "--num_calibration_samples",
        type=int,
        default=100,
        help="Number of samples to be used for calibration",
    )
    parser.add_argument(
        "--save_weight",
        action="store_true",
        help="Whether to save the compressed weights",
    )
    parser.add_argument(
        "--use_commonvoice",
        action="store_true",
        help="Include Common Voice (English) in calibration data for more diversity",
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="Whether to evaluate the model on LibriSpeech test splits after compression",
    )
    args = parser.parse_args()

    main(args)
