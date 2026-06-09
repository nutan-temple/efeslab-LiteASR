"""
Evaluate a compressed Moonshine model on LibriSpeech using ONLY the .pth file.

Loads the .pth state_dict, builds the model with LinearLowRank encoder layers,
loads original decoder from HuggingFace, and computes WER.

Usage:
    # Evaluate on test-other:
    python src/eval_moonshine.py \
        --compressed_weights lite-moonshine-moonshine-base_0.99:0.999.pth \
        --dataset_config other \
        --split test \
        --max_samples 100

    # Evaluate on test-clean:
    python src/eval_moonshine.py \
        --compressed_weights lite-moonshine-moonshine-base_0.99:0.999.pth \
        --dataset_config clean \
        --split test \
        --max_samples 100
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import evaluate
from tqdm import tqdm
from datasets import load_dataset, Audio
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoFeatureExtractor,
    MoonshineForConditionalGeneration,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalizer import EnglishTextNormalizer


class LinearLowRank(nn.Module):
    """Low-rank layer: forward(x) = (x @ weight1) @ weight2 + bias"""

    def __init__(self, weight1, weight2, bias):
        super().__init__()
        self.weight1 = nn.Parameter(weight1)
        self.weight2 = nn.Parameter(weight2)
        self.bias = nn.Parameter(bias)

    def forward(self, x):
        return (x @ self.weight1) @ self.weight2 + self.bias


def load_model(compressed_weights_path, model_name, device):
    """
    Load model using ONLY the .pth encoder weights + original decoder from HuggingFace.

    Strategy:
    1. Load original model from HuggingFace (correct decoder + architecture)
    2. Scan .pth for weight1/weight2 keys (compressed encoder layers)
    3. Replace those encoder nn.Linear modules with LinearLowRank
    4. Load all encoder weights from .pth (conv frontend + transformer layers)
    5. Decoder stays as loaded from HuggingFace (unchanged)
    """
    print(f"[1/4] Loading original model from HuggingFace: {model_name}")
    model = MoonshineForConditionalGeneration.from_pretrained(
        model_name, trust_remote_code=True
    )
    config = model.config

    print(f"[2/4] Loading .pth checkpoint: {compressed_weights_path}")
    state_dict = torch.load(compressed_weights_path, map_location="cpu", weights_only=False)

    # Extract only encoder keys from the .pth (strip "model." prefix)
    encoder_sd = {}
    for key, tensor in state_dict.items():
        if key.startswith("model.encoder."):
            new_key = key[len("model."):]  # "encoder.layers.0...."
            encoder_sd[new_key] = tensor

    if not encoder_sd:
        raise ValueError("No encoder keys found in .pth file. Expected keys like 'model.encoder.layers.*'")

    print(f"[3/4] Replacing compressed encoder layers with LinearLowRank...")

    # Component paths inside each encoder layer
    component_map = {
        "self_attn.q_proj": "self_attn.q_proj",
        "self_attn.k_proj": "self_attn.k_proj",
        "self_attn.v_proj": "self_attn.v_proj",
        "self_attn.o_proj": "self_attn.o_proj",
        "mlp.fc1": "mlp.fc1",
        "mlp.fc2": "mlp.fc2",
    }

    num_layers = config.encoder_num_hidden_layers
    replaced_count = 0

    for i in range(num_layers):
        layer = model.model.encoder.layers[i]

        for attr_path in component_map.values():
            w1_key = f"encoder.layers.{i}.{attr_path}.weight1"
            w2_key = f"encoder.layers.{i}.{attr_path}.weight2"
            bias_key = f"encoder.layers.{i}.{attr_path}.bias"

            if w1_key in encoder_sd and w2_key in encoder_sd:
                # Build LinearLowRank from .pth tensors
                w1 = encoder_sd.pop(w1_key)
                w2 = encoder_sd.pop(w2_key)
                bias = encoder_sd.pop(bias_key)

                low_rank_layer = LinearLowRank(w1, w2, bias)

                # Replace in model
                parts = attr_path.split(".")
                obj = layer
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], low_rank_layer)
                replaced_count += 1

                # Remove original weight/bias keys if they exist in encoder_sd
                encoder_sd.pop(f"encoder.layers.{i}.{attr_path}.weight", None)

    print(f"      Replaced {replaced_count} linear layers with LinearLowRank")

    # Load remaining encoder weights (conv frontend, layernorms, non-compressed linears)
    print(f"[4/4] Loading remaining encoder weights (conv, norms, uncompressed layers)...")
    missing, unexpected = model.model.load_state_dict(encoder_sd, strict=False)

    # Filter: only report encoder-related missing keys (decoder missing is expected)
    enc_missing = [k for k in missing if k.startswith("encoder.")]
    if enc_missing:
        print(f"      WARNING: {len(enc_missing)} encoder keys missing: {enc_missing[:3]}...")
    if unexpected:
        print(f"      WARNING: {len(unexpected)} unexpected keys: {unexpected[:3]}...")

    model = model.to(device)
    model.eval()

    # Print parameter counts
    encoder_params = sum(p.numel() for p in model.model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.model.decoder.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Encoder parameters: {encoder_params:,}")
    print(f"  Decoder parameters: {decoder_params:,}")
    print(f"  Total parameters:   {total_params:,}")

    return model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    if not os.path.isfile(args.compressed_weights):
        print(f"ERROR: File not found: {args.compressed_weights}")
        sys.exit(1)

    # Load model
    model = load_model(args.compressed_weights, args.model, device)

    # Load tokenizer and feature extractor
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model)

    # Load dataset
    print(f"\nLoading dataset: {args.dataset} (config={args.dataset_config}, split={args.split})")
    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    if args.max_samples is not None and args.max_samples < len(dataset):
        dataset = dataset.select(range(args.max_samples))

    print(f"Evaluating on {len(dataset)} samples\n")

    # Initialize
    normalizer = EnglishTextNormalizer()
    wer_metric = evaluate.load("wer")

    all_predictions = []
    all_references = []

    # Run inference
    for i in tqdm(range(len(dataset)), desc="Transcribing"):
        sample = dataset[i]
        audio = sample["audio"]["array"].astype(np.float32)

        ref_text = sample.get("text", "")
        if not ref_text.strip():
            continue

        norm_ref = normalizer(ref_text)
        if not norm_ref.strip():
            continue

        # Transcribe
        inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
        input_values = inputs["input_values"].to(device)

        with torch.no_grad():
            generated_ids = model.generate(input_values=input_values)

        pred_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        norm_pred = normalizer(pred_text)

        all_predictions.append(norm_pred)
        all_references.append(norm_ref)

    # Compute WER
    if not all_references:
        print("No valid samples found.")
        return

    wer = wer_metric.compute(references=all_references, predictions=all_predictions)
    wer_pct = round(100 * wer, 2)

    # Results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Model:              {args.model}")
    print(f"  Compressed weights: {args.compressed_weights}")
    print(f"  Dataset:            {args.dataset} ({args.dataset_config}, {args.split})")
    print(f"  Samples evaluated:  {len(all_references)}")
    print(f"  Word Error Rate:    {wer_pct}%")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate compressed Moonshine model on LibriSpeech"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="usefulsensors/moonshine-base",
        help="Base model on HuggingFace (source of decoder weights)",
    )
    parser.add_argument(
        "--compressed_weights",
        type=str,
        required=True,
        help="Path to the .pth file from compress_moonshine.py",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="librispeech_asr",
        help="HuggingFace dataset name (default: librispeech_asr)",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="other",
        help="Dataset config: 'clean' or 'other' (default: other)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (default: test)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples to evaluate (default: all)",
    )

    args = parser.parse_args()
    main(args)
