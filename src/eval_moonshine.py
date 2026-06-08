"""
Evaluate a compressed Moonshine model on LibriSpeech.

Loads the compressed .pth state_dict directly into a MoonshineForConditionalGeneration model
and computes Word Error Rate (WER) on LibriSpeech test-clean.

Usage:
    python src/eval_moonshine.py \
        --compressed_weights lite-moonshine-moonshine-base_0.99:0.999.pth \
        --max_samples 100
"""

import argparse
import os
import sys

import numpy as np
import torch
import evaluate
from tqdm import tqdm
from datasets import load_dataset, Audio
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoConfig, MoonshineForConditionalGeneration

# Ensure src/ is on the path for sibling module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from normalizer import EnglishTextNormalizer


def load_model(compressed_weights_path, model_name, device):
    """
    Load the Moonshine model with compressed encoder weights.

    The .pth file saved by compress_moonshine.py is a full model state_dict where
    encoder LinearLowRank layers have weight1/weight2/bias instead of weight/bias.
    We load the base model architecture and then override weights from the .pth.
    """
    print(f"Loading compressed weights from: {compressed_weights_path}")
    state_dict = torch.load(compressed_weights_path, map_location=device, weights_only=False)

    # Load config only (no model weights) to build architecture
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    # The compressed state_dict comes from a MoonshineForConditionalGeneration model
    # where some encoder nn.Linear layers have been replaced with LinearLowRank.
    # We need to instantiate a model that matches the state_dict structure.
    # Since LinearLowRank has (weight1, weight2, bias) instead of (weight, bias),
    # we use from_pretrained with state_dict override.

    # First load the base model to get correct architecture
    model = MoonshineForConditionalGeneration.from_pretrained(
        model_name, trust_remote_code=True
    )

    # Now replace encoder layers that have low-rank structure
    # Detect which layers are compressed by checking for weight1 keys
    import torch.nn as nn

    class LinearLowRank(nn.Module):
        def __init__(self, weight1, weight2, bias):
            super().__init__()
            self.weight1 = nn.Parameter(weight1)
            self.weight2 = nn.Parameter(weight2)
            self.bias = nn.Parameter(bias)

        def forward(self, x):
            return (x @ self.weight1) @ self.weight2 + self.bias

    num_layers = config.encoder_num_hidden_layers
    component_map = {
        "q_proj": "self_attn.q_proj",
        "k_proj": "self_attn.k_proj",
        "v_proj": "self_attn.v_proj",
        "o_proj": "self_attn.o_proj",
        "fc1": "mlp.fc1",
        "fc2": "mlp.fc2",
    }

    for i in range(num_layers):
        layer = model.model.encoder.layers[i]
        for comp_name, attr_path in component_map.items():
            w1_key = f"model.encoder.layers.{i}.{attr_path}.weight1"
            w2_key = f"model.encoder.layers.{i}.{attr_path}.weight2"
            bias_key = f"model.encoder.layers.{i}.{attr_path}.bias"

            if w1_key in state_dict:
                # This layer is compressed - replace with LinearLowRank
                w1 = state_dict[w1_key]
                w2 = state_dict[w2_key]
                bias = state_dict[bias_key]

                low_rank_layer = LinearLowRank(w1, w2, bias)

                # Set the attribute on the model
                parts = attr_path.split(".")
                obj = layer
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], low_rank_layer)

                # Remove these keys from state_dict so load_state_dict doesn't complain
                del state_dict[w1_key]
                del state_dict[w2_key]
                del state_dict[bias_key]

                # Also remove original weight/bias keys if present
                orig_w_key = f"model.encoder.layers.{i}.{attr_path}.weight"
                orig_b_key = f"model.encoder.layers.{i}.{attr_path}.bias"
                state_dict.pop(orig_w_key, None)
                state_dict.pop(orig_b_key, None)

    # Load remaining weights (decoder, embeddings, conv layers, etc.)
    # Use strict=False since we already handled low-rank layers
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    # The missing keys should only be the low-rank layers we already set
    if unexpected:
        print(f"Warning: unexpected keys in state_dict: {unexpected[:5]}...")

    model = model.to(device)
    model.eval()
    return model


def transcribe(model, feature_extractor, audio, device):
    """Transcribe a single audio sample."""
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
    input_values = inputs["input_values"].to(device)
    with torch.no_grad():
        generated_ids = model.generate(input_values=input_values)
    return generated_ids


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load compressed model
    if not args.compressed_weights:
        print("ERROR: --compressed_weights is required. Provide the path to the .pth file.")
        print("Example: --compressed_weights lite-moonshine-moonshine-base_0.99:0.999.pth")
        sys.exit(1)

    if not os.path.isfile(args.compressed_weights):
        print(f"ERROR: File not found: {args.compressed_weights}")
        sys.exit(1)

    model = load_model(args.compressed_weights, args.model, device)

    # Load tokenizer and feature extractor
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model)

    # Print model parameter counts
    encoder_params = sum(p.numel() for p in model.model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.model.decoder.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nEncoder parameters: {encoder_params:,}")
    print(f"Decoder parameters: {decoder_params:,}")
    print(f"Total parameters:   {total_params:,}")

    # Load LibriSpeech dataset
    print(f"\nLoading dataset: {args.dataset} (config: {args.dataset_config}, split: {args.split})")
    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    if args.max_samples is not None and args.max_samples < len(dataset):
        dataset = dataset.select(range(args.max_samples))

    print(f"Evaluating on {len(dataset)} samples")

    # Initialize text normalizer and WER metric
    normalizer = EnglishTextNormalizer()
    wer_metric = evaluate.load("wer")

    # Run inference
    all_predictions = []
    all_references = []

    for i in tqdm(range(len(dataset)), desc="Evaluating"):
        sample = dataset[i]
        audio = sample["audio"]["array"].astype(np.float32)

        # Get reference text
        ref_text = sample.get("text", "")
        if not ref_text.strip():
            continue

        norm_ref = normalizer(ref_text)
        if not norm_ref.strip():
            continue

        # Transcribe
        generated_ids = transcribe(model, feature_extractor, audio, device)
        pred_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        norm_pred = normalizer(pred_text)

        all_predictions.append(norm_pred)
        all_references.append(norm_ref)

    # Compute WER
    if len(all_references) == 0:
        print("No valid samples found for evaluation.")
        return

    wer = wer_metric.compute(references=all_references, predictions=all_predictions)
    wer_pct = round(100 * wer, 2)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model:              {args.model}")
    print(f"Compressed weights: {args.compressed_weights}")
    print(f"Dataset:            {args.dataset} ({args.dataset_config}, {args.split})")
    print(f"Samples evaluated:  {len(all_references)}")
    print(f"Encoder params:     {encoder_params:,}")
    print(f"Decoder params:     {decoder_params:,}")
    print(f"Total params:       {total_params:,}")
    print(f"Word Error Rate:    {wer_pct}%")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate compressed Moonshine model on LibriSpeech"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="usefulsensors/moonshine-base",
        help="Base model name on HuggingFace (default: usefulsensors/moonshine-base)",
    )
    parser.add_argument(
        "--compressed_weights",
        type=str,
        required=True,
        help="Path to compressed model state_dict .pth file (from compress_moonshine.py)",
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
        default="clean",
        help="Dataset configuration/subset (default: clean)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all)",
    )

    args = parser.parse_args()
    main(args)
