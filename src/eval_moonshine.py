"""
Evaluate a compressed Moonshine model on LibriSpeech test-clean.

Loads a compressed model state_dict (saved by compress_moonshine.py) into a
LiteMoonshineForConditionalGeneration model and computes Word Error Rate (WER)
on the LibriSpeech test-clean split.

Usage:
    python src/eval_moonshine.py \
        --compressed_weights lite-moonshine-moonshine-base_0.99:0.999.pth \
        --max_samples 100

    # Evaluate the uncompressed baseline (no --compressed_weights):
    python src/eval_moonshine.py --max_samples 100
"""

import argparse
import os
import sys

import numpy as np
import torch
import evaluate
import tqdm
from datasets import load_dataset, Audio
from transformers import AutoTokenizer, AutoFeatureExtractor, MoonshineForConditionalGeneration

# Ensure src/ is on the path for sibling module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lite_moonshine.configuration_lite_moonshine import LiteMoonshineConfig
from lite_moonshine.modeling_lite_moonshine import LiteMoonshineForConditionalGeneration
from normalizer import EnglishTextNormalizer


def infer_low_rank_config(state_dict, num_layers):
    """Infer low_rank_config from a saved state_dict by detecting LinearLowRank layers.

    LinearLowRank layers have weight1 (in_features, low_rank) and weight2 (low_rank, out_features)
    instead of a single weight matrix.
    """
    low_rank_config = []
    component_names = ["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"]

    for i in range(num_layers):
        layer_config = {}
        for comp in component_names:
            # Check if this component is a LinearLowRank (has weight1 key)
            if comp in ("q_proj", "k_proj", "v_proj", "o_proj"):
                key = f"model.encoder.layers.{i}.self_attn.{comp}.weight1"
            else:
                key = f"model.encoder.layers.{i}.mlp.{comp}.weight1"

            if key in state_dict:
                # low_rank = weight1.shape[1]
                layer_config[comp] = state_dict[key].shape[1]

        low_rank_config.append(layer_config)

    return low_rank_config


def load_compressed_model(model_name, compressed_weights_path, device):
    """Load a compressed Moonshine model from a state_dict .pth file.

    Infers the low_rank_config from the state_dict and creates a
    LiteMoonshineForConditionalGeneration model with the appropriate architecture.
    """
    state_dict = torch.load(compressed_weights_path, map_location=device, weights_only=True)

    # Get base config from HuggingFace to determine model dimensions
    base_config = MoonshineForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float32
    ).config
    num_layers = base_config.encoder_num_hidden_layers

    # Infer which layers are compressed and their ranks
    low_rank_config = infer_low_rank_config(state_dict, num_layers)

    # Create LiteMoonshine config with low_rank_config
    config = LiteMoonshineConfig(
        low_rank_config=low_rank_config,
        hidden_size=base_config.hidden_size,
        intermediate_size=base_config.intermediate_size,
        encoder_num_hidden_layers=base_config.encoder_num_hidden_layers,
        decoder_num_hidden_layers=base_config.decoder_num_hidden_layers,
        encoder_num_attention_heads=base_config.encoder_num_attention_heads,
        decoder_num_attention_heads=base_config.decoder_num_attention_heads,
        vocab_size=base_config.vocab_size,
        pad_token_id=base_config.pad_token_id,
        bos_token_id=base_config.bos_token_id,
        eos_token_id=base_config.eos_token_id,
        decoder_start_token_id=base_config.decoder_start_token_id,
    )

    # Instantiate model with LinearLowRank layers
    model = LiteMoonshineForConditionalGeneration(config)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model


def load_baseline_model(model_name, device):
    """Load the uncompressed baseline model from HuggingFace."""
    model = MoonshineForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float32
    ).to(device)
    model.eval()
    return model


def transcribe(model, tokenizer, audio, feature_extractor):
    """Transcribe a single audio sample."""
    inputs = feature_extractor(
        audio, sampling_rate=16000, return_tensors="pt"
    )
    input_values = inputs["input_values"].to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(input_values=input_values)
    text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text


def is_target_text_in_range(ref):
    if ref.strip() == "ignore time segment in scoring":
        return False
    return ref.strip() != ""


def get_text(sample):
    if "text" in sample:
        return sample["text"]
    elif "sentence" in sample:
        return sample["sentence"]
    elif "normalized_text" in sample:
        return sample["normalized_text"]
    elif "transcript" in sample:
        return sample["transcript"]
    elif "transcription" in sample:
        return sample["transcription"]
    else:
        raise ValueError(
            f"Expected transcript column of either 'text', 'sentence', "
            f"'normalized_text' or 'transcript'. Got sample keys: "
            f"{list(sample.keys())}"
        )


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    if args.compressed_weights:
        print(f"Loading compressed model from: {args.compressed_weights}")
        model = load_compressed_model(args.model, args.compressed_weights, device)
    else:
        print(f"Loading baseline (uncompressed) model: {args.model}")
        model = load_baseline_model(args.model, device)

    # Load tokenizer and feature extractor
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model)

    # Print model parameter counts
    encoder_params = sum(p.numel() for p in model.model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.model.decoder.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Encoder parameters: {encoder_params:,}")
    print(f"Decoder parameters: {decoder_params:,}")
    print(f"Total parameters:   {total_params:,}")

    # Load LibriSpeech dataset
    print(f"\nLoading dataset: {args.dataset} (split: {args.split})")
    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    if args.max_samples is not None and args.max_samples < len(dataset):
        dataset = dataset.select(range(args.max_samples))
        print(f"Evaluating on {args.max_samples} samples")
    else:
        print(f"Evaluating on {len(dataset)} samples")

    # Initialize text normalizer and WER metric
    normalizer = EnglishTextNormalizer()
    wer_metric = evaluate.load("wer")

    # Run inference
    all_predictions = []
    all_references = []
    skipped = 0

    print("\nRunning inference...")
    for i in tqdm.tqdm(range(len(dataset)), desc="Evaluating"):
        sample = dataset[i]
        audio = sample["audio"]["array"].astype(np.float32)

        # Get reference text
        ref_text = get_text(sample)
        norm_ref = normalizer(ref_text)

        # Skip empty references
        if not is_target_text_in_range(norm_ref):
            skipped += 1
            continue

        # Transcribe
        pred_text = transcribe(model, tokenizer, audio, feature_extractor)
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
    print(f"Model:             {args.model}")
    if args.compressed_weights:
        print(f"Compressed weights: {args.compressed_weights}")
    print(f"Dataset:           {args.dataset} ({args.dataset_config}, {args.split})")
    print(f"Samples evaluated: {len(all_references)}")
    if skipped > 0:
        print(f"Samples skipped:   {skipped}")
    print(f"Encoder params:    {encoder_params:,}")
    print(f"Total params:      {total_params:,}")
    print(f"Word Error Rate:   {wer_pct}%")
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
        default=None,
        help="Path to compressed model state_dict .pth file. "
             "If not provided, evaluates the uncompressed baseline.",
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
        help="Maximum number of samples to evaluate. Evaluates all if not set.",
    )

    args = parser.parse_args()
    main(args)
