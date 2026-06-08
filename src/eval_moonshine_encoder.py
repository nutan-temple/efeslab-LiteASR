"""
Evaluate a LiteASR-quantized Moonshine encoder on LibriSpeech.

This script assembles a Moonshine model from three weight sources:

  * Conv frontend (conv1/conv2/conv3/groupnorm/layer_norm) -> taken from the
    quantized model checkpoint (.pth).
  * Transformer encoder layers -> the low-rank ("quantized") layers produced by
    LiteASR low-rank factorization, loaded from the .pth checkpoint.
  * Decoder (+ token embedding / lm head) -> loaded from the ORIGINAL pretrained
    Moonshine architecture on HuggingFace.

It then computes Word Error Rate (WER) on LibriSpeech (default: test-clean).

The .pth file is expected to be a state_dict saved by `compress_moonshine.py`
(a full `MoonshineForConditionalGeneration.state_dict()` in which some encoder
`nn.Linear` modules were replaced by `LinearLowRank` modules whose parameters are
`weight1`, `weight2`, `bias` and whose forward is `(x @ weight1) @ weight2 + bias`).

Encoder-only checkpoints (keys starting with `encoder.` or `model.encoder.`) are
also supported -- decoder weights are always taken from the original model.

Usage:
    python src/eval_moonshine_encoder.py \
        --compressed_weights lite-moonshine-moonshine-base_0.99:0.999.pth \
        --model usefulsensors/moonshine-base \
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

# Ensure src/ is on the path for sibling module imports (normalizer/).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from normalizer import EnglishTextNormalizer


# Encoder transformer components that LiteASR may have factorized, mapped to the
# attribute path inside a HuggingFace Moonshine encoder layer.
COMPONENT_MAP = {
    "q_proj": "self_attn.q_proj",
    "k_proj": "self_attn.k_proj",
    "v_proj": "self_attn.v_proj",
    "o_proj": "self_attn.o_proj",
    "fc1": "mlp.fc1",
    "fc2": "mlp.fc2",
}


class LinearLowRank(nn.Module):
    """Low-rank linear layer matching the one saved by compress_moonshine.py.

    forward: (x @ weight1) @ weight2 + bias
      weight1: (in_features, low_rank_features)
      weight2: (low_rank_features, out_features)
      bias:    (out_features,)
    """

    def __init__(self, weight1: torch.Tensor, weight2: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.weight1 = nn.Parameter(weight1)
        self.weight2 = nn.Parameter(weight2)
        self.bias = nn.Parameter(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.weight1) @ self.weight2 + self.bias


def _load_raw_state_dict(path, device):
    """Load a checkpoint that may be a state_dict or a full module."""
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, dict):
        # Some checkpoints wrap the state dict.
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        return obj
    # A pickled nn.Module
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    raise ValueError(f"Unsupported checkpoint format in {path}: {type(obj)}")


def _normalize_encoder_keys(state_dict):
    """Return a state_dict whose keys are relative to a MoonshineModel.

    Target namespace uses keys like `encoder.layers.0.self_attn.q_proj.weight1`
    and `encoder.conv1.weight` (i.e. relative to `model.model`).

    Handles inputs whose keys start with `model.` (full
    MoonshineForConditionalGeneration state dict), `encoder.` (MoonshineModel
    state dict) or an encoder-only dict whose keys start directly with `layers.`,
    `conv1.`, `groupnorm.`, `layer_norm.`.
    """
    keys = list(state_dict.keys())

    # Full MoonshineForConditionalGeneration state dict: encoder weights live
    # under `model.encoder.`. Strip the leading `model.` so they become
    # `encoder.*` (other keys such as a tied `proj_out` are ignored).
    if any(k.startswith("model.encoder.") for k in keys):
        return {
            k[len("model."):]: v
            for k, v in state_dict.items()
            if k.startswith("model.encoder.")
        }

    # MoonshineModel state dict: keys already live under `encoder.` / `decoder.`.
    if any(k.startswith("encoder.") for k in keys):
        return {k: v for k, v in state_dict.items() if k.startswith("encoder.")}

    # Otherwise assume an encoder-only checkpoint and add the `encoder.` prefix.
    encoder_markers = ("layers.", "conv1", "conv2", "conv3", "groupnorm", "layer_norm")
    if any(k.startswith(encoder_markers) for k in keys):
        return {f"encoder.{k}": v for k, v in state_dict.items()}

    raise ValueError(
        "Could not locate encoder weights in the checkpoint. "
        f"Sample keys: {keys[:5]}"
    )


def load_model(compressed_weights_path, model_name, device):
    """Build a Moonshine model with quantized encoder + original decoder.

    1. Instantiate the original pretrained Moonshine (decoder + conv + encoder
       architecture with original weights).
    2. Replace factorized encoder linears with LinearLowRank built from the .pth.
    3. Overwrite the encoder (conv frontend + transformer weights) from the .pth,
       leaving the decoder exactly as it was loaded from the original model.
    """
    print(f"Loading original Moonshine architecture/decoder from: {model_name}")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = MoonshineForConditionalGeneration.from_pretrained(
        model_name, trust_remote_code=True
    )

    print(f"Loading quantized encoder weights from: {compressed_weights_path}")
    raw_sd = _load_raw_state_dict(compressed_weights_path, device)
    enc_sd = _normalize_encoder_keys(raw_sd)

    num_layers = config.encoder_num_hidden_layers
    compressed_layers = []

    # Step 1: swap in LinearLowRank modules wherever the checkpoint is factorized.
    for i in range(num_layers):
        layer = model.model.encoder.layers[i]
        for comp_name, attr_path in COMPONENT_MAP.items():
            w1_key = f"encoder.layers.{i}.{attr_path}.weight1"
            w2_key = f"encoder.layers.{i}.{attr_path}.weight2"
            bias_key = f"encoder.layers.{i}.{attr_path}.bias"

            if w1_key in enc_sd and w2_key in enc_sd:
                low_rank_layer = LinearLowRank(
                    enc_sd[w1_key].clone(),
                    enc_sd[w2_key].clone(),
                    enc_sd[bias_key].clone(),
                )
                # Navigate to the parent module and replace the attribute.
                parts = attr_path.split(".")
                obj = layer
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], low_rank_layer)
                compressed_layers.append(f"layer{i}.{comp_name}")

    # Step 2: load the encoder weights (conv frontend + low-rank/full transformer
    # weights) into the MoonshineModel. strict=False because decoder keys are
    # intentionally absent from enc_sd and stay at their original values.
    missing, unexpected = model.model.load_state_dict(enc_sd, strict=False)

    # Every "missing" key should belong to the decoder (we did not touch it).
    enc_missing = [k for k in missing if k.startswith("encoder.")]
    if enc_missing:
        print(
            f"Warning: {len(enc_missing)} encoder key(s) were not provided by the "
            f"checkpoint and keep their original values, e.g. {enc_missing[:5]}"
        )
    if unexpected:
        print(f"Warning: {len(unexpected)} unexpected key(s) in checkpoint, "
              f"e.g. {unexpected[:5]}")

    print(f"Replaced {len(compressed_layers)} encoder linear layer(s) with low-rank "
          f"factorizations.")

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

    if not os.path.isfile(args.compressed_weights):
        print(f"ERROR: File not found: {args.compressed_weights}")
        sys.exit(1)

    model = load_model(args.compressed_weights, args.model, device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model)

    # Parameter accounting.
    encoder_params = sum(p.numel() for p in model.model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.model.decoder.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nEncoder parameters: {encoder_params:,}")
    print(f"Decoder parameters: {decoder_params:,}")
    print(f"Total parameters:   {total_params:,}")

    # Load LibriSpeech.
    print(f"\nLoading dataset: {args.dataset} (config: {args.dataset_config}, "
          f"split: {args.split})")
    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    if args.max_samples is not None and args.max_samples < len(dataset):
        dataset = dataset.select(range(args.max_samples))
    print(f"Evaluating on {len(dataset)} samples")

    normalizer = EnglishTextNormalizer()
    wer_metric = evaluate.load("wer")

    all_predictions = []
    all_references = []

    for i in tqdm(range(len(dataset)), desc="Evaluating"):
        sample = dataset[i]
        audio = sample["audio"]["array"].astype(np.float32)

        ref_text = sample.get("text", "")
        if not ref_text.strip():
            continue
        norm_ref = normalizer(ref_text)
        if not norm_ref.strip():
            continue

        generated_ids = transcribe(model, feature_extractor, audio, device)
        pred_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        norm_pred = normalizer(pred_text)

        all_predictions.append(norm_pred)
        all_references.append(norm_ref)

    if len(all_references) == 0:
        print("No valid samples found for evaluation.")
        return

    wer = wer_metric.compute(references=all_references, predictions=all_predictions)
    wer_pct = round(100 * wer, 2)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Base model (decoder/arch): {args.model}")
    print(f"Quantized encoder weights: {args.compressed_weights}")
    print(f"Dataset:                   {args.dataset} "
          f"({args.dataset_config}, {args.split})")
    print(f"Samples evaluated:         {len(all_references)}")
    print(f"Encoder params:            {encoder_params:,}")
    print(f"Decoder params:            {decoder_params:,}")
    print(f"Total params:              {total_params:,}")
    print(f"Word Error Rate:           {wer_pct}%")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a LiteASR-quantized Moonshine encoder (with the "
                    "original Moonshine decoder) on LibriSpeech."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="usefulsensors/moonshine-base",
        help="Original Moonshine model on HuggingFace. Source of the decoder "
             "weights and the model architecture/config "
             "(default: usefulsensors/moonshine-base).",
    )
    parser.add_argument(
        "--compressed_weights",
        type=str,
        required=True,
        help="Path to the quantized .pth checkpoint (from compress_moonshine.py). "
             "Its conv frontend and low-rank transformer encoder weights are used.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="librispeech_asr",
        help="HuggingFace dataset name (default: librispeech_asr).",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="clean",
        help="Dataset configuration/subset (default: clean).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate on (default: test).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all).",
    )

    args = parser.parse_args()
    main(args)
