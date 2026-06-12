"""
Model utilities for Moonshine W8A8 quantization.

Handles:
  - Loading Moonshine from HuggingFace
  - Loading LiteASR .pth (replacing encoder layers with LinearLowRank)
  - Extracting layers, sublayers, and sequential groups
"""

import torch
import torch.nn as nn
from transformers import MoonshineForConditionalGeneration, AutoProcessor


# ── LinearLowRank module from LiteASR ────────────────────────────────────────

class LinearLowRank(nn.Module):
    """
    Low-rank linear layer from LiteASR compression.

    Instead of storing a full weight matrix W of shape (out_features, in_features),
    this stores two smaller matrices:
      weight1: (in_features, rank)
      weight2: (rank, out_features)
    plus a bias vector.

    Forward pass: output = (x @ weight1) @ weight2 + bias

    This reduces parameters from (in * out) to (in * rank + rank * out),
    which is beneficial when rank << min(in, out).
    """

    def __init__(self, weight1, weight2, bias):
        super().__init__()
        self.weight1 = nn.Parameter(weight1)  # (in_features, rank)
        self.weight2 = nn.Parameter(weight2)  # (rank, out_features)
        self.bias = nn.Parameter(bias)        # (out_features,)

    @property
    def in_features(self):
        return self.weight1.shape[0]

    @property
    def out_features(self):
        return self.weight2.shape[1]

    @property
    def rank(self):
        return self.weight1.shape[1]

    def forward(self, x):
        return (x @ self.weight1) @ self.weight2 + self.bias


# ── Model loading ────────────────────────────────────────────────────────────

def load_moonshine_model(model_name="usefulsensors/moonshine-base"):
    """
    Load Moonshine model from HuggingFace.

    Args:
        model_name: HuggingFace model identifier (default: usefulsensors/moonshine-base)

    Returns:
        model: MoonshineForConditionalGeneration instance
    """
    print(f"Loading base model: {model_name}")
    model = MoonshineForConditionalGeneration.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


def load_liteasr_pth(model, pth_path):
    """
    Load a LiteASR .pth checkpoint into a Moonshine model.

    The .pth file contains a state_dict where some encoder nn.Linear layers
    have been replaced with LinearLowRank (weight1, weight2, bias triplets).

    Keys in .pth:
      - model.encoder.layers.X.self_attn.q_proj.weight1 (compressed)
      - model.encoder.layers.X.self_attn.q_proj.weight2 (compressed)
      - model.encoder.layers.X.self_attn.q_proj.bias    (compressed)
      - model.encoder.layers.4.self_attn.q_proj.weight  (uncompressed, skipped)
      - model.decoder.layers.*                          (all uncompressed)

    Args:
        model: MoonshineForConditionalGeneration instance
        pth_path: Path to the .pth file

    Returns:
        model: Modified model with LinearLowRank encoder layers
        replaced_count: Number of layers replaced
    """
    print(f"Loading LiteASR weights: {pth_path}")
    state_dict = torch.load(pth_path, map_location="cpu", weights_only=False)

    # Extract encoder keys (strip "model." prefix for model.model access)
    encoder_sd = {}
    for key, tensor in state_dict.items():
        if key.startswith("model.encoder."):
            encoder_sd[key[len("model."):]] = tensor

    # Component paths that may be low-rank compressed
    component_paths = [
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.fc1", "mlp.fc2",
    ]

    replaced = 0
    config = model.config
    for i in range(config.encoder_num_hidden_layers):
        layer = model.model.encoder.layers[i]
        for attr_path in component_paths:
            w1_key = f"encoder.layers.{i}.{attr_path}.weight1"
            w2_key = f"encoder.layers.{i}.{attr_path}.weight2"
            bias_key = f"encoder.layers.{i}.{attr_path}.bias"

            if w1_key in encoder_sd and w2_key in encoder_sd:
                w1 = encoder_sd.pop(w1_key)
                w2 = encoder_sd.pop(w2_key)
                bias = encoder_sd.pop(bias_key)
                _setattr_nested(layer, attr_path, LinearLowRank(w1, w2, bias))
                replaced += 1
                # Remove uncompressed weight key if present
                encoder_sd.pop(f"encoder.layers.{i}.{attr_path}.weight", None)

    # Load remaining non-compressed state dict entries
    model.model.load_state_dict(encoder_sd, strict=False)
    print(f"  Replaced {replaced} encoder sublayers with LinearLowRank")
    return model, replaced


def _setattr_nested(obj, path, value):
    """Set a nested attribute: _setattr_nested(layer, 'self_attn.q_proj', module)"""
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


# ── Layer access ─────────────────────────────────────────────────────────────

def get_encoder_layers(model):
    """Return the list of encoder transformer layers."""
    return model.model.encoder.layers


def get_decoder_layers(model):
    """Return the list of decoder transformer layers."""
    return model.model.decoder.layers


def find_sublayers(module, layer_types=(nn.Linear, LinearLowRank)):
    """
    Find all sublayers of specified types within a module.

    Args:
        module: PyTorch module to search
        layer_types: Tuple of layer types to find

    Returns:
        dict: {name: layer} for all matching sublayers
    """
    res = {}
    for name, layer in module.named_modules():
        if isinstance(layer, layer_types):
            res[name] = layer
    return res


def get_encoder_sequential_groups():
    """
    Return sequential groups for Moonshine encoder layers.

    Groups are processed sequentially - layers within a group can be
    quantized in parallel because they share the same input.

    Moonshine encoder layers have:
      - self_attn: q_proj, k_proj, v_proj (share input), o_proj
      - mlp: fc1, fc2
    """
    return [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["self_attn.o_proj"],
        ["mlp.fc1"],
        ["mlp.fc2"],
    ]


def get_decoder_sequential_groups():
    """
    Return sequential groups for Moonshine decoder layers.

    Moonshine decoder layers have:
      - self_attn: q_proj, k_proj, v_proj, o_proj
      - encoder_attn (cross-attention): q_proj, k_proj, v_proj, o_proj
      - mlp (SwiGLU): fc1, fc2
    """
    return [
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["self_attn.o_proj"],
        ["encoder_attn.q_proj", "encoder_attn.k_proj", "encoder_attn.v_proj"],
        ["encoder_attn.o_proj"],
        ["mlp.fc1"],
        ["mlp.fc2"],
    ]


def get_processor(model_name="usefulsensors/moonshine-base"):
    """Load the AutoProcessor for Moonshine (handles tokenization and feature extraction)."""
    return AutoProcessor.from_pretrained(model_name)


def count_parameters(model):
    """Return (encoder_params, decoder_params, total_params)."""
    enc_params = sum(p.numel() for p in model.model.encoder.parameters())
    dec_params = sum(p.numel() for p in model.model.decoder.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    return enc_params, dec_params, total_params


def model_size_bytes(model):
    """Return total model size in bytes."""
    return sum(p.numel() * p.element_size() for p in model.parameters())
