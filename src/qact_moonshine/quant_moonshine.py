"""QuantizedMoonshine: wraps a pretrained Moonshine model with QACT quantization layers.

This module provides:
- QuantizedMoonshine class that replaces Linear and Conv1d layers with quantized variants
- load_pretrained_moonshine() to load from HuggingFace and wrap with quantization
- set_layerwise_precision() for multi-precision co-training support
"""

import math
import re
import warnings
from collections import OrderedDict
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .quant_modules import QuantLinear, QuantConv1d

# Import Moonshine model classes using relative imports from the src/ directory
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from moonshine_model import (
    Moonshine,
    MoonshineModelDimensions,
    Linear,
    Conv1d,
    MultiHeadAttention,
    EncoderBlock,
    DecoderBlock,
    AudioEncoder,
    TextDecoder,
)


def quantize_linear(linear_layer, weight_bit=2, use_scaling=True,
                    quant_mode='symmetric', **kwargs):
    """Wrap an existing Linear (or nn.Linear) layer with QuantLinear.

    Args:
        linear_layer: The original linear layer (Moonshine's Linear or nn.Linear).
        weight_bit: Number of bits for weight quantization.
        use_scaling: Whether to use learnable scaling factors.
        quant_mode: Quantization mode ('symmetric' or 'asymmetric').

    Returns:
        QuantLinear wrapping the original layer.
    """
    return QuantLinear(
        linear_layer,
        weight_bit=weight_bit,
        use_scaling=use_scaling,
        quant_mode=quant_mode,
        **kwargs,
    )


def quantize_conv1d(conv_layer, weight_bit=2, use_scaling=True,
                    quant_mode='symmetric', **kwargs):
    """Wrap an existing Conv1d layer with QuantConv1d.

    Args:
        conv_layer: The original Conv1d layer.
        weight_bit: Number of bits for weight quantization.
        use_scaling: Whether to use learnable scaling factors.
        quant_mode: Quantization mode ('symmetric' or 'asymmetric').

    Returns:
        QuantConv1d wrapping the original layer.
    """
    return QuantConv1d(
        conv_layer,
        weight_bit=weight_bit,
        use_scaling=use_scaling,
        quant_mode=quant_mode,
        **kwargs,
    )


class QuantizedMoonshine(nn.Module):
    """Wraps a pretrained Moonshine model with QACT quantization.

    Replaces Linear and Conv1d layers with QuantLinear and QuantConv1d
    to enable low-bit weight quantization with learnable scaling factors.

    Supports:
    - Encoder Conv1d frontend quantization (conv1, conv2, conv3) at conv_weight_bit
    - Encoder block attention (q/k/v/out) and MLP (fc1/fc2) at enc_weight_bit
    - Optional decoder quantization (self_attn, cross_attn, MLP) at dec_weight_bit
    - Per-layer precision control via set_layerwise_precision()

    Component-wise precision allows different bit-widths per component:
    - Encoder attention + MLP: 2-bit (co-trained with 1-bit via stochastic precision)
    - Encoder Conv frontend: 4-bit (conv layers are more sensitive)
    - Decoder (self-attn, cross-attn, MLP): 4-bit (fixed, no co-training needed)
    - Embeddings & LayerNorms: Full precision (unquantized)
    - Weight-only quantization: activations stay in FP32/FP16

    Args:
        model: A pretrained Moonshine model instance.
        enc_weight_bit: Number of bits for encoder attention/MLP quantization.
        dec_weight_bit: Number of bits for decoder quantization.
        conv_weight_bit: Number of bits for encoder conv frontend quantization.
        use_scaling: Whether to use learnable per-precision scaling factors.
        quant_mode: Quantization mode ('symmetric' or 'asymmetric').
        quant_decoder: Whether to also quantize decoder layers.
    """

    def __init__(
        self,
        model: Moonshine,
        enc_weight_bit: int = 2,
        dec_weight_bit: int = 4,
        conv_weight_bit: int = 4,
        use_scaling: bool = True,
        quant_mode: str = 'symmetric',
        quant_decoder: bool = True,
    ):
        super().__init__()
        self.model = model
        self.enc_weight_bit = enc_weight_bit
        self.dec_weight_bit = dec_weight_bit
        self.conv_weight_bit = conv_weight_bit
        self.use_scaling = use_scaling
        self.quant_mode = quant_mode
        self.quant_decoder = quant_decoder

        # Keep legacy attribute for backward compatibility
        self.weight_bit = enc_weight_bit

        # Track quantized layers per encoder layer for set_layerwise_precision
        self.encoder_layer_quant_modules: List[List[nn.Module]] = []

        # Quantize encoder conv frontend
        self._quant_encoder_convs()

        # Quantize encoder blocks
        self._quant_encoder_blocks()

        # Optionally quantize decoder blocks
        if quant_decoder:
            self.decoder_layer_quant_modules: List[List[nn.Module]] = []
            self._quant_decoder_blocks()

    def _get_quant_kwargs(self, weight_bit: Optional[int] = None):
        """Common kwargs for quantize_linear/quantize_conv1d.

        Args:
            weight_bit: Override bit-width. If None, uses enc_weight_bit.
        """
        return dict(
            weight_bit=weight_bit if weight_bit is not None else self.enc_weight_bit,
            use_scaling=self.use_scaling,
            quant_mode=self.quant_mode,
        )

    def _quant_encoder_convs(self):
        """Replace encoder conv1, conv2, conv3 with QuantConv1d.

        Uses conv_weight_bit (default 4-bit) since conv layers are more
        sensitive to quantization than attention/MLP layers.
        """
        encoder = self.model.encoder
        kwargs = self._get_quant_kwargs(weight_bit=self.conv_weight_bit)

        encoder.conv1 = quantize_conv1d(encoder.conv1, **kwargs)
        encoder.conv2 = quantize_conv1d(encoder.conv2, **kwargs)
        encoder.conv3 = quantize_conv1d(encoder.conv3, **kwargs)

        # Store conv quant modules (treated as "layer -1" or separate from per-layer)
        self.encoder_conv_quant_modules = [
            encoder.conv1, encoder.conv2, encoder.conv3
        ]

    def _quant_encoder_blocks(self):
        """Replace all Linear layers in encoder blocks with QuantLinear.

        Uses enc_weight_bit (default 2-bit) for encoder attention and MLP.
        These layers participate in QACT co-training with stochastic precision.
        """
        encoder = self.model.encoder
        kwargs = self._get_quant_kwargs(weight_bit=self.enc_weight_bit)

        for block in encoder.blocks:
            layer_modules = []

            # Self-attention q/k/v/out
            block.self_attn.query = quantize_linear(block.self_attn.query, **kwargs)
            block.self_attn.key = quantize_linear(block.self_attn.key, **kwargs)
            block.self_attn.value = quantize_linear(block.self_attn.value, **kwargs)
            block.self_attn.out = quantize_linear(block.self_attn.out, **kwargs)
            layer_modules.extend([
                block.self_attn.query,
                block.self_attn.key,
                block.self_attn.value,
                block.self_attn.out,
            ])

            # MLP fc1/fc2
            block.fc1 = quantize_linear(block.fc1, **kwargs)
            block.fc2 = quantize_linear(block.fc2, **kwargs)
            layer_modules.extend([block.fc1, block.fc2])

            self.encoder_layer_quant_modules.append(layer_modules)

    def _quant_decoder_blocks(self):
        """Replace all Linear layers in decoder blocks with QuantLinear.

        Uses dec_weight_bit (default 4-bit) for decoder self-attn, cross-attn,
        and MLP layers. The decoder uses fixed precision (no co-training needed
        at 4-bit since it is stable at this precision). Lower than 4-bit causes
        error accumulation in autoregressive decoding.
        """
        decoder = self.model.decoder
        kwargs = self._get_quant_kwargs(weight_bit=self.dec_weight_bit)

        for block in decoder.blocks:
            layer_modules = []

            # Self-attention q/k/v/out
            block.self_attn.query = quantize_linear(block.self_attn.query, **kwargs)
            block.self_attn.key = quantize_linear(block.self_attn.key, **kwargs)
            block.self_attn.value = quantize_linear(block.self_attn.value, **kwargs)
            block.self_attn.out = quantize_linear(block.self_attn.out, **kwargs)
            layer_modules.extend([
                block.self_attn.query,
                block.self_attn.key,
                block.self_attn.value,
                block.self_attn.out,
            ])

            # Cross-attention q/k/v/out
            block.cross_attn.query = quantize_linear(block.cross_attn.query, **kwargs)
            block.cross_attn.key = quantize_linear(block.cross_attn.key, **kwargs)
            block.cross_attn.value = quantize_linear(block.cross_attn.value, **kwargs)
            block.cross_attn.out = quantize_linear(block.cross_attn.out, **kwargs)
            layer_modules.extend([
                block.cross_attn.query,
                block.cross_attn.key,
                block.cross_attn.value,
                block.cross_attn.out,
            ])

            # MLP fc1/fc2
            block.fc1 = quantize_linear(block.fc1, **kwargs)
            block.fc2 = quantize_linear(block.fc2, **kwargs)
            layer_modules.extend([block.fc1, block.fc2])

            self.decoder_layer_quant_modules.append(layer_modules)

    def set_layerwise_precision(self, prec_list: List[int]):
        """Set per-layer precision for encoder layers.

        This enables multi-precision co-training where different encoder layers
        can operate at different bit widths (e.g., [2, 2, 1, 1, 2, 2]).

        Args:
            prec_list: List of precision values (bit widths), one per encoder layer.
                       Length must match the number of encoder layers.
        """
        n_layers = len(self.encoder_layer_quant_modules)
        if len(prec_list) != n_layers:
            raise ValueError(
                f"prec_list length ({len(prec_list)}) must match number of "
                f"encoder layers ({n_layers})"
            )

        for layer_idx, precision in enumerate(prec_list):
            for module in self.encoder_layer_quant_modules[layer_idx]:
                module.set_precision(precision)

    def set_encoder_conv_precision(self, precision: int):
        """Set precision for the encoder convolutional frontend.

        Args:
            precision: Bit width for all conv layers.
        """
        for module in self.encoder_conv_quant_modules:
            module.set_precision(precision)

    def set_decoder_layerwise_precision(self, prec_list: List[int]):
        """Set per-layer precision for decoder layers.

        Args:
            prec_list: List of precision values (bit widths), one per decoder layer.
                       Length must match the number of decoder layers.

        Raises:
            RuntimeError: If decoder was not quantized (quant_decoder=False).
        """
        if not self.quant_decoder:
            raise RuntimeError(
                "Decoder was not quantized. Set quant_decoder=True when creating QuantizedMoonshine."
            )

        n_layers = len(self.decoder_layer_quant_modules)
        if len(prec_list) != n_layers:
            raise ValueError(
                f"prec_list length ({len(prec_list)}) must match number of "
                f"decoder layers ({n_layers})"
            )

        for layer_idx, precision in enumerate(prec_list):
            for module in self.decoder_layer_quant_modules[layer_idx]:
                module.set_precision(precision)

    def set_all_precision(self, precision: int):
        """Set a uniform precision for all quantized modules.

        Args:
            precision: Bit width to apply to all quantized layers.
        """
        self.set_encoder_conv_precision(precision)
        prec_list = [precision] * len(self.encoder_layer_quant_modules)
        self.set_layerwise_precision(prec_list)

        if self.quant_decoder:
            dec_prec_list = [precision] * len(self.decoder_layer_quant_modules)
            self.set_decoder_layerwise_precision(dec_prec_list)

    def forward(self, waveform, decoder_input_ids):
        """Forward pass through the quantized model.

        Delegates to the underlying Moonshine model which now has
        quantized layers in place.

        Args:
            waveform: (batch, audio_len) raw audio waveform.
            decoder_input_ids: (batch, seq_len) initial decoder token IDs.

        Returns:
            List of generated token IDs.
        """
        return self.model(waveform, decoder_input_ids)

    def encode(self, waveform):
        """Run only the encoder (useful for encoder-only evaluation).

        Args:
            waveform: (batch, audio_len) raw audio waveform.

        Returns:
            Encoder output tensor.
        """
        return self.model.encoder(waveform)


def load_pretrained_moonshine(
    model_name: str = "usefulsensors/moonshine-base",
    device: str = "cpu",
    enc_weight_bit: int = 2,
    dec_weight_bit: int = 4,
    conv_weight_bit: int = 4,
    use_scaling: bool = True,
    quant_mode: str = 'symmetric',
    quant_decoder: bool = True,
) -> QuantizedMoonshine:
    """Load a pretrained Moonshine model from HuggingFace and wrap with quantization.

    Downloads the model from HuggingFace, converts it to the custom Moonshine
    architecture, and wraps all Linear/Conv1d layers with QACT quantization.

    Supports component-wise precision:
    - Encoder attention/MLP: enc_weight_bit (default 2-bit, co-trained with 1-bit)
    - Encoder conv frontend: conv_weight_bit (default 4-bit, higher precision)
    - Decoder: dec_weight_bit (default 4-bit, fixed precision)
    - Embeddings & LayerNorms: Full precision (unquantized)
    - Weight-only: activations remain in FP32/FP16

    Args:
        model_name: HuggingFace model identifier (e.g., 'usefulsensors/moonshine-base').
        device: Device to load the model on ('cpu' or 'cuda').
        enc_weight_bit: Number of bits for encoder attention/MLP weight quantization.
        dec_weight_bit: Number of bits for decoder weight quantization.
        conv_weight_bit: Number of bits for encoder conv frontend weight quantization.
        use_scaling: Whether to use learnable scaling factors per precision.
        quant_mode: Quantization mode ('symmetric' or 'asymmetric').
        quant_decoder: Whether to quantize decoder layers.

    Returns:
        QuantizedMoonshine model ready for training or evaluation.
    """
    from transformers import AutoModel

    # Load HuggingFace model
    hf_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    # Convert to custom Moonshine architecture using the same conversion
    # logic as run_moonshine.py
    config = hf_model.config

    # Extract rope_theta: may be a top-level attr or nested inside rope_parameters
    rope_theta = getattr(config, 'rope_theta', None)
    if rope_theta is None:
        rope_params = getattr(config, 'rope_parameters', None)
        if rope_params and isinstance(rope_params, dict):
            rope_theta = rope_params.get('rope_theta', 10000.0)
        else:
            rope_theta = 10000.0

    model_dims = MoonshineModelDimensions(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        n_audio_head=config.encoder_num_attention_heads,
        n_audio_layer=config.encoder_num_hidden_layers,
        n_vocab=config.vocab_size,
        n_text_head=config.decoder_num_attention_heads,
        n_text_layer=config.decoder_num_hidden_layers,
        n_text_ctx=config.max_position_embeddings,
        head_dim=getattr(config, 'head_dim', config.hidden_size // config.encoder_num_attention_heads),
        partial_rotary_factor=config.partial_rotary_factor,
        rope_theta=rope_theta,
        pad_head_dim_to_multiple_of=getattr(config, 'pad_head_dim_to_multiple_of', 8),
    )

    low_rank_config = getattr(config, 'low_rank_config', None)

    moonshine_model = Moonshine(
        model_dims,
        low_rank_config=low_rank_config,
        bs=1,
        device=device,
    )

    # Build state dict translation (same as run_moonshine.py convert_from_hf_moonshine)
    reverse_translation = OrderedDict({
        # Encoder conv layers
        r"^encoder\.conv1\.weight$": r"encoder.conv1.weight",
        r"^encoder\.conv2\.(\w+)$": r"encoder.conv2.\1",
        r"^encoder\.conv3\.(\w+)$": r"encoder.conv3.\1",
        r"^encoder\.groupnorm\.(\w+)$": r"encoder.group_norm.\1",
        r"^encoder\.layer_norm\.weight$": r"encoder.ln_post.weight",

        # Encoder transformer layers
        r"^encoder\.layers\.(\d+)\.input_layernorm\.weight$": r"encoder.blocks.\1.input_layernorm.weight",
        r"^encoder\.layers\.(\d+)\.post_attention_layernorm\.weight$": r"encoder.blocks.\1.post_attention_layernorm.weight",
        r"^encoder\.layers\.(\d+)\.self_attn\.q_proj\.(\w+)$": r"encoder.blocks.\1.self_attn.query.\2",
        r"^encoder\.layers\.(\d+)\.self_attn\.k_proj\.(\w+)$": r"encoder.blocks.\1.self_attn.key.\2",
        r"^encoder\.layers\.(\d+)\.self_attn\.v_proj\.(\w+)$": r"encoder.blocks.\1.self_attn.value.\2",
        r"^encoder\.layers\.(\d+)\.self_attn\.o_proj\.(\w+)$": r"encoder.blocks.\1.self_attn.out.\2",
        r"^encoder\.layers\.(\d+)\.mlp\.fc1\.(\w+)$": r"encoder.blocks.\1.fc1.\2",
        r"^encoder\.layers\.(\d+)\.mlp\.fc2\.(\w+)$": r"encoder.blocks.\1.fc2.\2",

        # Decoder embedding and norm
        r"^decoder\.embed_tokens\.weight$": r"decoder.token_embedding.weight",
        r"^decoder\.norm\.weight$": r"decoder.ln.weight",

        # Decoder self-attention
        r"^decoder\.layers\.(\d+)\.input_layernorm\.weight$": r"decoder.blocks.\1.input_layernorm.weight",
        r"^decoder\.layers\.(\d+)\.self_attn\.q_proj\.(\w+)$": r"decoder.blocks.\1.self_attn.query.\2",
        r"^decoder\.layers\.(\d+)\.self_attn\.k_proj\.(\w+)$": r"decoder.blocks.\1.self_attn.key.\2",
        r"^decoder\.layers\.(\d+)\.self_attn\.v_proj\.(\w+)$": r"decoder.blocks.\1.self_attn.value.\2",
        r"^decoder\.layers\.(\d+)\.self_attn\.o_proj\.(\w+)$": r"decoder.blocks.\1.self_attn.out.\2",

        # Decoder cross-attention
        r"^decoder\.layers\.(\d+)\.post_attention_layernorm\.weight$": r"decoder.blocks.\1.post_attention_layernorm.weight",
        r"^decoder\.layers\.(\d+)\.encoder_attn\.q_proj\.(\w+)$": r"decoder.blocks.\1.cross_attn.query.\2",
        r"^decoder\.layers\.(\d+)\.encoder_attn\.k_proj\.(\w+)$": r"decoder.blocks.\1.cross_attn.key.\2",
        r"^decoder\.layers\.(\d+)\.encoder_attn\.v_proj\.(\w+)$": r"decoder.blocks.\1.cross_attn.value.\2",
        r"^decoder\.layers\.(\d+)\.encoder_attn\.o_proj\.(\w+)$": r"decoder.blocks.\1.cross_attn.out.\2",

        # Decoder MLP
        r"^decoder\.layers\.(\d+)\.final_layernorm\.weight$": r"decoder.blocks.\1.final_layernorm.weight",
        r"^decoder\.layers\.(\d+)\.mlp\.fc1\.(\w+)$": r"decoder.blocks.\1.fc1.\2",
        r"^decoder\.layers\.(\d+)\.mlp\.fc2\.(\w+)$": r"decoder.blocks.\1.fc2.\2",
    })

    new_state_dict = {}
    hf_state_dict = hf_model.state_dict()
    unmatched_keys = []

    for key, value in hf_state_dict.items():
        # Strip the 'model.' prefix if present (some HF model variants add it)
        stripped_key = key[len("model."):] if key.startswith("model.") else key

        matched = False
        for pattern, replacement in reverse_translation.items():
            if re.match(pattern, stripped_key):
                new_key = re.sub(pattern, replacement, stripped_key)
                # Transpose weight1 and weight2 for low-rank layers
                if stripped_key.endswith("weight1") or stripped_key.endswith("weight2"):
                    value = value.T.contiguous()
                new_state_dict[new_key] = value
                matched = True
                break

        if not matched:
            unmatched_keys.append(key)

    if unmatched_keys:
        warnings.warn(
            f"The following {len(unmatched_keys)} HF state dict key(s) were not matched "
            f"during conversion and will be skipped: {unmatched_keys}"
        )

    moonshine_model.load_state_dict(new_state_dict, strict=True)

    # Clean up HF model
    del hf_model

    # Move to device
    moonshine_model = moonshine_model.to(device)

    # Wrap with quantization
    quantized_model = QuantizedMoonshine(
        model=moonshine_model,
        enc_weight_bit=enc_weight_bit,
        dec_weight_bit=dec_weight_bit,
        conv_weight_bit=conv_weight_bit,
        use_scaling=use_scaling,
        quant_mode=quant_mode,
        quant_decoder=quant_decoder,
    )

    return quantized_model
