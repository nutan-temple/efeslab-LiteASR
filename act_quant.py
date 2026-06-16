"""
Dynamic INT8 activation quantization for true W8A8 inference.

This module adds the "A8" half of dynamic W8A8 on top of the SpQR weight
quantization ("W8") performed by `quantize_encoder_spqr.py`. After the encoder
weights have been SpQR-quantized (int8-quantized then dequantized in place),
each quantized sublayer can be wrapped with `ActQuantWrapper`, which quantizes
the *activations* entering every matmul to int8 dynamically (at runtime, with a
scale recomputed on each forward pass) and dequantizes them, so the matmul
simulates an int8 x int8 product.

Dynamic vs. static
------------------
The activation scale is computed *per forward pass* from the activation itself
(`x.abs().amax(...) / 127`), so no calibration set or stored activation
statistics are needed. This is the standard accurate dynamic scheme.

Granularity
-----------
* ``per_token`` (default, recommended): one scale per token position, computed
  over the last/hidden dimension. This matches the per-row granularity of
  activation matrices and is markedly more accurate than per-tensor.
* ``per_tensor``: a single scale for the whole activation tensor.

Both are symmetric (zero-point = 0), which is the right choice for the roughly
zero-centered activations of a transformer and keeps the int8 matmul cheap.

LinearLowRank handling
----------------------
A `LinearLowRank` layer computes ``(x @ weight1) @ weight2 + bias`` -- i.e. two
matmuls. A faithful W8A8 must quantize the activation entering *each* matmul, so
the wrapper quantizes ``x`` before ``@ weight1`` and quantizes the intermediate
``x @ weight1`` before ``@ weight2``. For a plain ``nn.Linear`` the wrapper
quantizes ``x`` and calls ``F.linear``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modelutils import LinearLowRank


def dynamic_quantize_dequantize(x, granularity="per_token"):
    """Symmetric dynamic int8 quantize -> dequantize of an activation tensor.

    Args:
        x: activation tensor; the last dim is the hidden/feature dim.
        granularity: "per_token" (scale over last dim, one per token) or
            "per_tensor" (a single scale for the whole tensor).

    Returns:
        x_dq: tensor of the same shape/dtype as x, equal to
            ``round(x / scale).clamp(-127, 127) * scale`` -- i.e. x carrying the
            int8 quantization error, so a subsequent matmul simulates int8 x int8.
    """
    if granularity == "per_tensor":
        scale = x.abs().amax() / 127.0
    else:  # per_token: one scale per row of the last dim
        scale = x.abs().amax(dim=-1, keepdim=True) / 127.0
    scale = scale.clamp(min=1e-8)
    x_int8 = torch.clamp(torch.round(x / scale), -127, 127)
    return x_int8 * scale


class ActQuantWrapper(nn.Module):
    """Wrap an (already weight-quantized) layer to add dynamic int8 activations.

    Supports both ``nn.Linear`` and LiteASR's ``LinearLowRank``. The wrapped
    layer's weights are used as-is (they are already SpQR-quantized in place);
    this wrapper only injects activation quantization before each matmul.
    """

    def __init__(self, layer, granularity="per_token"):
        super().__init__()
        self.layer = layer
        self.granularity = granularity
        self.is_lowrank = isinstance(layer, LinearLowRank)

    def forward(self, x):
        if self.is_lowrank:
            # (x @ weight1) @ weight2 + bias, quantizing the activation that
            # enters each of the two matmuls.
            xq = dynamic_quantize_dequantize(x, self.granularity)
            h = xq @ self.layer.weight1
            hq = dynamic_quantize_dequantize(h, self.granularity)
            return hq @ self.layer.weight2 + self.layer.bias
        # nn.Linear: quantize x, then standard linear with the (quantized) weight.
        xq = dynamic_quantize_dequantize(x, self.granularity)
        return F.linear(xq, self.layer.weight, self.layer.bias)


def wrap_encoder_activations(model, granularity="per_token", max_layers=None):
    """Wrap every quantized encoder sublayer with ``ActQuantWrapper``.

    Walks ``model.model.encoder.layers`` and replaces each target sublayer
    (q_proj, k_proj, v_proj, o_proj, fc1, fc2) with an ``ActQuantWrapper`` around
    it. Call this *after* SpQR weight quantization so the wrapped weights are
    already int8.

    Returns the number of sublayers wrapped.
    """
    component_paths = [
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.fc1", "mlp.fc2",
    ]
    layers = model.model.encoder.layers
    n_layers = len(layers) if max_layers is None else min(max_layers, len(layers))

    wrapped = 0
    for i in range(n_layers):
        layer = layers[i]
        for path in component_paths:
            try:
                sublayer = _getattr_nested(layer, path)
            except AttributeError:
                continue
            if isinstance(sublayer, ActQuantWrapper):
                continue
            if isinstance(sublayer, (nn.Linear, LinearLowRank)):
                _setattr_nested(layer, path, ActQuantWrapper(sublayer, granularity))
                wrapped += 1
    return wrapped


def _getattr_nested(obj, path):
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _setattr_nested(obj, path, value):
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)
