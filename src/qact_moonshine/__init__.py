"""QACT quantization primitives adapted for Moonshine ASR models."""

from .quant_utils import (
    symmetric_linear_quantization_params,
    asymmetric_linear_quantization_params,
    uniform_quantization,
    uniform_truncate,
    ScaleSymmetricLinearQuant_old,
    UniformSymmetricLinearQuant,
    UniformSymmetricTruncate,
    InplaceSymmetricLinearQuant,
)
from .quant_modules import QuantLinear, QuantConv1d, QuantConv2d, SwitchLayerNorm, QuantModule
from .losses import LabelSmoothingLoss, LabelSoftLoss
from .quant_moonshine import QuantizedMoonshine, load_pretrained_moonshine
