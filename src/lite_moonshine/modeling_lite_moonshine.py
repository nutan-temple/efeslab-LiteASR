import torch
import torch.utils.checkpoint
from torch import nn
from transformers.models.moonshine.configuration_moonshine import MoonshineConfig
from transformers.models.moonshine.modeling_moonshine import (
    MoonshineEncoderLayer,
    MoonshineEncoder,
    MoonshineModel,
    MoonshineForConditionalGeneration,
)

from .configuration_lite_moonshine import LiteMoonshineConfig


class LinearLowRank(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        low_rank_features: int,
    ):
        super().__init__()

        self.weight1 = nn.Parameter(torch.randn(in_features, low_rank_features))
        self.weight2 = nn.Parameter(torch.randn(low_rank_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.weight1) @ self.weight2 + self.bias


class LiteMoonshineEncoderLayer(MoonshineEncoderLayer):
    def __init__(self, config: MoonshineConfig, layer_idx: int, low_rank_config: dict[str, int]):
        super().__init__(config, layer_idx)

        if "q_proj" in low_rank_config:
            self.self_attn.q_proj = LinearLowRank(config.hidden_size, config.hidden_size, low_rank_config["q_proj"])

        if "k_proj" in low_rank_config:
            self.self_attn.k_proj = LinearLowRank(config.hidden_size, config.hidden_size, low_rank_config["k_proj"])

        if "v_proj" in low_rank_config:
            self.self_attn.v_proj = LinearLowRank(config.hidden_size, config.hidden_size, low_rank_config["v_proj"])

        if "o_proj" in low_rank_config:
            self.self_attn.o_proj = LinearLowRank(config.hidden_size, config.hidden_size, low_rank_config["o_proj"])

        if "fc1" in low_rank_config:
            self.mlp.fc1 = LinearLowRank(config.hidden_size, config.intermediate_size, low_rank_config["fc1"])

        if "fc2" in low_rank_config:
            self.mlp.fc2 = LinearLowRank(config.intermediate_size, config.hidden_size, low_rank_config["fc2"])


class LiteMoonshineEncoder(MoonshineEncoder):
    def __init__(self, config: MoonshineConfig, low_rank_config: list[dict[str, int]]):
        super().__init__(config)

        self.layers = nn.ModuleList([
            LiteMoonshineEncoderLayer(config, i, low_rank_config[i])
            for i in range(config.encoder_num_hidden_layers)
        ])


class LiteMoonshineModel(MoonshineModel):
    def __init__(self, config: MoonshineConfig, low_rank_config: list[dict[str, int]]):
        super().__init__(config)

        self.encoder = LiteMoonshineEncoder(config, low_rank_config)


class LiteMoonshineForConditionalGeneration(MoonshineForConditionalGeneration):
    config_class = LiteMoonshineConfig

    def __init__(self, config: LiteMoonshineConfig):
        low_rank_config = getattr(config, "low_rank_config", None)

        super().__init__(config)
        self.model = LiteMoonshineModel(config, low_rank_config)
