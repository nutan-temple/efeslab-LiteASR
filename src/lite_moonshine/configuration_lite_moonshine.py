from transformers import MoonshineConfig

class LiteMoonshineConfig(MoonshineConfig):
    model_type = "lite-moonshine"

    def __init__(
        self, 
        low_rank_config: list[dict[str, int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.low_rank_config = low_rank_config
