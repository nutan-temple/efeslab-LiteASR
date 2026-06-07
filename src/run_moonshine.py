# Running inference for the compressed Moonshine models.
# This code is for demonstration purposes only, and longer audio, custom sampling algorithms,
# or batch inference would require modification of the code.
import argparse
from collections import OrderedDict
import re
import warnings

import torch
from transformers import AutoModel, AutoTokenizer

from moonshine_model import Moonshine, MoonshineModelDimensions
from utils import ffmpeg_read


def convert_from_hf_moonshine(hf_model, device=torch.device("cuda")):
    print("Converting from HuggingFace model to Moonshine model...")
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

    config = hf_model.config

    model_dims = MoonshineModelDimensions(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        n_audio_head=config.encoder_num_attention_heads,
        n_audio_layer=config.encoder_num_hidden_layers,
        n_vocab=config.vocab_size,
        n_text_head=config.decoder_num_attention_heads,
        n_text_layer=config.decoder_num_hidden_layers,
        n_text_ctx=config.max_position_embeddings,
        head_dim=config.head_dim,
        partial_rotary_factor=config.partial_rotary_factor,
        rope_theta=config.rope_theta,
        pad_head_dim_to_multiple_of=8,
    )

    low_rank_config = config.low_rank_config

    new_state_dict = {}
    new_model = Moonshine(
        model_dims,
        low_rank_config=low_rank_config,
        bs=1,
        device=device,
    )

    # Get model state dict, stripping 'model.' prefix from HF model keys
    hf_state_dict = hf_model.state_dict()
    unmatched_keys = []
    for key, value in hf_state_dict.items():
        # Remove the 'model.' prefix that HF adds
        if key.startswith("model."):
            key = key[len("model."):]
        else:
            unmatched_keys.append(key)
            continue

        matched = False
        for pattern, replacement in reverse_translation.items():
            if re.match(pattern, key):
                new_key = re.sub(pattern, replacement, key)
                # Transpose weight1 and weight2 for low-rank layers
                if key.endswith("weight1") or key.endswith("weight2"):
                    value = value.T.contiguous()
                new_state_dict[new_key] = value
                matched = True
                break

        if not matched:
            unmatched_keys.append(f"model.{key}")

    if unmatched_keys:
        warnings.warn(
            f"The following {len(unmatched_keys)} HF state dict key(s) were not matched "
            f"during conversion and will be skipped: {unmatched_keys}"
        )

    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()
    new_model.load_state_dict(new_state_dict, strict=True)

    del hf_model
    print("Conversion complete.")
    return new_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--model", type=str, default="efficient-speech/lite-moonshine-base")
    parser.add_argument("--base-model", type=str, default="usefulsensors/moonshine-base")
    parser.add_argument("--audio-path", type=str, default="audio.wav")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=dtype)

    # Convert the model
    model = convert_from_hf_moonshine(model, device=device)
    model = model.to(device).to(dtype)

    # Capture CUDA Graph (only on CUDA)
    if device != "cpu":
        model.init_cuda_graph()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    audio = ffmpeg_read(args.audio_path, sampling_rate=16_000)

    # Pass eos_token_id from tokenizer to the model
    if tokenizer.eos_token_id is not None:
        model.eos_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        waveform = torch.tensor(audio, dtype=dtype).unsqueeze(0).to(device)
        decoder_input_ids = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).to(device)

        ret_tokens = model.forward(waveform, decoder_input_ids)

        print(tokenizer.decode(ret_tokens, skip_special_tokens=True))
        model.reinit_kv_cache()
