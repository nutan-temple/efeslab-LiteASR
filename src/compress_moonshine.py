import argparse
import numpy as np
import torch
import torch.nn as nn
import evaluate
from normalizer import data_utils
import tqdm
from transformers import MoonshineForConditionalGeneration, AutoTokenizer, AutoFeatureExtractor
from datasets import load_dataset, concatenate_datasets

wer_metric = evaluate.load("wer")
torch.set_float32_matmul_precision('high')


class LinearLowRank(torch.nn.Module):
    def __init__(self, weight1: torch.Tensor, weight2: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.weight1 = nn.Parameter(weight1)
        self.weight2 = nn.Parameter(weight2)
        self.bias = nn.Parameter(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.weight1) @ self.weight2 + self.bias


def transcribe(model, tokenizer, audio, feature_extractor):
    """Transcribe audio using Moonshine model.generate()."""
    inputs = feature_extractor(
        audio, sampling_rate=16000, return_tensors="pt"
    )
    input_values = inputs["input_values"].to(model.device)
    generated_ids = model.generate(input_values=input_values)
    text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text


def apply_low_rank(model, dataset, rank_threshold, tokenizer, feature_extractor, device):
    """Apply low-rank approximation to Moonshine encoder layers using PCA/SVD.

    Uses forward hooks to collect activations from encoder linear layers during
    calibration, then decomposes each layer using SVD and replaces with LinearLowRank.
    """
    base_dim = model.config.hidden_size  # 416 for moonshine-base

    # Storage for collected activations
    # calibration_data[layer_idx][component_name] = list of tensors
    num_layers = model.config.num_hidden_layers
    calibration_data = {
        i: {"q_proj": [], "k_proj": [], "v_proj": [], "o_proj": [], "fc1": [], "fc2": []}
        for i in range(num_layers)
    }

    # Register forward hooks on encoder linear layers to capture activations
    hooks = []
    encoder_layers = model.model.encoder.layers

    for i_layer in range(num_layers):
        layer = encoder_layers[i_layer]

        def make_hook(layer_idx, component_name):
            def hook_fn(module, input, output):
                # Capture output activation (detached, moved to cpu to save memory)
                calibration_data[layer_idx][component_name].append(
                    output.detach().cpu()
                )
            return hook_fn

        hooks.append(layer.self_attn.q_proj.register_forward_hook(
            make_hook(i_layer, "q_proj")))
        hooks.append(layer.self_attn.k_proj.register_forward_hook(
            make_hook(i_layer, "k_proj")))
        hooks.append(layer.self_attn.v_proj.register_forward_hook(
            make_hook(i_layer, "v_proj")))
        hooks.append(layer.self_attn.o_proj.register_forward_hook(
            make_hook(i_layer, "o_proj")))
        hooks.append(layer.mlp.fc1.register_forward_hook(
            make_hook(i_layer, "fc1")))
        hooks.append(layer.mlp.fc2.register_forward_hook(
            make_hook(i_layer, "fc2")))

    # Run calibration: process each sample to collect activations
    model.eval()
    with torch.no_grad():
        for i in tqdm.tqdm(range(len(dataset)), desc="Calibration"):
            audio = dataset[i]["audio"]["array"].astype(np.float32)
            transcribe(model, tokenizer, audio, feature_extractor)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Apply SVD decomposition and replace layers
    component_names = ["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"]

    for i_layer in tqdm.tqdm(range(num_layers), desc="Compressing layers"):
        layer = encoder_layers[i_layer]

        for i, comp_name in enumerate(component_names):
            # Get the linear module
            if comp_name == "q_proj":
                linear_module = layer.self_attn.q_proj
            elif comp_name == "k_proj":
                linear_module = layer.self_attn.k_proj
            elif comp_name == "v_proj":
                linear_module = layer.self_attn.v_proj
            elif comp_name == "o_proj":
                linear_module = layer.self_attn.o_proj
            elif comp_name == "fc1":
                linear_module = layer.mlp.fc1
            elif comp_name == "fc2":
                linear_module = layer.mlp.fc2

            # Concatenate collected activations
            features = torch.cat(
                calibration_data[i_layer][comp_name],
                dim=0,
            )
            # Reshape to 2D: (total_tokens, out_features)
            if features.dim() == 3:
                features = features.reshape(-1, features.shape[-1])

            # Determine threshold
            if ":" in rank_threshold:
                # Different thresholds for self-attention and MLP layers
                thresh = float(rank_threshold.split(":")[0]) if i <= 3 else float(rank_threshold.split(":")[1])
            else:
                thresh = float(rank_threshold)

            features = features.to(device).float()
            Y_mean = features.mean(dim=0, keepdim=False).to(device)

            # Center the data
            features_centered = features - Y_mean

            # Compute SVD (X_centered = U * S * Vt)
            U, S, Vt = torch.linalg.svd(features_centered, full_matrices=False)

            # Determine k: smallest multiple of 16 that satisfies accuracy constraint
            S_F = S ** 2

            k = -1
            for j in range(0, len(S_F), 16):
                if S_F[:j].sum() / S_F.sum() > thresh:
                    k = j
                    break
            print(f"Layer {i_layer}, Component {comp_name}, k = {k} / {len(S_F)}")

            # If k is too big, no benefit in using low rank approximation
            if i <= 3:
                if k > 0.5 * base_dim or k == -1:
                    continue
            else:
                if k > 0.8 * base_dim or k == -1:
                    continue

            V = Vt.T
            V_k = V[:, :k]  # top k principal components

            V_k = V_k.to(device)
            W = linear_module.weight.T.to(device).float()
            w1 = W @ V_k
            w2 = V_k.T

            # Handle bias based on whether the original layer has bias
            if linear_module.bias is None:
                # No bias in original (attention projections)
                bias = Y_mean - Y_mean @ V_k @ V_k.T
            else:
                # Has bias (MLP fc1, fc2)
                original_bias = linear_module.bias.to(device).float()
                bias = Y_mean + (original_bias - Y_mean) @ V_k @ V_k.T

            # Create low-rank replacement
            new_layer = LinearLowRank(
                w1.to(linear_module.weight.dtype).to(device),
                w2.to(linear_module.weight.dtype).to(device),
                bias.to(linear_module.weight.dtype).to(device),
            )

            # Replace the linear module
            if comp_name == "q_proj":
                layer.self_attn.q_proj = new_layer
            elif comp_name == "k_proj":
                layer.self_attn.k_proj = new_layer
            elif comp_name == "v_proj":
                layer.self_attn.v_proj = new_layer
            elif comp_name == "o_proj":
                layer.self_attn.o_proj = new_layer
            elif comp_name == "fc1":
                layer.mlp.fc1 = new_layer
            elif comp_name == "fc2":
                layer.mlp.fc2 = new_layer

            # Free memory
            features = features.cpu()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MoonshineForConditionalGeneration.from_pretrained(args.base_model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.base_model)

    eval_dataset_list = []
    calibrate_dataset = []
    benchs = [
        "voxpopuli",
        "ami",
        "earnings22",
        "gigaspeech",
        "librispeech:test.clean",
        "librispeech:test.other",
        "spgispeech",
        "tedlium",
    ]

    # Prepare calibration and test datasets from ESB benchmarks
    for bench in benchs:
        split = "test"
        if ":" in bench:
            bench, split = bench.split(":")
        print(bench, split)
        dataset = load_dataset(
            "hf-audio/esb-datasets-test-only-sorted",
            bench,
            split=split,
            streaming=False,
            token=True,
        )
        dataset = dataset.shuffle(seed=42)
        dataset = data_utils.prepare_data(dataset)
        if args.max_eval_samples is None:
            calibrate_dataset.append(dataset.select(range(args.num_calibration_samples)))
            eval_dataset_list.append(dataset.select(range(
                args.num_calibration_samples,
                len(dataset),
            )))
        else:
            eval_dataset_list.append(dataset.select(range(args.max_eval_samples)))
            calibrate_dataset.append(dataset.select(range(
                args.max_eval_samples,
                args.max_eval_samples + args.num_calibration_samples,
            )))

    calibrate_dataset = concatenate_datasets(calibrate_dataset).shuffle(seed=42).select(range(args.num_calibration_samples))

    if args.low_rank:
        model = apply_low_rank(
            model, calibrate_dataset, args.rank_threshold,
            tokenizer, feature_extractor, device,
        )

        if args.save_weight:
            torch.save(model.state_dict(), f"lite-moonshine-{args.base_model.split('/')[-1]}_{args.rank_threshold}.pth")

    print(f"Number of parameters in encoder: {sum(p.numel() for p in model.model.encoder.parameters())}")

    if not args.do_eval:
        return

    # Accuracy benchmark
    total_sum_wer = 0
    for i_bench, dataset in enumerate(eval_dataset_list):
        sum_wer = 0
        wer_metric = evaluate.load("wer")
        for i in range(len(dataset)):
            audio = dataset[i]["audio"]["array"].astype(np.float32)

            pred = transcribe(model, tokenizer, audio, feature_extractor)
            wer = wer_metric.compute(
                references=[dataset[i]["norm_text"]], predictions=[data_utils.normalizer(pred)]
            )
            wer = round(100 * wer, 2)
            sum_wer += wer

        with open('output.txt', 'a') as f:
            f.write(f"Number of parameters: {sum(p.numel() for p in model.model.encoder.parameters())}\n")
            f.write(str(args) + '\n')
            f.write(f"{benchs[i_bench]}\n")
            f.write(f"Average WER: {sum_wer / len(dataset)}\n")
            f.write("=====================================\n")
        total_sum_wer += sum_wer / len(dataset)

    total_avg_wer = total_sum_wer / len(eval_dataset_list)
    model_name = ('lite-' if args.low_rank else '') + 'moonshine-' + args.base_model.split('/')[-1] + ('-' + args.rank_threshold if args.low_rank else '')
    with open('output.txt', 'a') as f:
        f.write(f'Final model evaluation results:\n')
        f.write(f'max_eval_samples:{args.max_eval_samples} | {model_name} | total_avg_wer: {total_avg_wer} | encoder_params: {sum(p.numel() for p in model.model.encoder.parameters())} | decoder_params: {sum(p.numel() for p in model.model.decoder.parameters())}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_model",
        type=str,
        default="usefulsensors/moonshine-base",
        help="Base model name on HuggingFace. E.g. 'usefulsensors/moonshine-base'.",
    )
    parser.add_argument(
        "--low_rank",
        action="store_true",
        help="Whether to apply low-rank approximation for the activations",
    )
    parser.add_argument(
        "--rank_threshold",
        type=str,
        default="0.99:0.999",
        help="The threshold for the rank approximation. If two values separated by ':', the first is for self-attention layers and the second is for MLP layers",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=1000,
        help="Number of samples to be evaluated. Put a lower number e.g. 64 for testing this script.",
    )
    parser.add_argument(
        "--num_calibration_samples",
        type=int,
        default=100,
        help="Number of samples to be used for calibration",
    )
    parser.add_argument(
        "--save_weight",
        action="store_true",
        help="Whether to save the compressed weights",
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="Whether to evaluate the model",
    )
    args = parser.parse_args()

    main(args)
