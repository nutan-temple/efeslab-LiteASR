"""
Evaluate a QACT-trained quantized Moonshine model on LibriSpeech.

Loads a trained quantized checkpoint, sets a specific precision level
(configurable: 1-bit, 2-bit, or various 1.5-bit mixed configurations),
and evaluates Word Error Rate (WER).

Precision codes (matching QACT's encode() convention):
    1  = all 1-bit
    2  = all 2-bit
    11 = first half layers 1-bit, second half 2-bit (1.5-bit avg)
    12 = first half layers 2-bit, second half 1-bit (1.5-bit avg)
    13 = outer layers 1-bit, middle layers 2-bit
    14 = outer layers 2-bit, middle layers 1-bit
    15 = alternating 1-bit/2-bit
    0  = random half at 1-bit, half at 2-bit

Usage:
    python -m qact_moonshine.eval_qact_moonshine \\
        --checkpoint /path/to/checkpoint.pt \\
        --precision 2 \\
        --max-samples 100
"""

import argparse
import logging
import os
import sys
from typing import List

import numpy as np
import torch

# Ensure src/ is on the path for sibling module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qact_moonshine.quant_moonshine import QuantizedMoonshine, load_pretrained_moonshine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args(args=None):
    """Parse command-line arguments for evaluation.

    Args:
        args: Optional list of argument strings (for testing). If None, uses sys.argv.

    Returns:
        argparse.Namespace with evaluation settings.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate QACT-trained Moonshine model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to QACT training checkpoint (.pt file)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="usefulsensors/moonshine-base",
        help="HuggingFace model name (must match training)",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=2,
        help=(
            "Precision level for evaluation. Codes: "
            "1=all 1-bit, 2=all 2-bit, "
            "11=first-half 1-bit/second-half 2-bit, "
            "12=first-half 2-bit/second-half 1-bit, "
            "13=outer 1-bit/middle 2-bit, "
            "14=outer 2-bit/middle 1-bit, "
            "15=alternating, 0=random"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="librispeech_asr",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="clean",
        help="Dataset configuration/subset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Evaluation batch size (1 for greedy decoding)",
    )

    return parser.parse_args(args)


def precision_code_to_list(precision: int, num_layers: int) -> List[int]:
    """Convert a precision code to a per-layer precision list.

    Args:
        precision: Precision code (1, 2, 11, 12, 13, 14, 15, or 0).
        num_layers: Number of encoder layers.

    Returns:
        List of per-layer precision values.
    """
    half = num_layers // 2
    third = num_layers // 3

    if precision == 1:
        return [1] * num_layers
    elif precision == 2:
        return [2] * num_layers
    elif precision == 11:
        # First half 1-bit, second half 2-bit
        return [1] * half + [2] * (num_layers - half)
    elif precision == 12:
        # First half 2-bit, second half 1-bit
        return [2] * half + [1] * (num_layers - half)
    elif precision == 13:
        # Outer layers 1-bit, middle layers 2-bit
        return [1] * third + [2] * (num_layers - 2 * third) + [1] * third
    elif precision == 14:
        # Outer layers 2-bit, middle layers 1-bit
        return [2] * third + [1] * (num_layers - 2 * third) + [2] * third
    elif precision == 15:
        # Alternating 1-bit/2-bit
        return [1 if i % 2 == 0 else 2 for i in range(num_layers)]
    elif precision == 0:
        # Random: half at 1-bit, half at 2-bit
        idx = np.random.choice(num_layers, num_layers // 2, replace=False)
        prec_list = [2] * num_layers
        for i in idx:
            prec_list[i] = 1
        logger.info(f"Random precision: {prec_list}")
        return prec_list
    else:
        raise ValueError(
            f"Unknown precision code: {precision}. "
            f"Valid codes: 0, 1, 2, 11, 12, 13, 14, 15"
        )


def load_quantized_model(checkpoint_path, model_name, device):
    """Load a QACT-trained quantized model from checkpoint.

    Args:
        checkpoint_path: Path to the saved checkpoint.
        model_name: HuggingFace model name used during training.
        device: Device to load on.

    Returns:
        Tuple of (quantized_model, training_args_dict).
    """
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get training args from checkpoint
    train_args = checkpoint.get("args", {})

    # Recreate the quantized model with the same configuration
    quantized_model = load_pretrained_moonshine(
        model_name=model_name,
        device=str(device),
        weight_bit=train_args.get("enc_weight_bit", 2),
        use_scaling=train_args.get("use_scaling", True),
        quant_mode=train_args.get("quant_mode", "symmetric"),
        quant_decoder=train_args.get("quant_decoder", False),
    )

    # Load trained weights
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    quantized_model.load_state_dict(state_dict, strict=True)

    return quantized_model, train_args


def evaluate_wer(model, tokenizer, dataset, precision_list, device, max_samples=None):
    """Evaluate WER on a dataset at a specific precision level.

    Args:
        model: QuantizedMoonshine model.
        tokenizer: HuggingFace tokenizer.
        dataset: HuggingFace dataset with audio and text.
        precision_list: Per-layer precision configuration.
        device: Computation device.
        max_samples: Optional limit on number of samples.

    Returns:
        WER as a float (0-1 range).
    """
    import evaluate as hf_evaluate
    from tqdm import tqdm

    model.eval()
    model.set_layerwise_precision(precision_list)
    model.set_encoder_conv_precision(precision_list[0])

    wer_metric = hf_evaluate.load("wer")
    all_predictions = []
    all_references = []

    num_samples = len(dataset)
    if max_samples is not None:
        num_samples = min(num_samples, max_samples)

    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Evaluating"):
            sample = dataset[i]
            audio = sample["audio"]["array"].astype(np.float32)
            ref_text = sample.get("text", "")

            if not ref_text.strip():
                continue

            # Encode audio
            waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
            encoder_output = model.encode(waveform)

            # Greedy decoding
            decoder = model.model.decoder
            # Start with BOS token
            bos_token_id = tokenizer.bos_token_id
            if bos_token_id is None:
                bos_token_id = tokenizer.pad_token_id or 0

            generated_ids = [bos_token_id]
            max_len = 256

            for _ in range(max_len):
                input_ids = torch.tensor(
                    [generated_ids], dtype=torch.long, device=device
                )
                logits = decoder(input_ids, encoder_output)

                # Take the last token's logits
                if logits.dim() == 3:
                    next_token_logits = logits[:, -1, :]
                else:
                    next_token_logits = logits[-1:, :]

                next_token = torch.argmax(next_token_logits, dim=-1).item()

                if next_token == tokenizer.eos_token_id:
                    break

                generated_ids.append(next_token)

            # Decode tokens to text
            pred_text = tokenizer.decode(
                generated_ids[1:], skip_special_tokens=True
            )

            all_predictions.append(pred_text.lower().strip())
            all_references.append(ref_text.lower().strip())

    if not all_references:
        logger.warning("No valid samples found for evaluation.")
        return 0.0

    wer = wer_metric.compute(references=all_references, predictions=all_predictions)
    return wer


def main(args=None):
    """Main evaluation entry point.

    Args:
        args: Optional parsed args. If None, parses from sys.argv.
    """
    if args is None:
        args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    from transformers import AutoTokenizer

    quantized_model, train_args = load_quantized_model(
        args.checkpoint, args.model_name, device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Determine number of encoder layers
    num_layers = len(quantized_model.encoder_layer_quant_modules)

    # Convert precision code to per-layer list
    precision_list = precision_code_to_list(args.precision, num_layers)
    avg_bits = sum(precision_list) / len(precision_list)
    logger.info(
        f"Precision config (code={args.precision}): {precision_list} "
        f"(avg {avg_bits:.2f} bits)"
    )

    # Load dataset
    from datasets import load_dataset, Audio

    logger.info(
        f"Loading dataset: {args.dataset} (config: {args.dataset_config}, "
        f"split: {args.split})"
    )
    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    if args.max_samples is not None and args.max_samples < len(dataset):
        dataset = dataset.select(range(args.max_samples))

    # Evaluate
    wer = evaluate_wer(
        model=quantized_model,
        tokenizer=tokenizer,
        dataset=dataset,
        precision_list=precision_list,
        device=device,
        max_samples=args.max_samples,
    )
    wer_pct = round(100 * wer, 2)

    # Print results
    print("\n" + "=" * 60)
    print("QACT EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model:           {args.model_name}")
    print(f"Checkpoint:      {args.checkpoint}")
    print(f"Precision code:  {args.precision}")
    print(f"Precision list:  {precision_list}")
    print(f"Average bits:    {avg_bits:.2f}")
    print(f"Dataset:         {args.dataset} ({args.dataset_config}, {args.split})")
    print(f"Samples:         {args.max_samples or 'all'}")
    print(f"Word Error Rate: {wer_pct}%")
    print("=" * 60)

    return wer_pct


if __name__ == "__main__":
    main()
