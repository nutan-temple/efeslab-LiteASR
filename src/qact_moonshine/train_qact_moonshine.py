"""
QACT Co-Training Script for Moonshine ASR.

Implements multi-precision co-training with stochastic precision scheduling
and knowledge distillation (KD) loss between precision levels.

Supports PyTorch DistributedDataParallel (DDP) for multi-GPU training.
Launch with torchrun for multi-GPU:
    torchrun --nproc_per_node=8 -m qact_moonshine.train_qact_moonshine \\
        --output-dir /path/to/checkpoints

The co-training loop performs 3 forward passes per batch:
1. 2-bit precision: standard CE loss, stores soft targets (detached softmax of logits)
2. 1-bit precision: KD loss = lambda_2 * LabelSoftLoss(logits, soft_targets, hard_targets)
                              + lambda_1 * CE(logits, hard_targets)
3. Stochastic-mixed precision: same KD loss formula as pass 2

Final loss = (loss_1 + loss_2 + loss_3) / 3

Stochastic precision schedule (mix_rate=1.8, log-linear from paper):
    a = numpy.linspace(numpy.exp(0.2), numpy.exp(0.8), num_encoder_layers)
    mix_rate_per_layer = numpy.log(a / 0.9)
    For each layer i: pick 2-bit if random() > mix_rate[i], else 1-bit

Usage:
    # Single GPU:
    python -m qact_moonshine.train_qact_moonshine --output-dir /path/to/checkpoints

    # Multi-GPU (8 GPUs on a single node):
    torchrun --nproc_per_node=8 -m qact_moonshine.train_qact_moonshine \\
        --output-dir /path/to/checkpoints \\
        --enc-weight-bit 2 \\
        --mix-rate 1.8 \\
        --epochs 100 \\
        --batch-size 64
"""

import argparse
import logging
import math
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Ensure src/ is on the path for sibling module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qact_moonshine.quant_moonshine import QuantizedMoonshine, load_pretrained_moonshine
from qact_moonshine.losses import LabelSoftLoss, LabelSmoothingLoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args(args=None):
    """Parse command-line arguments for QACT co-training.

    Args:
        args: Optional list of argument strings (for testing). If None, uses sys.argv.

    Returns:
        argparse.Namespace with all training hyperparameters.
    """
    parser = argparse.ArgumentParser(
        description="QACT Co-Training for Moonshine ASR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model settings
    parser.add_argument(
        "--model-name",
        type=str,
        default="usefulsensors/moonshine-base",
        help="HuggingFace model name for the pretrained Moonshine model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save checkpoints and logs",
    )

    # Quantization settings
    parser.add_argument(
        "--enc-weight-bit",
        type=int,
        default=2,
        help="Default weight bit-width for encoder quantization",
    )
    parser.add_argument(
        "--use-scaling",
        type=bool,
        default=True,
        help="Use learnable scaling factors for quantization",
    )
    parser.add_argument(
        "--quant-mode",
        type=str,
        default="symmetric",
        choices=["symmetric", "asymmetric"],
        help="Quantization mode",
    )
    parser.add_argument(
        "--quant-decoder",
        action="store_true",
        default=False,
        help="Whether to also quantize decoder layers",
    )

    # QACT co-training hyperparameters
    parser.add_argument(
        "--mix-rate",
        type=float,
        default=1.8,
        help="Mix rate for stochastic precision schedule (0, 0.2, 0.8, 1.0, or 1.8)",
    )
    parser.add_argument(
        "--lambda-1",
        type=float,
        default=0.5,
        help="Weight for CE loss in KD passes",
    )
    parser.add_argument(
        "--lambda-2",
        type=float,
        default=1.0,
        help="Weight for soft KD loss in KD passes",
    )

    # Training settings
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Per-GPU training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Peak learning rate",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1000,
        help="Number of warmup steps for learning rate scheduler",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=5.0,
        help="Gradient clipping max norm",
    )
    parser.add_argument(
        "--accum-grad",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor for CE loss",
    )

    # Dataset settings
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
        "--train-split",
        type=str,
        default="train.clean.100",
        help="Dataset split to use for training (e.g. train.clean.100 for LibriSpeech 100hrs)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of training samples (for debugging)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )

    # Logging and saving
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Log training loss every N steps",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    return parser.parse_args(args)


def compute_mix_rate_schedule(mix_rate: float, num_layers: int) -> np.ndarray:
    """Compute the stochastic precision schedule per encoder layer.

    Implements the log-linear schedule from the QACT paper.

    Args:
        mix_rate: The mix_rate hyperparameter (0, 0.2, 0.8, 1.0, or 1.8).
        num_layers: Number of encoder layers.

    Returns:
        numpy array of per-layer mix rates.
    """
    if mix_rate == 1.0:
        return np.linspace(0.5, 0.8, num_layers)
    elif mix_rate == 0.0:
        return np.linspace(0.8, 0.2, num_layers)
    elif mix_rate == 0.8:
        a = np.linspace(np.log(0.2), np.log(0.8), num_layers)
        return np.exp(a / 0.9)
    elif mix_rate == 0.2:
        a = np.linspace(np.log(0.8), np.log(0.2), num_layers)
        return np.exp(a / 0.9)
    elif mix_rate == 1.8:
        # Log-linear schedule (used in the paper)
        a = np.linspace(np.exp(0.2), np.exp(0.8), num_layers)
        return np.log(a / 0.9)
    else:
        raise ValueError(
            f"mix_rate must be 0, 0.2, 0.8, 1.0, or 1.8, got {mix_rate}"
        )


def sample_stochastic_precision(mix_rate_schedule: np.ndarray) -> List[int]:
    """Sample a stochastic precision configuration for one forward pass.

    For each layer i: pick 2-bit if random() > mix_rate[i], else 1-bit.

    Args:
        mix_rate_schedule: Per-layer mix rates from compute_mix_rate_schedule.

    Returns:
        List of precision values (1 or 2) for each layer.
    """
    prec_list = []
    for i in range(len(mix_rate_schedule)):
        if np.random.rand() > mix_rate_schedule[i]:
            prec_list.append(2)
        else:
            prec_list.append(1)
    return prec_list


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Create a cosine learning rate scheduler with linear warmup.

    Args:
        optimizer: The optimizer.
        warmup_steps: Number of warmup steps.
        total_steps: Total number of training steps.

    Returns:
        A lambda LR scheduler.
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class LibriSpeechCollator:
    """Collate function for LibriSpeech that pads waveforms and tokenizes text.

    Args:
        tokenizer: HuggingFace tokenizer for the Moonshine model.
        max_audio_len: Maximum audio length in samples (for truncation).
    """

    def __init__(self, tokenizer, max_audio_len: Optional[int] = None):
        self.tokenizer = tokenizer
        self.max_audio_len = max_audio_len

    def __call__(self, batch):
        """Collate a batch of samples.

        Args:
            batch: List of dataset samples, each with 'audio' and 'text' fields.

        Returns:
            Dictionary with:
                - 'input_values': padded waveform tensor (B, max_audio_len)
                - 'attention_mask': mask for padded positions (B, max_audio_len)
                - 'labels': tokenized text target IDs (B, max_seq_len)
        """
        waveforms = []
        texts = []

        for sample in batch:
            audio_array = sample["audio"]["array"]
            if self.max_audio_len is not None:
                audio_array = audio_array[: self.max_audio_len]
            waveforms.append(torch.tensor(audio_array, dtype=torch.float32))
            texts.append(sample["text"])

        # Pad waveforms to the same length
        max_len = max(w.shape[0] for w in waveforms)
        padded_waveforms = torch.zeros(len(waveforms), max_len)
        attention_mask = torch.zeros(len(waveforms), max_len, dtype=torch.long)

        for i, w in enumerate(waveforms):
            padded_waveforms[i, : w.shape[0]] = w
            attention_mask[i, : w.shape[0]] = 1

        # Tokenize text targets
        tokenized = self.tokenizer(
            texts,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        labels = tokenized["input_ids"]

        return {
            "input_values": padded_waveforms,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_dataset_splits(args):
    """Load and prepare the training dataset.

    Args:
        args: Parsed arguments with dataset configuration.

    Returns:
        HuggingFace dataset object with audio resampled to 16kHz.
    """
    from datasets import load_dataset, Audio

    logger.info(
        f"Loading dataset: {args.dataset} (config: {args.dataset_config}, "
        f"split: {args.train_split})"
    )

    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.train_split,
    )

    # Resample audio to 16kHz
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    if args.max_samples is not None and args.max_samples < len(dataset):
        dataset = dataset.select(range(args.max_samples))

    logger.info(f"Training on {len(dataset)} samples")
    return dataset


def setup_model(args, device):
    """Load the pretrained Moonshine model and wrap with QACT quantization.

    Args:
        args: Parsed arguments with model configuration.
        device: Device to load the model on.

    Returns:
        Tuple of (quantized_model, tokenizer, num_encoder_layers).
    """
    from transformers import AutoTokenizer

    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    quantized_model = load_pretrained_moonshine(
        model_name=args.model_name,
        device=str(device),
        weight_bit=args.enc_weight_bit,
        use_scaling=args.use_scaling,
        quant_mode=args.quant_mode,
        quant_decoder=args.quant_decoder,
    )

    num_encoder_layers = len(quantized_model.encoder_layer_quant_modules)
    logger.info(f"Model loaded with {num_encoder_layers} encoder layers")
    logger.info(
        f"Quantization: {args.enc_weight_bit}-bit, "
        f"use_scaling={args.use_scaling}, mode={args.quant_mode}"
    )

    return quantized_model, tokenizer, num_encoder_layers


def qact_forward_pass(
    model,
    batch,
    device,
    num_encoder_layers,
    mix_rate_schedule,
    lambda_1,
    lambda_2,
    ce_criterion,
    soft_criterion,
    vocab_size,
    pad_token_id,
):
    """Perform the QACT co-training forward pass with 3 sub-passes.

    Pass 1 (2-bit): Standard CE loss, stores soft targets.
    Pass 2 (1-bit): KD loss = lambda_2 * LabelSoftLoss + lambda_1 * CE.
    Pass 3 (stochastic-mixed): Same KD loss formula as pass 2.

    Args:
        model: QuantizedMoonshine model.
        batch: Collated batch dictionary.
        device: Computation device.
        num_encoder_layers: Number of encoder layers.
        mix_rate_schedule: Per-layer stochastic precision schedule.
        lambda_1: Weight for CE component in KD loss.
        lambda_2: Weight for soft KD component.
        ce_criterion: Cross-entropy or label smoothing loss function.
        soft_criterion: LabelSoftLoss instance.
        vocab_size: Vocabulary size for the model.
        pad_token_id: Padding token ID for ignoring in loss.

    Returns:
        Averaged loss over all 3 passes.
    """
    input_values = batch["input_values"].to(device)
    labels = batch["labels"].to(device)

    # Prepare decoder inputs (teacher forcing): shift labels right
    # decoder_input_ids = [pad, token_1, token_2, ..., token_{n-1}]
    # targets = [token_1, token_2, ..., token_n]
    decoder_input_ids = labels[:, :-1].contiguous()
    targets = labels[:, 1:].contiguous()

    # Define the precision configurations for 3 passes
    prec_2bit = [2] * num_encoder_layers
    prec_1bit = [1] * num_encoder_layers
    prec_mixed = sample_stochastic_precision(mix_rate_schedule)

    precision_configs = [prec_2bit, prec_1bit, prec_mixed]
    losses = []
    soft_targets = None

    for pass_idx, prec_list in enumerate(precision_configs):
        # Set encoder precision for this pass
        model.set_layerwise_precision(prec_list)
        model.set_encoder_conv_precision(prec_list[0])

        # Forward pass through encoder
        encoder_output = model.encode(input_values)

        # Forward pass through decoder (teacher forcing)
        # Pass kv_cache=None to avoid copy_() which severs gradients.
        # The no-cache path in MultiHeadAttention computes K/V directly,
        # preserving gradient flow through decoder key/value projections.
        decoder = model.model.decoder
        logits = decoder(
            decoder_input_ids, encoder_output,
            offset=0, kv_cache=None, is_prefilling=True,
        )

        # Reshape logits for loss computation: (B, seq_len, vocab_size)
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)

        if pass_idx == 0:
            # First pass (2-bit): standard CE loss, store soft targets
            loss = ce_criterion(logits, targets)
            soft_targets = torch.softmax(logits, dim=-1).detach()
        else:
            # Subsequent passes (1-bit, mixed): KD loss + CE loss
            kd_loss = soft_criterion(logits, soft_targets, targets)
            ce_loss = ce_criterion(logits, targets)
            loss = lambda_2 * kd_loss + lambda_1 * ce_loss

        losses.append(loss)

    # Average all 3 losses
    total_loss = sum(losses) / len(losses)
    return total_loss


def setup_ddp():
    """Initialize the distributed process group if running under torchrun.

    Returns:
        Tuple of (rank, local_rank, world_size, is_distributed).
        If not running in distributed mode, returns (0, 0, 1, False).
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size, True
    else:
        return 0, 0, 1, False


def cleanup_ddp():
    """Destroy the distributed process group if it was initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()


def train(args):
    """Main training loop for QACT co-training on Moonshine.

    Supports both single-GPU and multi-GPU (DDP) training. When launched
    with torchrun, automatically uses DistributedDataParallel.

    Args:
        args: Parsed arguments from parse_args().
    """
    # Setup distributed training
    rank, local_rank, world_size, is_distributed = setup_ddp()

    # Set random seed (offset by rank for different data ordering per GPU)
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    # Device setup
    if is_distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        logger.info(f"Using device: {device}")
        if is_distributed:
            logger.info(
                f"Distributed training: {world_size} GPUs, "
                f"effective batch size = {args.batch_size * world_size}"
            )

    # Create output directory (only on rank 0)
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    # Synchronize all processes before proceeding
    if is_distributed:
        dist.barrier()

    # Load model and tokenizer
    quantized_model, tokenizer, num_encoder_layers = setup_model(args, device)
    quantized_model.train()

    # Wrap with DDP if distributed
    if is_distributed:
        quantized_model = DDP(
            quantized_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
        # Access the underlying model for set_layerwise_precision etc.
        raw_model = quantized_model.module
    else:
        raw_model = quantized_model

    # Compute stochastic precision schedule
    mix_rate_schedule = compute_mix_rate_schedule(args.mix_rate, num_encoder_layers)
    if rank == 0:
        logger.info(f"Mix rate schedule: {mix_rate_schedule}")

    # Setup loss functions
    vocab_size = raw_model.model.decoder.token_embedding.weight.shape[0]
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    ce_criterion = LabelSmoothingLoss(
        size=vocab_size,
        padding_idx=pad_token_id,
        smoothing=args.label_smoothing,
        normalize_length=False,
    )
    soft_criterion = LabelSoftLoss(
        size=vocab_size,
        padding_idx=pad_token_id,
        normalize_length=False,
    )

    # Load dataset
    dataset = load_dataset_splits(args)
    collator = LibriSpeechCollator(tokenizer=tokenizer)

    # Use DistributedSampler for multi-GPU
    if is_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=collator,
            pin_memory=True,
            drop_last=True,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collator,
            pin_memory=True if device.type == "cuda" else False,
        )

    # Setup optimizer and scheduler
    optimizer = AdamW(quantized_model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(dataloader) * args.epochs // args.accum_grad
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup_steps, total_steps
    )

    if rank == 0:
        logger.info(f"Total training steps: {total_steps}")
        logger.info(f"Warmup steps: {args.warmup_steps}")
        logger.info(
            f"QACT config: lambda_1={args.lambda_1}, lambda_2={args.lambda_2}, "
            f"mix_rate={args.mix_rate}"
        )

    # Training loop
    global_step = 0
    optimizer.zero_grad()
    for epoch in range(args.epochs):
        # Set epoch for DistributedSampler to shuffle data differently each epoch
        if is_distributed:
            sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_steps = 0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(dataloader):
            loss = qact_forward_pass(
                model=raw_model,
                batch=batch,
                device=device,
                num_encoder_layers=num_encoder_layers,
                mix_rate_schedule=mix_rate_schedule,
                lambda_1=args.lambda_1,
                lambda_2=args.lambda_2,
                ce_criterion=ce_criterion,
                soft_criterion=soft_criterion,
                vocab_size=vocab_size,
                pad_token_id=pad_token_id,
            )

            # Scale loss for gradient accumulation
            loss = loss / args.accum_grad
            loss.backward()

            if (batch_idx + 1) % args.accum_grad == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    quantized_model.parameters(), args.grad_clip
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item() * args.accum_grad
            epoch_steps += 1

            # Logging (only rank 0)
            if rank == 0 and (batch_idx + 1) % args.log_interval == 0:
                avg_loss = epoch_loss / epoch_steps
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"Epoch {epoch + 1}/{args.epochs} "
                    f"Step {batch_idx + 1}/{len(dataloader)} "
                    f"Loss: {avg_loss:.4f} LR: {lr:.2e}"
                )

        # End of epoch
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        if rank == 0:
            logger.info(
                f"Epoch {epoch + 1}/{args.epochs} completed in {epoch_time:.1f}s - "
                f"Avg Loss: {avg_epoch_loss:.4f}"
            )

        # Save checkpoint (only rank 0)
        if rank == 0 and (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(
                args.output_dir, f"checkpoint_epoch_{epoch + 1}.pt"
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "args": vars(args),
                    "loss": avg_epoch_loss,
                },
                checkpoint_path,
            )
            logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Synchronize before next epoch
        if is_distributed:
            dist.barrier()

    # Save final model (only rank 0)
    if rank == 0:
        final_path = os.path.join(args.output_dir, "final_model.pt")
        torch.save(
            {
                "model_state_dict": raw_model.state_dict(),
                "args": vars(args),
            },
            final_path,
        )
        logger.info(f"Training complete! Final model saved to: {final_path}")

    # Cleanup DDP
    cleanup_ddp()


def main():
    """Entry point for the training script."""
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
