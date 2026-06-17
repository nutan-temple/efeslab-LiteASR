"""
Modal app for running QACT Moonshine training on cloud GPUs.

This script defines a Modal application that:
1. Builds a container image with all required dependencies.
2. Mounts a persistent Volume at /checkpoints for saving/resuming training.
3. Launches the QACT co-training script on 8x A100 GPUs using torchrun for DDP.

Usage:
    # Run with default settings (8x A100, batch_size=64 per GPU):
    modal run src/qact_moonshine/modal_train.py

    # Run with custom arguments:
    modal run src/qact_moonshine/modal_train.py --epochs 50 --batch-size 32
"""

from pathlib import Path

import modal

# Resolve the src/ directory relative to this script's location so that
# `modal run` works regardless of the working directory.
SRC_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Modal App and Infrastructure
# ---------------------------------------------------------------------------

app = modal.App("qact-moonshine-training")

# Persistent volume for checkpoints -- data survives across runs
checkpoints_volume = modal.Volume.from_name(
    "qact-moonshine-checkpoints", create_if_missing=True
)

CHECKPOINTS_DIR = "/checkpoints"

# Container image with all training dependencies
training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Core ML dependencies
        "torch==2.6.0",
        "triton==3.2.0",
        "transformers==4.49.0",
        "safetensors==0.5.3",
        # Data and audio processing
        "datasets==3.3.2",
        "librosa==0.10.2.post1",
        "numba==0.61.0",
        "numpy==2.1.3",
        "sentencepiece==0.2.0",
        # Evaluation
        "evaluate==0.4.3",
        # Utilities
        "tqdm>=4.60.0",
        "pyyaml>=6.0",
    )
    .add_local_dir(str(SRC_DIR), "/root/src")
)

# ---------------------------------------------------------------------------
# Training Function
# ---------------------------------------------------------------------------


@app.function(
    image=training_image,
    gpu="A100:8",
    volumes={CHECKPOINTS_DIR: checkpoints_volume},
    timeout=86400,  # 24 hours max
)
def train(
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 5e-5,
    enc_weight_bit: int = 2,
    dec_weight_bit: int = 4,
    conv_weight_bit: int = 4,
    mix_rate: float = 1.8,
    model_name: str = "usefulsensors/moonshine-base",
    dataset: str = "librispeech_asr",
    dataset_config: str = "clean",
    train_split: str = "train.clean.100",
):
    """Run QACT co-training for Moonshine on 8x A100 GPUs via DDP.

    Uses torchrun to launch distributed training across all 8 GPUs.
    Batch size is per-GPU, so effective batch size = batch_size * 8.
    Checkpoints are saved to the persistent volume at /checkpoints so they
    persist across runs and can be downloaded later.

    Component-wise precision:
    - Encoder attention/MLP: enc_weight_bit (default 2-bit, co-trained with 1-bit)
    - Encoder conv frontend: conv_weight_bit (default 4-bit)
    - Decoder: dec_weight_bit (default 4-bit, fixed)
    """
    import subprocess
    import sys

    # Use torchrun to launch DDP training across all 8 GPUs
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node=8",
        "--master_addr=127.0.0.1",
        "--master_port=29500",
        "-m", "qact_moonshine.train_qact_moonshine",
        "--output-dir", CHECKPOINTS_DIR,
        "--model-name", model_name,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--enc-weight-bit", str(enc_weight_bit),
        "--dec-weight-bit", str(dec_weight_bit),
        "--conv-weight-bit", str(conv_weight_bit),
        "--mix-rate", str(mix_rate),
        "--dataset", dataset,
        "--dataset-config", dataset_config,
        "--train-split", train_split,
        "--quant-decoder",
    ]

    print(f"Starting QACT DDP training on 8x A100 with command:\n  {' '.join(cmd)}")
    print(f"Per-GPU batch size: {batch_size}, effective batch size: {batch_size * 8}")
    print(f"Checkpoints will be saved to: {CHECKPOINTS_DIR}")

    # Run the training script via torchrun for distributed data parallel
    result = subprocess.run(
        cmd,
        cwd="/root",
        env={
            **__import__("os").environ,
            "PYTHONPATH": "/root/src",
        },
    )

    # Commit the volume so checkpoints persist after the function exits
    checkpoints_volume.commit()

    if result.returncode != 0:
        raise RuntimeError(
            f"Training script exited with code {result.returncode}"
        )

    print("Training complete. Checkpoints saved to volume.")


# ---------------------------------------------------------------------------
# CLI Entrypoint (modal run)
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 5e-5,
    enc_weight_bit: int = 2,
    dec_weight_bit: int = 4,
    conv_weight_bit: int = 4,
    mix_rate: float = 1.8,
    model_name: str = "usefulsensors/moonshine-base",
    dataset: str = "librispeech_asr",
    dataset_config: str = "clean",
    train_split: str = "train.clean.100",
):
    """Local entrypoint invoked by `modal run src/qact_moonshine/modal_train.py`.

    Forwards all arguments to the remote training function.
    """
    train.remote(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        enc_weight_bit=enc_weight_bit,
        dec_weight_bit=dec_weight_bit,
        conv_weight_bit=conv_weight_bit,
        mix_rate=mix_rate,
        model_name=model_name,
        dataset=dataset,
        dataset_config=dataset_config,
        train_split=train_split,
    )
