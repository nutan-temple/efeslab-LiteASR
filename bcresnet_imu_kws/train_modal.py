"""Modal training app for BC-ResNet IMU keyword spotting.

Run:
    modal run bcresnet_imu_kws/train_modal.py                 # defaults (BC-ResNet-3)
    modal run bcresnet_imu_kws/train_modal.py --tau 1 --epochs 150
    modal run bcresnet_imu_kws/train_modal.py --use-filtered False   # train on raw wavs

Reads audio from the existing ``kws-imu-data`` volume and writes checkpoints +
metrics to the ``kws-imu-models`` volume (created if missing), committing on every
best checkpoint and at the end.
"""

import json
import os

import modal

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

app = modal.App("bcresnet-imu-kws")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.1",
        "torchaudio==2.3.1",
        "numpy<2",
        "scipy",
        "scikit-learn",
        "soundfile",
        "tqdm",
        "requests",
    )
    # Ship our package + the vendored (unmodified) BC-ResNet code into the image.
    .add_local_dir(PROJECT_DIR, remote_path="/root/app", copy=True)
)

# Existing volume that already holds the recordings.
raw_vol = modal.Volume.from_name("kws-imu-data", create_if_missing=False)
# Output volume for trained models + reports.
model_vol = modal.Volume.from_name("kws-imu-models", create_if_missing=True)

WAV_DIR = "/data/imu_data_wav_3333hz"
OUT_DIR = "/models/bcresnet_imu"


@app.function(
    image=image,
    gpu="T4",
    volumes={"/data": raw_vol, "/models": model_vol},
    timeout=60 * 60 * 3,
)
def train_remote(
    tau: float = 3.0,
    epochs: int = 120,
    batch_size: int = 32,
    lr: float = 0.05,
    weight_decay: float = 1e-3,
    warmup_epochs: int = 5,
    patience: int = 25,
    seed: int = 42,
    use_filtered: bool = True,
    manifest: str = None,
    length_percentile: float = 99.0,
    balanced_sampler: bool = False,
    target_len: int = 0,
):
    import sys

    import torch

    sys.path.insert(0, "/root/app")
    from imu_kws.engine import train
    from imu_kws.labels import CLASSES

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    os.makedirs(OUT_DIR, exist_ok=True)

    ckpt_path = os.path.join(OUT_DIR, "bcresnet_imu_best.pt")
    cfg = dict(
        wav_dir=WAV_DIR,
        use_filtered=use_filtered,
        manifest=manifest,
        tau=tau,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        patience=patience,
        seed=seed,
        length_percentile=length_percentile,
        balanced_sampler=balanced_sampler,
        target_len=(target_len or None),
        num_workers=4,
    )

    def on_best(state_dict, epoch, val_f1):
        torch.save(
            {
                "state_dict": state_dict,
                "classes": CLASSES,
                "tau": tau,
                "epoch": epoch + 1,
                "val_macro_f1": val_f1,
                "mel": dict(sample_rate=3333, n_fft=512, win_length=256, hop_length=64, n_mels=40),
            },
            ckpt_path,
        )
        model_vol.commit()
        print("[checkpoint] epoch %d val_macroF1 %.4f -> committed to volume" % (epoch + 1, val_f1))

    result = train(cfg, device, on_best=on_best)

    # Persist the final best checkpoint + a human-readable summary, then commit.
    torch.save(
        {
            "state_dict": result["best_state"],
            "classes": result["classes"],
            "tau": tau,
            "val_macro_f1": result["best_val_macro_f1"],
            "target_len": result["target_len"],
            "mel": dict(sample_rate=3333, n_fft=512, win_length=256, hop_length=64, n_mels=40),
        },
        ckpt_path,
    )
    summary = {
        "best_val_macro_f1": result["best_val_macro_f1"],
        "test_acc": result["test_acc"],
        "test_macro_f1": result["test_macro_f1"],
        "report": result["report"],
        "confusion_matrix": result["confusion_matrix"],
        "target_len": result["target_len"],
        "n_params": result["n_params"],
        "classes": result["classes"],
        "config": {k: v for k, v in cfg.items() if k != "num_workers"},
        "num_skipped": len(result["skipped"]),
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    model_vol.commit()

    print("saved best checkpoint to", ckpt_path)
    return summary


@app.local_entrypoint()
def main(
    tau: float = 3.0,
    epochs: int = 120,
    batch_size: int = 32,
    use_filtered: bool = True,
    manifest: str = "",
    balanced_sampler: bool = False,
):
    summary = train_remote.remote(
        tau=tau,
        epochs=epochs,
        batch_size=batch_size,
        use_filtered=use_filtered,
        manifest=(manifest or None),
        balanced_sampler=balanced_sampler,
    )
    print(json.dumps(
        {k: summary[k] for k in ("best_val_macro_f1", "test_acc", "test_macro_f1", "target_len")},
        indent=2,
    ))
