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
    target_sr: int = 3333,
    strict_sr: bool = False,
    test_frac: float = 0.10,
    val_frac: float = 0.10,
    loso: bool = True,
    optimizer: str = "sgd",
    standardize: bool = True,
    feature: str = "logmel_40",
    preproc: str = "hp_peak_crop",
    window_seconds: float = 2.5,
    fmin: float = 50.0,
    fmax: float = 500.0,
    specaug: bool = True,
):
    import sys

    import torch

    sys.path.insert(0, "/root/app")
    from imu_kws.engine import run_loso, train
    from imu_kws.labels import CLASSES

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    os.makedirs(OUT_DIR, exist_ok=True)

    mel_cfg = dict(sample_rate=target_sr, n_fft=512, hop_length=64, n_mels=40,
                   feature=feature, preproc=preproc, window_seconds=window_seconds,
                   fmin=fmin, fmax=fmax)
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
        target_sr=target_sr,
        strict_sr=strict_sr,
        test_frac=test_frac,
        val_frac=val_frac,
        optimizer=optimizer,
        standardize=standardize,
        feature=feature,
        preproc=preproc,
        window_seconds=window_seconds,
        fmin=fmin,
        fmax=fmax,
        specaug=specaug,
        num_workers=4,
    )

    if loso:
        # Leave-One-Speaker-Out: one model per held-out speaker, saved + committed.
        def on_fold(test_speaker, state_dict, epoch, val_f1):
            fold_path = os.path.join(OUT_DIR, "bcresnet_imu_fold_%s.pt" % test_speaker)
            torch.save(
                {
                    "state_dict": state_dict,
                    "classes": CLASSES,
                    "tau": tau,
                    "held_out_speaker": test_speaker,
                    "epoch": epoch + 1,
                    "val_macro_f1": val_f1,
                    "mel": mel_cfg,
                },
                fold_path,
            )
            model_vol.commit()
            print("[fold %s] saved + committed (epoch %d, val_macroF1 %.4f)" % (
                test_speaker, epoch + 1, val_f1))

        result = run_loso(cfg, device, on_fold=on_fold)
        summary = {
            "mode": "loso",
            "pooled_acc": result["pooled_acc"],
            "pooled_macro_f1": result["pooled_macro_f1"],
            "fold_macro_f1_mean": result["fold_macro_f1_mean"],
            "fold_macro_f1_std": result["fold_macro_f1_std"],
            "fold_acc_mean": result["fold_acc_mean"],
            "fold_acc_std": result["fold_acc_std"],
            "folds": result["folds"],
            "report": result["report"],
            "confusion_matrix": result["confusion_matrix"],
            "target_len": result["target_len"],
            "n_params": result["n_params"],
            "classes": result["classes"],
            "speakers": result["speakers"],
            "config": {k: v for k, v in cfg.items() if k != "num_workers"},
            "num_skipped": len(result["skipped"]),
        }
        with open(os.path.join(OUT_DIR, "loso_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        model_vol.commit()
        print("saved LOSO summary + %d per-fold checkpoints to %s" % (len(result["folds"]), OUT_DIR))
        return summary

    # --- single speaker-disjoint 80/10/10 split (loso=False) ---
    ckpt_path = os.path.join(OUT_DIR, "bcresnet_imu_best.pt")

    def on_best(state_dict, epoch, val_f1):
        torch.save(
            {
                "state_dict": state_dict,
                "classes": CLASSES,
                "tau": tau,
                "epoch": epoch + 1,
                "val_macro_f1": val_f1,
                "mel": mel_cfg,
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
            "mel": mel_cfg,
        },
        ckpt_path,
    )
    summary = {
        "mode": "single_split",
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
    lr: float = 0.05,
    optimizer: str = "sgd",
    length_percentile: float = 99.0,
    use_filtered: bool = True,
    manifest: str = "",
    balanced_sampler: bool = False,
    strict_sr: bool = False,
    loso: bool = True,
    standardize: bool = True,
    feature: str = "logmel_40",
    preproc: str = "hp_peak_crop",
    window_seconds: float = 2.5,
):
    summary = train_remote.remote(
        tau=tau,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        optimizer=optimizer,
        length_percentile=length_percentile,
        use_filtered=use_filtered,
        manifest=(manifest or None),
        balanced_sampler=balanced_sampler,
        strict_sr=strict_sr,
        loso=loso,
        standardize=standardize,
        feature=feature,
        preproc=preproc,
        window_seconds=window_seconds,
    )
    if summary.get("mode") == "loso":
        keys = ("pooled_acc", "pooled_macro_f1", "fold_macro_f1_mean",
                "fold_macro_f1_std", "fold_acc_mean", "fold_acc_std", "target_len")
    else:
        keys = ("best_val_macro_f1", "test_acc", "test_macro_f1", "target_len")
    print(json.dumps({k: summary[k] for k in keys}, indent=2))
