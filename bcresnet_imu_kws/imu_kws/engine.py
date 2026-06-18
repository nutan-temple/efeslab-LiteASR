"""Training / evaluation engine.

Reuses the Qualcomm BC-ResNet *functions only* (imported from the vendored,
unmodified repo): the ``BCResNets`` model and the ``Preprocess`` feature
extractor (which internally uses their ``LogMel`` + ``spec_augment``).

Design choices for an accurate model on a small, imbalanced dataset (no brute
force):
  * keep ``n_mels = 40`` (REQUIRED by the architecture: SubSpectralNorm uses 5
    groups and the classifier uses a 5-wide frequency kernel, so the frequency
    axis must reduce 40 -> 20 -> 10 -> 5 -> 1);
  * mel front-end retuned for 3333 Hz audio (window/hop/n_fft);
  * stratified train/val/test split;
  * class-weighted cross-entropy (handles the begin=213 ... emergency=73 imbalance)
    OR an optional balanced sampler;
  * cosine LR schedule with warmup (same recipe shape as the original main.py);
  * model selection + early stopping on validation macro-F1 (robust to imbalance).
"""

import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import IMUKeywordDataset, build_index, compute_fixed_len, TARGET_SR
from .labels import CLASSES, NUM_CLASSES
from .splits import (
    assign_by_speaker,
    leave_one_speaker_out_plan,
    make_speaker_disjoint_splits,
)

# --- import the UNMODIFIED Qualcomm BC-ResNet code from the vendored copy ------
_HERE = os.path.dirname(os.path.abspath(__file__))
_VENDOR_BCRESNET = os.path.abspath(os.path.join(_HERE, "..", "vendor", "bcresnet"))
if _VENDOR_BCRESNET not in sys.path:
    sys.path.insert(0, _VENDOR_BCRESNET)

from bcresnet import BCResNets  # noqa: E402  (Qualcomm, unmodified)
from utils import Preprocess    # noqa: E402  (Qualcomm, unmodified)

# n_mels is fixed by the BC-ResNet architecture - do not change.
N_MELS = 40
# SpecAugment frequency-mask width per model size, mirroring the original main.py.
_FREQ_MASK_PARA = {1: 0, 1.5: 1, 2: 3, 3: 5, 6: 7, 8: 7}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_preprocessors(device, tau, sample_rate=TARGET_SR):
    """Two vendored ``Preprocess`` instances: train (SpecAugment) and eval (clean).

    We pass ``noise_loc=None`` and always call them with ``augment=False`` so the
    GSC-specific (16 kHz, background-noise) waveform path is never touched; only
    the sample-rate-correct LogMel (and SpecAugment for training) runs.
    """
    mel_kwargs = dict(
        sample_rate=sample_rate,
        n_fft=512,
        win_length=256,   # ~77 ms @ 3333 Hz
        hop_length=64,    # ~19 ms @ 3333 Hz
        n_mels=N_MELS,
    )
    specaug = tau >= 1.5
    pre_train = Preprocess(
        None,
        device,
        specaug=specaug,
        frequency_masking_para=_FREQ_MASK_PARA.get(tau, 5),
        time_masking_para=15,
        frequency_mask_num=2,
        time_mask_num=2,
        **mel_kwargs,
    )
    pre_eval = Preprocess(None, device, specaug=False, **mel_kwargs)
    return pre_train, pre_eval


def _make_loaders(splits, target_len, batch_size, balanced_sampler, num_workers, target_sr=TARGET_SR):
    p_tr, y_tr = splits["train"]
    p_va, y_va = splits["valid"]
    p_te, y_te = splits["test"]

    ds_tr = IMUKeywordDataset(p_tr, y_tr, target_len, train=True, augment=True, target_sr=target_sr)
    ds_va = IMUKeywordDataset(p_va, y_va, target_len, train=False, augment=False, target_sr=target_sr)
    ds_te = IMUKeywordDataset(p_te, y_te, target_len, train=False, augment=False, target_sr=target_sr)

    if balanced_sampler:
        counts = np.bincount(y_tr, minlength=NUM_CLASSES).astype(np.float64)
        w_per_class = 1.0 / np.maximum(counts, 1.0)
        sample_weights = [w_per_class[y] for y in y_tr]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        tr_loader = DataLoader(
            ds_tr, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, drop_last=True,
        )
    else:
        tr_loader = DataLoader(
            ds_tr, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=True,
        )
    va_loader = DataLoader(ds_va, batch_size=batch_size, num_workers=num_workers)
    te_loader = DataLoader(ds_te, batch_size=batch_size, num_workers=num_workers)
    return (ds_tr, ds_va, ds_te), (tr_loader, va_loader, te_loader)


def _standardize(feats, eps=1e-5):
    """Per-utterance standardization of the log-mel features (zero mean, unit std
    over freq+time, per sample). Removes per-recording offset/scale, which helps
    consistency across speakers/sessions."""
    m = feats.mean(dim=(1, 2, 3), keepdim=True)
    s = feats.std(dim=(1, 2, 3), keepdim=True)
    return (feats - m) / (s + eps)


@torch.no_grad()
def evaluate(model, loader, preprocess, device, standardize=False):
    model.eval()
    y_true, y_pred = [], []
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        feats = preprocess(inputs, labels, augment=False, is_train=False)
        if standardize:
            feats = _standardize(feats)
        outputs = model(feats)
        preds = outputs.argmax(dim=-1)
        y_true.append(labels.cpu().numpy())
        y_pred.append(preds.cpu().numpy())
    y_true = np.concatenate(y_true) if y_true else np.array([])
    y_pred = np.concatenate(y_pred) if y_pred else np.array([])
    acc = float((y_true == y_pred).mean() * 100.0) if len(y_true) else 0.0
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if len(y_true) else 0.0
    return acc, macro_f1, y_true, y_pred


def _train_core(cfg, device, splits, target_len, sample_rate, on_best=None, verbose=True):
    """Train one model on the given ``splits`` and evaluate on its test set.

    Returns a results dict including the pooled ``y_true``/``y_pred`` for the test
    split so callers (e.g. LOSO) can aggregate across folds.
    """
    (_, _, _), (tr_loader, va_loader, te_loader) = _make_loaders(
        splits, target_len, cfg["batch_size"], cfg["balanced_sampler"],
        cfg.get("num_workers", 4), target_sr=sample_rate,
    )
    has_val = len(splits["valid"][1]) > 0

    model = BCResNets(int(cfg["tau"] * 8), num_classes=NUM_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print("model: BC-ResNet-%.1f | params: %d (~%.1f KB fp32)" % (
            cfg["tau"], n_params, n_params * 4 / 1024.0))

    pre_train, pre_eval = build_preprocessors(device, cfg["tau"], sample_rate=sample_rate)
    do_std = bool(cfg.get("standardize", True))

    if cfg["balanced_sampler"]:
        ce_weight = None
    else:
        y_tr = splits["train"][1]
        tr_counts = np.bincount(y_tr, minlength=NUM_CLASSES).astype(np.float64)
        w = tr_counts.sum() / (NUM_CLASSES * np.maximum(tr_counts, 1.0))
        ce_weight = torch.tensor(w, dtype=torch.float32, device=device)

    opt_name = str(cfg.get("optimizer", "sgd")).lower()
    if opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0, weight_decay=cfg["weight_decay"])
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.0, weight_decay=cfg["weight_decay"], momentum=0.9
        )
    steps_per_epoch = max(1, len(tr_loader))
    total_iter = steps_per_epoch * cfg["epochs"]
    warmup_iter = steps_per_epoch * cfg["warmup_epochs"]
    init_lr = cfg["lr"]

    best_f1 = -1.0
    best_state = None
    bad_epochs = 0
    iteration = 0
    lr = init_lr

    for epoch in range(cfg["epochs"]):
        model.train()
        run_loss, tr_correct, tr_total = 0.0, 0, 0
        for inputs, lab in tr_loader:
            iteration += 1
            if iteration < warmup_iter:
                lr = init_lr * iteration / max(1, warmup_iter)
            else:
                progress = (iteration - warmup_iter) / max(1, total_iter - warmup_iter)
                lr = 0.5 * init_lr * (1 + np.cos(np.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = lr

            inputs = inputs.to(device)
            lab = lab.to(device)
            feats = pre_train(inputs, lab, augment=False, is_train=True)
            if do_std:
                feats = _standardize(feats)
            outputs = model(feats)
            loss = F.cross_entropy(outputs, lab, weight=ce_weight)
            loss.backward()
            optimizer.step()
            model.zero_grad()

            bs = lab.size(0)
            run_loss += float(loss.item()) * bs
            tr_correct += int((outputs.argmax(-1) == lab).sum().item())
            tr_total += bs

        train_loss = run_loss / max(1, tr_total)
        train_acc = 100.0 * tr_correct / max(1, tr_total)

        if has_val:
            va_acc, va_f1, _, _ = evaluate(model, va_loader, pre_eval, device, standardize=do_std)
            if verbose:
                print("epoch %3d/%d | lr %.4f | train_loss %.3f train_acc %.2f | val_acc %.2f val_macroF1 %.4f%s" % (
                    epoch + 1, cfg["epochs"], lr, train_loss, train_acc, va_acc, va_f1,
                    "  *best*" if va_f1 > best_f1 else ""))
            if va_f1 > best_f1:
                best_f1 = va_f1
                bad_epochs = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if on_best is not None:
                    on_best(best_state, epoch, best_f1)
            else:
                bad_epochs += 1
                if cfg["patience"] and bad_epochs >= cfg["patience"]:
                    if verbose:
                        print("early stopping at epoch %d (no val improvement for %d)" % (
                            epoch + 1, cfg["patience"]))
                    break
        elif verbose and (epoch + 1) % 10 == 0:
            print("epoch %3d/%d | lr %.4f | train_loss %.3f train_acc %.2f (no val speaker)" % (
                epoch + 1, cfg["epochs"], lr, train_loss, train_acc))

    if not has_val:
        # No validation split: keep the final-epoch weights.
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if on_best is not None:
            on_best(best_state, cfg["epochs"] - 1, float("nan"))

    if best_state is not None:
        model.load_state_dict(best_state)

    te_acc, te_f1, y_true, y_pred = evaluate(model, te_loader, pre_eval, device, standardize=do_std)
    report = classification_report(y_true, y_pred, labels=list(range(NUM_CLASSES)),
                                   target_names=CLASSES, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES))).tolist()
    return {
        "model": model,
        "best_state": best_state,
        "best_val_macro_f1": best_f1,
        "test_acc": te_acc,
        "test_macro_f1": te_f1,
        "report": report,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "n_params": int(n_params),
        "classes": CLASSES,
    }


def _prepare(cfg):
    """Shared setup: index the data, log stats, return (paths, labels, skipped,
    target_len, sample_rate)."""
    set_seed(cfg["seed"])
    sample_rate = int(cfg.get("target_sr", TARGET_SR))
    strict_sr = bool(cfg.get("strict_sr", False))

    paths, labels, skipped = build_index(
        cfg["wav_dir"], cfg["use_filtered"], cfg.get("manifest"),
        target_sr=sample_rate, strict_sr=strict_sr,
    )
    if len(paths) == 0:
        raise RuntimeError("No usable wav files found under %s" % cfg["wav_dir"])

    counts = np.bincount(labels, minlength=NUM_CLASSES)
    print("usable files: %d | per-class: %s" % (
        len(paths), {CLASSES[i]: int(counts[i]) for i in range(NUM_CLASSES)}))
    print("training rate locked to %d Hz (strict_sr=%s)" % (sample_rate, strict_sr))
    if skipped:
        print("skipped %d files (e.g. %s)" % (len(skipped), skipped[:3]))

    target_len = cfg.get("target_len") or compute_fixed_len(
        paths, cfg["length_percentile"], target_sr=sample_rate)
    print("fixed input length: %d samples (~%.2fs @ %d Hz)" % (
        target_len, target_len / float(sample_rate), sample_rate))
    return paths, labels, skipped, target_len, sample_rate


def train(cfg, device, on_best=None):
    """Single speaker-disjoint 80/10/10 split, train once, test once."""
    paths, labels, skipped, target_len, sample_rate = _prepare(cfg)

    splits = make_speaker_disjoint_splits(
        paths, labels, cfg["wav_dir"], seed=cfg["seed"],
        test_frac=cfg.get("test_frac", 0.10), val_frac=cfg.get("val_frac", 0.10),
    )
    for name in ("train", "valid", "test"):
        print("  %-5s: %d" % (name, len(splits[name][1])))

    res = _train_core(cfg, device, splits, target_len, sample_rate, on_best=on_best, verbose=True)
    res.update({"target_len": int(target_len), "skipped": skipped})
    print("\n=== TEST ===\nacc %.2f | macroF1 %.4f\n%s\nconfusion_matrix=%s" % (
        res["test_acc"], res["test_macro_f1"], res["report"], res["confusion_matrix"]))
    return res


def run_loso(cfg, device, on_fold=None):
    """Leave-One-Speaker-Out cross-validation.

    Trains one model per held-out test speaker and aggregates results. ``on_fold``,
    if given, is called as ``on_fold(test_speaker, state_dict, epoch, val_f1)`` on
    each fold's best checkpoint (used to save + commit per-fold models on Modal).
    """
    paths, labels, skipped, target_len, sample_rate = _prepare(cfg)

    folds, speakers = leave_one_speaker_out_plan(paths, labels, cfg["wav_dir"])
    print("LOSO: %d folds over %d speakers: %s" % (len(folds), len(speakers), speakers))

    pooled_true, pooled_pred, per_fold = [], [], []
    last_n_params = None
    for fi, fold in enumerate(folds):
        test_spk, val_spk = fold["test"], fold["val"]
        print("\n===== FOLD %d/%d | test=%s | val=%s =====" % (
            fi + 1, len(folds), test_spk, val_spk))
        splits = assign_by_speaker(
            paths, labels, cfg["wav_dir"], [test_spk], [val_spk] if val_spk else [])
        for name in ("train", "valid", "test"):
            print("  %-5s: %d" % (name, len(splits[name][1])))

        fold_on_best = None
        if on_fold is not None:
            def fold_on_best(state, epoch, f1, _ts=test_spk):
                on_fold(_ts, state, epoch, f1)

        res = _train_core(cfg, device, splits, target_len, sample_rate,
                          on_best=fold_on_best, verbose=True)
        last_n_params = res["n_params"]
        pooled_true.append(res["y_true"])
        pooled_pred.append(res["y_pred"])
        per_fold.append({
            "test_speaker": test_spk, "val_speaker": val_spk,
            "n_test": int(len(res["y_true"])),
            "test_acc": res["test_acc"], "test_macro_f1": res["test_macro_f1"],
        })
        print("fold %s -> test_acc %.2f | macroF1 %.4f" % (
            test_spk, res["test_acc"], res["test_macro_f1"]))

    y_true = np.concatenate(pooled_true)
    y_pred = np.concatenate(pooled_pred)
    pooled_acc = float((y_true == y_pred).mean() * 100.0)
    pooled_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    report = classification_report(y_true, y_pred, labels=list(range(NUM_CLASSES)),
                                   target_names=CLASSES, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES))).tolist()
    accs = [f["test_acc"] for f in per_fold]
    f1s = [f["test_macro_f1"] for f in per_fold]

    print("\n===== LOSO SUMMARY =====")
    print("per-fold macroF1 %.4f +/- %.4f | per-fold acc %.2f +/- %.2f" % (
        float(np.mean(f1s)), float(np.std(f1s)), float(np.mean(accs)), float(np.std(accs))))
    print("pooled (all held-out speakers) acc %.2f | macroF1 %.4f" % (pooled_acc, pooled_f1))
    print(report)
    print("pooled confusion_matrix=%s" % cm)

    return {
        "mode": "loso",
        "folds": per_fold,
        "pooled_acc": pooled_acc,
        "pooled_macro_f1": pooled_f1,
        "fold_macro_f1_mean": float(np.mean(f1s)),
        "fold_macro_f1_std": float(np.std(f1s)),
        "fold_acc_mean": float(np.mean(accs)),
        "fold_acc_std": float(np.std(accs)),
        "report": report,
        "confusion_matrix": cm,
        "target_len": int(target_len),
        "n_params": last_n_params,
        "classes": CLASSES,
        "skipped": skipped,
        "speakers": speakers,
    }
