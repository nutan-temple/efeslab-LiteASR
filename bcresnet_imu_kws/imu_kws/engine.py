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
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import IMUKeywordDataset, build_index, compute_fixed_len
from .labels import CLASSES, NUM_CLASSES

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


def build_preprocessors(device, tau, sample_rate=3333):
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


def make_splits(paths, labels, seed):
    """Stratified 70/15/15 split. Returns dict of (paths, labels) per split."""
    p_tr, p_tmp, y_tr, y_tmp = train_test_split(
        paths, labels, test_size=0.30, random_state=seed, stratify=labels
    )
    p_va, p_te, y_va, y_te = train_test_split(
        p_tmp, y_tmp, test_size=0.50, random_state=seed, stratify=y_tmp
    )
    return {
        "train": (p_tr, y_tr),
        "valid": (p_va, y_va),
        "test": (p_te, y_te),
    }


def _make_loaders(splits, target_len, batch_size, balanced_sampler, num_workers):
    p_tr, y_tr = splits["train"]
    p_va, y_va = splits["valid"]
    p_te, y_te = splits["test"]

    ds_tr = IMUKeywordDataset(p_tr, y_tr, target_len, train=True, augment=True)
    ds_va = IMUKeywordDataset(p_va, y_va, target_len, train=False, augment=False)
    ds_te = IMUKeywordDataset(p_te, y_te, target_len, train=False, augment=False)

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


@torch.no_grad()
def evaluate(model, loader, preprocess, device):
    model.eval()
    y_true, y_pred = [], []
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        feats = preprocess(inputs, labels, augment=False, is_train=False)
        outputs = model(feats)
        preds = outputs.argmax(dim=-1)
        y_true.append(labels.cpu().numpy())
        y_pred.append(preds.cpu().numpy())
    y_true = np.concatenate(y_true) if y_true else np.array([])
    y_pred = np.concatenate(y_pred) if y_pred else np.array([])
    acc = float((y_true == y_pred).mean() * 100.0) if len(y_true) else 0.0
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if len(y_true) else 0.0
    return acc, macro_f1, y_true, y_pred


def train(cfg, device, on_best=None):
    """Run the full pipeline and return a results dict.

    ``cfg`` keys: wav_dir, use_filtered, manifest, tau, epochs, batch_size, lr,
    weight_decay, warmup_epochs, patience, seed, length_percentile, target_len,
    balanced_sampler, num_workers.
    ``on_best(state_dict, epoch, val_macro_f1)`` is called whenever validation
    macro-F1 improves (used to checkpoint + commit the Modal volume).
    """
    set_seed(cfg["seed"])

    paths, labels, skipped = build_index(cfg["wav_dir"], cfg["use_filtered"], cfg.get("manifest"))
    if len(paths) == 0:
        raise RuntimeError("No usable wav files found under %s" % cfg["wav_dir"])

    counts = np.bincount(labels, minlength=NUM_CLASSES)
    print("usable files: %d | per-class: %s" % (
        len(paths), {CLASSES[i]: int(counts[i]) for i in range(NUM_CLASSES)}))
    if skipped:
        print("skipped %d files (e.g. %s)" % (len(skipped), skipped[:3]))

    target_len = cfg.get("target_len") or compute_fixed_len(paths, cfg["length_percentile"])
    print("fixed input length: %d samples (~%.2fs @ 3333 Hz)" % (target_len, target_len / 3333.0))

    splits = make_splits(paths, labels, cfg["seed"])
    for name in ("train", "valid", "test"):
        _, ys = splits[name]
        print("  %-5s: %d" % (name, len(ys)))

    (_, _, _), (tr_loader, va_loader, te_loader) = _make_loaders(
        splits, target_len, cfg["batch_size"], cfg["balanced_sampler"], cfg.get("num_workers", 4)
    )

    model = BCResNets(int(cfg["tau"] * 8), num_classes=NUM_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("model: BC-ResNet-%.1f | params: %d (~%.1f KB fp32)" % (
        cfg["tau"], n_params, n_params * 4 / 1024.0))

    pre_train, pre_eval = build_preprocessors(device, cfg["tau"])

    # class-weighted CE (skip weights if the balanced sampler already rebalances)
    if cfg["balanced_sampler"]:
        ce_weight = None
    else:
        y_tr = splits["train"][1]
        tr_counts = np.bincount(y_tr, minlength=NUM_CLASSES).astype(np.float64)
        w = tr_counts.sum() / (NUM_CLASSES * np.maximum(tr_counts, 1.0))
        ce_weight = torch.tensor(w, dtype=torch.float32, device=device)
        print("class weights: %s" % {CLASSES[i]: round(float(w[i]), 3) for i in range(NUM_CLASSES)})

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

    for epoch in range(cfg["epochs"]):
        model.train()
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
            outputs = model(feats)
            loss = F.cross_entropy(outputs, lab, weight=ce_weight)
            loss.backward()
            optimizer.step()
            model.zero_grad()

        va_acc, va_f1, _, _ = evaluate(model, va_loader, pre_eval, device)
        print("epoch %3d/%d | lr %.4f | val_acc %.2f | val_macroF1 %.4f%s" % (
            epoch + 1, cfg["epochs"], lr, va_acc, va_f1,
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
                print("early stopping at epoch %d (no val improvement for %d epochs)" % (
                    epoch + 1, cfg["patience"]))
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    te_acc, te_f1, y_true, y_pred = evaluate(model, te_loader, pre_eval, device)
    report = classification_report(y_true, y_pred, labels=list(range(NUM_CLASSES)),
                                   target_names=CLASSES, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES))).tolist()
    print("\n=== TEST ===\nacc %.2f | macroF1 %.4f\n%s\nconfusion_matrix=%s" % (
        te_acc, te_f1, report, cm))

    return {
        "model": model,
        "best_state": best_state,
        "best_val_macro_f1": best_f1,
        "test_acc": te_acc,
        "test_macro_f1": te_f1,
        "report": report,
        "confusion_matrix": cm,
        "target_len": int(target_len),
        "n_params": int(n_params),
        "skipped": skipped,
        "classes": CLASSES,
    }
