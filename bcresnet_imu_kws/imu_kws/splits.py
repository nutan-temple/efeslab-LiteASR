"""Speaker-disjoint train/val/test splitting.

Requirements implemented here:
  * 80 / 10 / 10 (train / val / test) by number of files;
  * **no speaker overlap** - every recording from a given speaker lands entirely
    in a single split (the speaker is the top-level folder under ``wav_dir``,
    e.g. ``arnav`` in ``arnav/begin_activity_..._accel.wav``);
  * the **test set is class-balanced to match the train distribution** (its
    per-class proportions track the train split's per-class proportions).

Because speakers are kept whole, we cannot pick individual files to balance the
test set. Instead we run a fast randomized search over speaker -> split
assignments and keep the assignment that best matches the target split sizes and
makes the test class-distribution closest to the train class-distribution (while
guaranteeing every class is present in train).
"""

import os
import random
from collections import defaultdict

import numpy as np

from .labels import CLASSES, NUM_CLASSES


def speaker_from_path(path, wav_dir):
    """Top-level folder under ``wav_dir`` (the speaker id). Falls back to a single
    bucket when files sit directly in ``wav_dir`` (no per-speaker subfolders)."""
    try:
        rel = os.path.relpath(path, wav_dir)
    except Exception:
        rel = str(path)
    parts = rel.replace("\\", "/").split("/")
    return parts[0] if len(parts) > 1 else "_root_"


def _class_dist(label_array):
    counts = np.bincount(label_array, minlength=NUM_CLASSES).astype(np.float64)
    total = counts.sum()
    return counts / total if total > 0 else counts


def _pack(paths, labels, tr_idx, va_idx, te_idx):
    def take(idxs):
        return ([paths[i] for i in idxs], [labels[i] for i in idxs])
    return {"train": take(tr_idx), "valid": take(va_idx), "test": take(te_idx)}


def speaker_groups(paths, wav_dir):
    """Map speaker id -> list of sample indices."""
    groups = defaultdict(list)
    for i, p in enumerate(paths):
        groups[speaker_from_path(p, wav_dir)].append(i)
    return groups


def assign_by_speaker(paths, labels, wav_dir, test_speakers, val_speakers=None):
    """Build a splits dict by assigning whole speakers to test / val / (rest=train)."""
    test_set = set(test_speakers)
    val_set = set(val_speakers or [])
    groups = speaker_groups(paths, wav_dir)
    tr_idx, va_idx, te_idx = [], [], []
    for spk, idxs in groups.items():
        if spk in test_set:
            te_idx += idxs
        elif spk in val_set:
            va_idx += idxs
        else:
            tr_idx += idxs
    return _pack(paths, labels, tr_idx, va_idx, te_idx)


def leave_one_speaker_out_plan(paths, labels, wav_dir):
    """Return ``(folds, speakers)`` for Leave-One-Speaker-Out cross-validation.

    Each fold holds one speaker out as the test set. A second, *representative*
    speaker (class-distribution closest to the global distribution) is held out as
    validation for early stopping, so test/val/train are all speaker-disjoint. With
    fewer than 3 speakers there is no room for a separate val speaker, so ``val`` is
    ``None`` (training then runs for the full epoch budget with no early stopping).
    """
    groups = speaker_groups(paths, wav_dir)
    speakers = sorted(groups)
    global_dist = _class_dist(np.array(labels))

    def dist_l1(spk):
        d = _class_dist(np.array([labels[i] for i in groups[spk]]))
        return float(np.abs(d - global_dist).sum())

    folds = []
    for test_spk in speakers:
        remaining = [s for s in speakers if s != test_spk]
        val_spk = None
        if len(speakers) >= 3 and remaining:
            val_spk = min(remaining, key=lambda s: (dist_l1(s), s))
        folds.append({"test": test_spk, "val": val_spk})
    return folds, speakers


def _stratified_fallback(paths, labels, seed, test_frac, val_frac, reason):
    from sklearn.model_selection import train_test_split

    print("WARNING: %s -> falling back to a stratified (NON speaker-disjoint) split." % reason)
    idx = list(range(len(paths)))
    tr, tmp = train_test_split(
        idx, test_size=(test_frac + val_frac), random_state=seed, stratify=labels
    )
    rel = test_frac / (test_frac + val_frac)
    va, te = train_test_split(
        tmp, test_size=rel, random_state=seed, stratify=[labels[i] for i in tmp]
    )
    return _pack(paths, labels, tr, va, te)


def make_speaker_disjoint_splits(
    paths, labels, wav_dir, seed=42, test_frac=0.10, val_frac=0.10, n_trials=5000, verbose=True
):
    labels = list(labels)
    n_total = len(paths)
    speakers = [speaker_from_path(p, wav_dir) for p in paths]
    unique_speakers = sorted(set(speakers))

    if len(unique_speakers) < 3:
        return _stratified_fallback(
            paths, labels, seed, test_frac, val_frac,
            "only %d speaker(s) found, need >= 3 for disjoint train/val/test" % len(unique_speakers),
        )

    spk_idx = defaultdict(list)
    for i, s in enumerate(speakers):
        spk_idx[s].append(i)
    spk_count = {s: len(v) for s, v in spk_idx.items()}

    target_test = test_frac * n_total
    target_val = val_frac * n_total

    rng = random.Random(seed)
    best = None
    for _ in range(n_trials):
        order = list(unique_speakers)
        rng.shuffle(order)

        test_spk, val_spk, train_spk = [], [], []
        test_n = val_n = 0
        for s in order:
            if test_n < target_test:
                test_spk.append(s)
                test_n += spk_count[s]
            elif val_n < target_val:
                val_spk.append(s)
                val_n += spk_count[s]
            else:
                train_spk.append(s)

        if not test_spk or not val_spk or not train_spk:
            continue

        tr_idx = [i for s in train_spk for i in spk_idx[s]]
        va_idx = [i for s in val_spk for i in spk_idx[s]]
        te_idx = [i for s in test_spk for i in spk_idx[s]]

        tr_lab = np.array([labels[i] for i in tr_idx])
        te_lab = np.array([labels[i] for i in te_idx])
        va_lab = np.array([labels[i] for i in va_idx])

        # Hard requirement: train must contain every class.
        if len(set(tr_lab.tolist())) < NUM_CLASSES:
            continue

        train_dist = _class_dist(tr_lab)
        test_dist = _class_dist(te_lab)

        dist_pen = float(np.abs(test_dist - train_dist).sum())          # balance test to train
        miss_test = (NUM_CLASSES - len(set(te_lab.tolist()))) * 0.5     # prefer all classes in test
        miss_val = (NUM_CLASSES - len(set(va_lab.tolist()))) * 0.25
        size_pen = 0.5 * (abs(len(te_idx) - target_test) + abs(len(va_idx) - target_val)) / n_total
        score = dist_pen + miss_test + miss_val + size_pen

        if best is None or score < best["score"]:
            best = {
                "score": score, "tr_idx": tr_idx, "va_idx": va_idx, "te_idx": te_idx,
                "train_spk": sorted(train_spk), "val_spk": sorted(val_spk), "test_spk": sorted(test_spk),
                "train_dist": train_dist, "test_dist": test_dist, "dist_pen": dist_pen,
            }

    if best is None:
        return _stratified_fallback(
            paths, labels, seed, test_frac, val_frac,
            "no valid speaker-disjoint assignment kept all classes in train",
        )

    if verbose:
        _report(best, n_total, labels)

    return _pack(paths, labels, best["tr_idx"], best["va_idx"], best["te_idx"])


def _report(best, n_total, labels):
    def pct(idxs):
        return 100.0 * len(idxs) / n_total
    print("speaker-disjoint split (no speaker appears in more than one split):")
    print("  train: %5.1f%% (%d files) | %d speakers: %s" % (
        pct(best["tr_idx"]), len(best["tr_idx"]), len(best["train_spk"]), best["train_spk"]))
    print("  valid: %5.1f%% (%d files) | %d speakers: %s" % (
        pct(best["va_idx"]), len(best["va_idx"]), len(best["val_spk"]), best["val_spk"]))
    print("  test : %5.1f%% (%d files) | %d speakers: %s" % (
        pct(best["te_idx"]), len(best["te_idx"]), len(best["test_spk"]), best["test_spk"]))
    print("  test class-distribution vs train (balanced target), L1=%.3f:" % best["dist_pen"])
    for c in range(NUM_CLASSES):
        print("    %-15s train %.3f | test %.3f" % (
            CLASSES[c], best["train_dist"][c], best["test_dist"][c]))
