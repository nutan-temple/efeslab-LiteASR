"""Manifest helpers: add a ``label`` column to the recordings CSV.

The CSV looks like::

    source_csv,wav_path,filtered_path
    arnav/begin_activity_20260423_151702_accel.csv,arnav/...accel.wav,arnav/...accel_50_500hz.wav

The label is the keyword that prefixes the file name (after ``<speaker>/``), e.g.
``begin_activity``, ``stop_activity``, ``wake_up``, ``end``, ``emergency``. We reuse
the exact same parser the dataset uses (``filename_to_class``) so the CSV labels and
the training labels can never drift apart. Rows that don't match a known keyword get
``unknown`` (and are skipped at training time).
"""

import csv
import os

from .labels import filename_to_class

LABEL_COL = "label"
DEFAULT_PATH_COLS = ("filtered_path", "wav_path")


def label_from_path(rel_or_path):
    """Return the class string for a (relative or absolute) wav/csv path, else None."""
    return filename_to_class(rel_or_path)


def _ref_path(row, path_cols=DEFAULT_PATH_COLS):
    for col in path_cols:
        val = row.get(col)
        if val:
            return val
    return None


def augment_manifest(in_csv, out_csv, label_col=LABEL_COL, path_cols=DEFAULT_PATH_COLS):
    """Read ``in_csv``, add/overwrite a ``label`` column, write ``out_csv``.

    Returns a small stats dict.
    """
    with open(in_csv, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if label_col not in fieldnames:
        fieldnames = fieldnames + [label_col]

    labeled, unknown, per_class = 0, 0, {}
    for row in rows:
        ref = _ref_path(row, path_cols)
        cls = label_from_path(ref) if ref else None
        row[label_col] = cls if cls else "unknown"
        if cls:
            labeled += 1
            per_class[cls] = per_class.get(cls, 0) + 1
        else:
            unknown += 1

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "rows": len(rows),
        "labeled": labeled,
        "unknown": unknown,
        "per_class": per_class,
        "columns": fieldnames,
        "out_csv": out_csv,
    }


def build_manifest_by_scan(wav_dir, out_csv, label_col=LABEL_COL):
    """Walk ``wav_dir`` and build a fully-labeled manifest from scratch.

    For each band-pass file ``<spk>/<kw>_..._accel_50_500hz.wav`` we derive the raw
    wav path and the source csv path by string substitution, and the label from the
    file name.
    """
    rows = []
    per_class = {}
    for root, _, files in os.walk(wav_dir):
        for fn in sorted(files):
            if not fn.endswith("_50_500hz.wav"):
                continue
            filtered_rel = os.path.relpath(os.path.join(root, fn), wav_dir)
            wav_rel = filtered_rel.replace("_50_500hz.wav", ".wav")
            source_csv_rel = wav_rel[:-4] + ".csv" if wav_rel.endswith(".wav") else wav_rel
            cls = filename_to_class(fn) or "unknown"
            if cls != "unknown":
                per_class[cls] = per_class.get(cls, 0) + 1
            rows.append({
                "source_csv": source_csv_rel,
                "wav_path": wav_rel,
                "filtered_path": filtered_rel,
                label_col: cls,
            })

    fieldnames = ["source_csv", "wav_path", "filtered_path", label_col]
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "rows": len(rows),
        "labeled": sum(per_class.values()),
        "unknown": len(rows) - sum(per_class.values()),
        "per_class": per_class,
        "columns": fieldnames,
        "out_csv": out_csv,
    }
