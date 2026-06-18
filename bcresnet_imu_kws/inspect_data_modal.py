"""Quick data sanity-check on Modal before training.

Run:
    modal run bcresnet_imu_kws/inspect_data_modal.py
    modal run bcresnet_imu_kws/inspect_data_modal.py --use-filtered False

Prints per-class counts, sample rates, and waveform-length percentiles so you can
confirm the labels parse correctly and choose a sensible fixed input length.
"""

import os

import modal

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

app = modal.App("bcresnet-imu-inspect")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.3.1", "torchaudio==2.3.1", "numpy<2", "soundfile")
    .add_local_dir(PROJECT_DIR, remote_path="/root/app", copy=True)
)

raw_vol = modal.Volume.from_name("kws-imu-data", create_if_missing=False)
WAV_DIR = "/data/imu_data_wav_3333hz"


@app.function(image=image, volumes={"/data": raw_vol}, timeout=600)
def inspect(use_filtered: bool = True, manifest: str = None):
    import collections
    import sys

    import numpy as np
    import torchaudio

    sys.path.insert(0, "/root/app")
    from imu_kws.dataset import build_index
    from imu_kws.labels import IDX_TO_CLASS

    paths, labels, skipped = build_index(WAV_DIR, use_filtered, manifest)
    counts = collections.Counter(IDX_TO_CLASS[l] for l in labels)

    lengths, srs = [], collections.Counter()
    for p in paths:
        try:
            info = torchaudio.info(p)
            lengths.append(info.num_frames)
            srs[info.sample_rate] += 1
        except Exception:
            pass

    print("WAV_DIR:", WAV_DIR, "| use_filtered:", use_filtered)
    print("usable files:", len(paths))
    print("per-class:", dict(counts))
    print("sample_rates:", dict(srs))
    if lengths:
        a = np.asarray(lengths)
        print("length frames  min/median/p95/p99/max: %d / %d / %d / %d / %d" % (
            a.min(), int(np.median(a)), int(np.percentile(a, 95)),
            int(np.percentile(a, 99)), a.max()))
        print("length seconds min/median/p99/max:      %.2f / %.2f / %.2f / %.2f" % (
            a.min() / 3333.0, np.median(a) / 3333.0,
            np.percentile(a, 99) / 3333.0, a.max() / 3333.0))
    print("skipped:", len(skipped))
    for item in skipped[:10]:
        print("  -", item)
    return {"usable": len(paths), "per_class": dict(counts), "skipped": len(skipped)}


@app.local_entrypoint()
def main(use_filtered: bool = True, manifest: str = ""):
    inspect.remote(use_filtered=use_filtered, manifest=(manifest or None))
