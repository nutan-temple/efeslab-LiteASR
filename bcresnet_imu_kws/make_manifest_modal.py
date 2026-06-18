"""Generate a labeled manifest CSV on the Modal volume and commit it.

Two modes:
  * augment an existing CSV that already has source_csv,wav_path,filtered_path:
        modal run bcresnet_imu_kws/make_manifest_modal.py --in-csv /data/manifest.csv
  * build a fresh labeled manifest by scanning the recordings:
        modal run bcresnet_imu_kws/make_manifest_modal.py

The result (default ``/data/manifest_labeled.csv``) is written to the kws-imu-data
volume and committed, so training can use it with
``--manifest /data/manifest_labeled.csv``.
"""

import os

import modal

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

app = modal.App("bcresnet-imu-make-manifest")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .add_local_dir(PROJECT_DIR, remote_path="/root/app", copy=True)
)

raw_vol = modal.Volume.from_name("kws-imu-data", create_if_missing=False)
WAV_DIR = "/data/imu_data_wav_3333hz"


@app.function(image=image, volumes={"/data": raw_vol}, timeout=900)
def make_manifest(in_csv: str = None, out_csv: str = "/data/manifest_labeled.csv"):
    import sys

    sys.path.insert(0, "/root/app")
    from imu_kws.manifest import augment_manifest, build_manifest_by_scan

    if in_csv and os.path.isfile(in_csv):
        print("augmenting existing manifest:", in_csv)
        stats = augment_manifest(in_csv, out_csv)
    else:
        if in_csv:
            print("in_csv %s not found -> scanning %s instead" % (in_csv, WAV_DIR))
        else:
            print("scanning", WAV_DIR)
        stats = build_manifest_by_scan(WAV_DIR, out_csv)

    raw_vol.commit()
    print("wrote + committed:", stats["out_csv"])
    print("  rows: %d | labeled: %d | unknown: %d" % (
        stats["rows"], stats["labeled"], stats["unknown"]))
    print("  per-class:", stats["per_class"])
    print("  columns:", stats["columns"])
    return stats


@app.local_entrypoint()
def main(in_csv: str = "", out_csv: str = "/data/manifest_labeled.csv"):
    make_manifest.remote(in_csv=(in_csv or None), out_csv=out_csv)
