"""Add a ``label`` column to a recordings manifest CSV (runs locally, no Modal).

Usage:
    python bcresnet_imu_kws/add_labels_to_manifest.py input.csv [output.csv]

If ``output.csv`` is omitted, writes ``<input>_labeled.csv``. The label is the
keyword that prefixes each file name (begin_activity / stop_activity / wake_up /
end / emergency); unmatched rows get ``unknown``.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from imu_kws.manifest import augment_manifest


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    in_csv = argv[1]
    if len(argv) >= 3:
        out_csv = argv[2]
    else:
        base, ext = os.path.splitext(in_csv)
        out_csv = base + "_labeled" + (ext or ".csv")
    stats = augment_manifest(in_csv, out_csv)
    print("wrote %s" % stats["out_csv"])
    print("  rows: %d | labeled: %d | unknown: %d" % (
        stats["rows"], stats["labeled"], stats["unknown"]))
    print("  per-class: %s" % stats["per_class"])
    print("  columns: %s" % stats["columns"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
