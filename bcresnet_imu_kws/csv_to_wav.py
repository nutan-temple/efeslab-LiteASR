#!/usr/bin/env python3
"""Convert the accel_z column of a CSV file to a .wav at 3333 Hz.

Reads time-series accelerometer data from a CSV, extracts the specified column
(default ``accel_z``), normalizes to [-1, 1], and writes a 32-bit float WAV at
the pipeline target sample rate (3333 Hz).

Example
-------
    python bcresnet_imu_kws/csv_to_wav.py \
        --input recording.csv \
        --output recording.wav \
        --column accel_z
"""

import argparse
import sys

import numpy as np
import soundfile as sf

# Pipeline target sample rate (single source of truth in imu_kws/dataset.py).
TARGET_SR = 3333


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert a CSV accel_z column to a .wav file at 3333 Hz.")
    parser.add_argument("--input", "-i", required=True,
                        help="Path to the input CSV file.")
    parser.add_argument("--output", "-o", default=None,
                        help="Path to the output .wav file. "
                             "Defaults to <input_stem>.wav in the same directory.")
    parser.add_argument("--column", "-c", default="accel_z",
                        help="Name of the column to extract (default: accel_z).")
    parser.add_argument("--sample-rate", "-sr", type=int, default=TARGET_SR,
                        help="Output sample rate in Hz (default: %d)." % TARGET_SR)
    parser.add_argument("--no-normalize", action="store_true",
                        help="Skip normalization to [-1, 1].")
    return parser.parse_args(argv)


def read_column(csv_path, column):
    """Read a single column from a CSV as a float32 numpy array.

    Uses pandas if available for robust parsing; falls back to the stdlib csv
    module otherwise.
    """
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        if column not in df.columns:
            raise ValueError(
                "Column '%s' not found in CSV. Available columns: %s"
                % (column, list(df.columns)))
        return df[column].to_numpy(dtype=np.float32)
    except ImportError:
        import csv as csvmod
        with open(csv_path, newline="") as f:
            reader = csvmod.DictReader(f)
            if column not in (reader.fieldnames or []):
                raise ValueError(
                    "Column '%s' not found in CSV. Available columns: %s"
                    % (column, reader.fieldnames))
            values = [float(row[column]) for row in reader]
        return np.array(values, dtype=np.float32)


def main(argv=None):
    args = parse_args(argv)

    # Determine output path
    output_path = args.output
    if output_path is None:
        import os
        stem = os.path.splitext(args.input)[0]
        output_path = stem + ".wav"

    # Read the column
    data = read_column(args.input, args.column)
    if data.size == 0:
        print("ERROR: column '%s' is empty." % args.column, file=sys.stderr)
        sys.exit(1)

    # Normalize to [-1, 1]
    if not args.no_normalize:
        peak = np.abs(data).max()
        if peak > 0:
            data = data / peak

    # Write wav
    sf.write(output_path, data, args.sample_rate, subtype="FLOAT")
    print("Wrote %s (%d samples, %.3f s @ %d Hz)"
          % (output_path, len(data), len(data) / args.sample_rate, args.sample_rate))


if __name__ == "__main__":
    main()
