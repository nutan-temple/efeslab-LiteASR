"""Class definitions and filename -> label parsing for the IMU KWS dataset.

Files look like::

    arnav/begin_activity_20260423_151702_accel.wav
    arnav/begin_activity_20260423_151702_accel_50_500hz.wav
    arnav/emergency_20260423_151758_accel.wav
    arnav/end_20260423_...._accel.wav

The keyword is the prefix before the ``_<YYYYMMDD>_<HHMMSS>_accel`` part. We match
by the longest known class prefix so multi-word keywords (e.g. ``begin_activity``)
are handled correctly and never confused with ``end``.
"""

import os

# Order here defines the integer label index (0..4).
CLASSES = [
    "begin_activity",
    "stop_activity",
    "wake_up",
    "end",
    "emergency",
]

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}
NUM_CLASSES = len(CLASSES)

# Longest-first so "begin_activity" wins over any shorter accidental prefix.
_CLASSES_BY_LEN = sorted(CLASSES, key=len, reverse=True)


def filename_to_class(path):
    """Return the class string for a wav path, or ``None`` if it does not match."""
    name = os.path.basename(str(path))
    for cls in _CLASSES_BY_LEN:
        if name.startswith(cls + "_"):
            return cls
    return None


def filename_to_label(path):
    """Return the integer label for a wav path, or ``None`` if unmatched (skip it)."""
    cls = filename_to_class(path)
    return CLASS_TO_IDX[cls] if cls is not None else None
