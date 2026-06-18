# BC-ResNet IMU Keyword Spotting

Train a 5-class keyword-spotting model on **3333 Hz IMU (accel-Z) `.wav`** data,
reusing the **Qualcomm BC-ResNet** code *without modifying it*. Training runs on
**Modal** and reads/writes Modal Volumes.

Classes: `begin_activity`, `stop_activity`, `wake_up`, `end`, `emergency`.

## Why it's built this way

- **`vendor/bcresnet/`** is an exact, byte-for-byte copy of
  [Qualcomm-AI-research/bcresnet](https://github.com/Qualcomm-AI-research/bcresnet).
  It is **never edited**. We only *import* `BCResNets` (the model) and `Preprocess`
  (their LogMel + SpecAugment) from it.
- **`imu_kws/`** is all-new glue code: label parsing, the IMU dataset, and the
  train/eval loop.

### Key adaptations (no architecture changes)
- `n_mels = 40` is kept **fixed** — the BC-ResNet architecture requires it
  (`SubSpectralNorm` uses 5 groups and the classifier uses a 5-wide frequency
  kernel, so the frequency axis must reduce `40 -> 20 -> 10 -> 5 -> 1`).
- The mel front-end is retuned for 3333 Hz audio: `n_fft=512`, `win_length=256`
  (~77 ms), `hop_length=64` (~19 ms).
- Their waveform noise-augmentation path is GSC/16 kHz-specific and needs
  background-noise wavs, so we call `Preprocess(..., augment=False)` — only the
  (sample-rate-correct) LogMel + SpecAugment run. Light waveform augmentation
  (gain / time-shift / gaussian noise) is done in our own dataset.

### Accuracy-oriented choices (not brute force)
- Stratified 70/15/15 split.
- Class-weighted cross-entropy (default) or an optional balanced sampler to handle
  the imbalance (`begin_activity=213 ... emergency=73`).
- Cosine LR schedule with warmup (same shape as the original `main.py`).
- Model selection + early stopping on **validation macro-F1** (robust to imbalance).
- Data-driven fixed input length (99th-percentile of clip lengths) — only crops a
  few outliers, pads the rest.
- The stray malformed file (and anything unreadable/unparseable) is skipped safely.

## Expected volume layout

`kws-imu-data` volume, mounted at `/data`:

```
/data/imu_data_wav_3333hz/
  arnav/
    begin_activity_20260423_151702_accel.wav
    begin_activity_20260423_151702_accel_50_500hz.wav   # band-pass 50-500 Hz
    emergency_20260423_151758_accel.wav
    ...
```

By default we train on the band-pass-filtered wavs (`*_50_500hz.wav`). Use
`--use-filtered False` to train on the raw wavs instead. If you have the manifest
CSV (`source_csv,wav_path,filtered_path`) on the volume, pass `--manifest /data/...csv`.

## Usage

```bash
pip install modal && modal token new      # one-time

# 1) sanity-check the data (counts, sample rates, clip lengths)
modal run bcresnet_imu_kws/inspect_data_modal.py

# 2) train (writes to the kws-imu-models volume, commits on every best epoch)
modal run bcresnet_imu_kws/train_modal.py --tau 3 --epochs 120

# smaller / edge-tiny model:
modal run bcresnet_imu_kws/train_modal.py --tau 1 --epochs 150
```

Outputs land in the `kws-imu-models` volume at `/models/bcresnet_imu/`:
`bcresnet_imu_best.pt` (state dict + class list + mel config) and `summary.json`
(test accuracy, macro-F1, per-class report, confusion matrix).

Download them locally with:

```bash
modal volume get kws-imu-models bcresnet_imu/summary.json
modal volume get kws-imu-models bcresnet_imu/bcresnet_imu_best.pt
```
