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
- **Training rate is locked to 3.3 kHz only.** `TARGET_SR = 3333` is the single
  source of truth (`imu_kws/dataset.py`). On load, anything not at 3.3 kHz is
  resampled to 3.3 kHz; with `--strict-sr` it is instead skipped. The model never
  sees any other sample rate.
- Their waveform noise-augmentation path is GSC/16 kHz-specific and needs
  background-noise wavs, so we call `Preprocess(..., augment=False)` — only the
  (sample-rate-correct) LogMel + SpecAugment run. Light waveform augmentation
  (gain / time-shift / gaussian noise) is done in our own dataset.

### Front-end (waveform preprocessing + features)

The front-end is configurable (adapted from the provided script):

- **Waveform preproc** (`--preproc`, numpy/scipy, per clip): high-pass Butterworth
  (80 Hz) + normalization + `crop_max_energy` (takes the highest-energy fixed
  window instead of padding to several seconds). Choices: `hp_peak_crop` (default),
  `hp_peak_trim_crop` (adds silence trim), `no_hp`, `hp_rms_crop`.
- **Window length** (`--window-seconds`, default 2.5 s) sets the crop size.
- **Feature** (`--feature`): `logmel_40` (default), `mel_linear`, or `pcen`. All are
  band-limited to `--fmin`/`--fmax` (default **50-500 Hz**) so the 40 mel bins land
  on the informative IMU band, and are mean/var normalized per utterance.
- `logmel_30`, `logmel_64`, `logmel_deltas` are **not** usable with the unmodified
  BC-ResNet (it requires 40 mels / 1 channel) and raise a clear error.
- `--feature vendored` falls back to the original vendored LogMel + length-percentile path.

### Accuracy-oriented choices (not brute force)
- **Evaluation defaults to Leave-One-Speaker-Out (LOSO) cross-validation**: each
  speaker is held out as the test set in turn, the model is trained on the others,
  and results are pooled across all folds (so every recording is tested exactly
  once, on a speaker the model never trained on). A second *representative* speaker
  is held out per fold for validation/early-stopping, so train/val/test stay
  speaker-disjoint. Reported as per-fold mean +/- std and a pooled confusion matrix.
- A single **speaker-disjoint 80/10/10 split** is still available with `--loso False`
  (test set class-balanced to the train distribution via a randomized speaker search).
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

## Labels in the manifest CSV

The label is the keyword that prefixes each file name after `<speaker>/`
(`begin_activity`, `stop_activity`, `wake_up`, `end`, `emergency`). You can
materialize it as a `label` column in the manifest and have training read it from
there (it takes precedence over filename parsing; rows labeled `unknown` are
skipped). Same parser is used either way, so they stay consistent.

Add the column locally:

```bash
python bcresnet_imu_kws/add_labels_to_manifest.py manifest.csv manifest_labeled.csv
```

...or generate/commit it on the volume:

```bash
# augment an existing CSV on the volume
modal run bcresnet_imu_kws/make_manifest_modal.py --in-csv /data/manifest.csv
# or build a fresh labeled manifest by scanning the recordings
modal run bcresnet_imu_kws/make_manifest_modal.py
```

Then train against it:

```bash
modal run bcresnet_imu_kws/train_modal.py --manifest /data/manifest_labeled.csv
```

The resulting CSV adds one column:

```
source_csv,wav_path,filtered_path,label
arnav/begin_activity_..._accel.csv,arnav/begin_activity_..._accel.wav,arnav/begin_activity_..._accel_50_500hz.wav,begin_activity
arnav/emergency_..._accel.csv,arnav/emergency_..._accel.wav,arnav/emergency_..._accel_50_500hz.wav,emergency
```

## Usage

```bash
pip install modal && modal token new      # one-time

# 1) sanity-check the data (counts, sample rates, clip lengths)
modal run bcresnet_imu_kws/inspect_data_modal.py

# 2) train (writes to the kws-imu-models volume, commits on every best epoch)
modal run bcresnet_imu_kws/train_modal.py --tau 3 --epochs 120

# default is Leave-One-Speaker-Out CV (one model per held-out speaker).
# to use a single speaker-disjoint 80/10/10 split instead:
modal run bcresnet_imu_kws/train_modal.py --tau 3 --loso False

# smaller / edge-tiny model:
modal run bcresnet_imu_kws/train_modal.py --tau 1 --epochs 150

# strict 3.3 kHz only: skip (don't resample) any file not stored at 3333 Hz
modal run bcresnet_imu_kws/train_modal.py --strict-sr
```

Outputs land in the `kws-imu-models` volume at `/models/bcresnet_imu/`. LOSO writes
one checkpoint per held-out speaker (`bcresnet_imu_fold_<speaker>.pt`) plus
`loso_summary.json` (per-fold metrics, pooled accuracy/macro-F1, pooled confusion
matrix). The single-split mode writes `bcresnet_imu_best.pt` + `summary.json`.

Download them locally with:

```bash
modal volume get kws-imu-models bcresnet_imu/loso_summary.json
modal volume get kws-imu-models bcresnet_imu/summary.json
modal volume get kws-imu-models bcresnet_imu/bcresnet_imu_best.pt
```

## Utility Scripts

### csv_to_wav.py -- Convert CSV accelerometer data to WAV

Reads the `accel_z` column (or any column you specify) from a CSV file and writes
a mono `.wav` at 3333 Hz (the pipeline target sample rate). The signal is
normalized to [-1, 1] by default.

```bash
# Basic usage (writes <input_stem>.wav next to the CSV)
python bcresnet_imu_kws/csv_to_wav.py --input recording.csv

# Specify output path and column name
python bcresnet_imu_kws/csv_to_wav.py \
    --input data/raw_accel.csv \
    --output data/accel_z.wav \
    --column accel_z

# Skip normalization (keep raw amplitude)
python bcresnet_imu_kws/csv_to_wav.py --input recording.csv --no-normalize
```

Options:
- `--input / -i` (required): path to the input CSV
- `--output / -o`: output WAV path (default: same stem as input + `.wav`)
- `--column / -c`: column name to extract (default: `accel_z`)
- `--sample-rate / -sr`: output sample rate (default: 3333)
- `--no-normalize`: skip peak normalization to [-1, 1]

### preproc_demo.py -- Preprocessing pipeline visualization

Takes a single `.wav` file and runs the full IMU preprocessing pipeline
step-by-step, producing a multi-panel plot that proves each DSP hyperparameter
choice visually:

1. Raw waveform (time domain)
2. After high-pass Butterworth filter (HP_CUTOFF=25 Hz)
3. After peak normalization (scaled to [-1, 1])
4. After `crop_max_energy` (highest-energy fixed window)
5. STFT spectrogram (annotated with Nyquist and HP cutoff lines)
6. Log-mel spectrogram (40 bins, fmin=40 Hz, fmax=1600 Hz annotated)
7. Per-frame energy distribution histogram
8. Summary table of all hyperparameter values

```bash
# Basic usage (saves preproc_demo.png in the current directory)
python bcresnet_imu_kws/preproc_demo.py --wav my_recording.wav

# Custom output path and crop window
python bcresnet_imu_kws/preproc_demo.py \
    --wav data/accel_z.wav \
    --output plots/pipeline_verification.png \
    --window-seconds 2.5 \
    --dpi 200
```

Options:
- `--wav / -w` (required): path to the input WAV (resampled to 3333 Hz if needed)
- `--output / -o`: output image path (default: `preproc_demo.png`)
- `--window-seconds`: crop window duration in seconds (default: 2.5)
- `--dpi`: output image resolution (default: 150)

**Typical workflow** (CSV to verified preprocessing):

```bash
# 1) Convert your accelerometer CSV to WAV
python bcresnet_imu_kws/csv_to_wav.py -i raw_data.csv -o signal.wav

# 2) Visualize and verify the full preprocessing pipeline
python bcresnet_imu_kws/preproc_demo.py -w signal.wav -o pipeline_check.png
```
