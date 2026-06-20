# IMU Speech Enhancement (fixed 3333 Hz hardware)

Tooling to turn an IMU/accelerometer voice capture into something a speech-to-text (STT) engine
can transcribe, **without inventing words on noise**.

## The hardware constraint

The IMU samples at a **fixed 3333 Hz** (cannot be re-captured at a higher output data rate).
By Nyquist, the usable band is **~80–1600 Hz**. That carries pitch (F0) and the first two formants
but **none of the fricative/sibilant energy** (s, sh, f, t live at 2–8 kHz). On loud speakers the
formant structure alone is enough for STT; on **normal-volume speakers** the formants sit under the
sensor noise floor and there is nothing above 1.6 kHz to fall back on.

This caused the two failure modes observed in evaluation:

| Mode (old script) | Result | Root cause |
|---|---|---|
| `bp_mmse` (MMSE-LSA denoise + RMS-normalize) | 59/94 non-empty, **mostly hallucinations** (Hindi/Russian gibberish) | MMSE shapes broadband noise into speech-like spectra, then RMS-normalize boosts noise-only clips to full loudness → STT invents text |
| `fullband` (no denoise) | 15/94, honest but only the loud clips | conservative by accident; the weak normal-speaker signal never survives |

## What these scripts do about it

Two things classical processing **can** do within 1.6 kHz, and one thing only a learned model can do.

### 1. `imu_enhance.py` — improved DSP pipeline (numpy + scipy)

Drop-in successor to the old `imu_denoise.py`. Same CSV front-end, fixes the failures:

- **OM-LSA + MCRA denoiser** (replaces MMSE-LSA / Wiener / specsub). The gain is weighted by a
  per-bin **Speech-Presence Probability**, so noise-only bins collapse to a floor instead of being
  sculpted into fake harmonics. Far fewer STT hallucinations.
- **VAD gate to TRUE silence** — non-speech frames become hard zeros *before* normalization, so the
  STT receives real silence, not shaped noise.
- **SNR-conditional normalization** — quiet / noise-only clips are emitted as **silence** instead of
  being amplified to a fixed RMS target. Returns an SNR estimate so callers can refuse junk.
- **Per-axis band-pass before combine** + **SNR-weighted multi-axis combine** — squeezes extra dB
  out of normal-volume speech by letting the axis best-coupled to bone vibration dominate (plain PCA
  maximized gross-motion variance instead).

```bash
python imu_enhance.py raw.csv --axis snr --denoise omlsa --vad-gate --snr-norm --min-snr-db 3
# library:
#   t, x3 = load_axes("raw.csv")
#   sig, sr, snr = reconstruct_multi(t, x3, 3333, combine="snr", denoise="omlsa",
#                                    vad=True, snr_norm=True)
```

### 2. `stt_gate.py` — keep junk out of the recognizer

- **Guard A — submission gate**: a clip is only sent to the STT if it clears an SNR / voiced-fraction
  / spectral-flatness / duration bar. This is the principled version of "fullband is conservative":
  we *decide* to skip noise instead of hoping it stays quiet.
- **Guard B — constrained decode**: pin `language="en"` (no auto-detect → no Hindi/Russian),
  bias toward known command phrases (`word_boost` / Whisper `initial_prompt`), and reject
  low-confidence output so noise maps to `""`.
- Pluggable backends: local **Whisper**, this repo's **lite-whisper / LiteASR**, or **AssemblyAI**
  (reads `ASSEMBLYAI_API_KEY` from env — no keys in code).

```bash
python stt_gate.py clip.wav --backend whisper --min-snr-db 4
python stt_gate.py --batch ./wavs --backend assemblyai \
    --phrases "wake up,emergency,start activity,stop activity" --report report.csv
```

### 3. `train_bcse.py` — the real fix for normal speakers (PyTorch)

Classical DSP is bounded by Nyquist; it cannot recreate the >1.6 kHz band the hardware never
captured. But you have **paired (IMU, clean-mic) recordings** — a supervised training set. This
trains a small mel U-Net (≈1.9M params) that maps **IMU low-band log-mel → clean-mic full-band
log-mel**, doing both:

- **learned denoise** (won't fabricate noise-as-speech the way MMSE does), and
- **bandwidth extension** — reconstructs the missing high band *conditioned on real speech*,
  supervised by the mic, so the extension is speech-consistent rather than invented.

```bash
# pairs.csv columns: imu_wav,mic_wav  (both 16 kHz; imu_wav = imu_enhance.py output at --out-rate 16000)
python train_bcse.py train  --index pairs.csv --epochs 60 --out bcse_ckpt.pt
python train_bcse.py enhance --ckpt bcse_ckpt.pt --in imu_clip.wav --out enhanced.wav
```

Inference uses a mel-pseudo-inverse + Griffin-Lim vocoder (swap in HiFi-GAN for listening quality).

## Recommended end-to-end flow

```
CSV (IMU) ──imu_enhance.py──▶ enhanced.wav ──train_bcse.py enhance──▶ wideband.wav ──stt_gate.py──▶ text
            (OM-LSA + VAD + SNR gate)        (denoise + BW-extension)   (gate + constrained ASR)
```

For the **device-command** portion (`wake up`, `emergency`, `start activity`, `stop activity`),
prefer a closed-vocabulary keyword spotter over open-vocabulary ASR — it cannot hallucinate
out-of-vocabulary text. `stt_gate.py`'s phrase biasing is the lightweight stand-in.

## Evaluate on intelligibility, not "non-empty" count

"Non-empty transcript" rewards hallucination (that's why `bp_mmse` looked productive). Tune the
denoiser and gate thresholds against the time-aligned mic reference using **STOI/ESTOI** and
**SI-SDR**, then report **WER** against the known transcript.

## Dependencies

- `imu_enhance.py`: numpy, scipy (pandas for the CSV front-end)
- `stt_gate.py`: numpy + one backend (`openai-whisper`, or transformers/torch/librosa, or `assemblyai`)
- `train_bcse.py`: torch (in repo `requirements.txt`), librosa, soundfile

All three were smoke-tested: DSP pipeline on synthetic 3-axis data, the gate on voiced/noise/quiet
clips (loud noise correctly gated out), and the model train→enhance loop end-to-end on CPU.
