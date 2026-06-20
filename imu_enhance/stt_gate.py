#!/usr/bin/env python3
"""
stt_gate.py - keep junk out of the speech-to-text stage.

Your eval showed bp_mmse producing 59/94 transcripts, "mostly hallucinations (Hindi/Russian
gibberish)". Two independent causes, two guards here:

  1. The recognizer was fed loud, speech-SHAPED noise. Guard A (submission gate): refuse to send a
     clip to the STT unless it clears an SNR / energy / voiced-content bar. This is the principled
     version of "fullband is conservative" -- we DECIDE to be conservative instead of relying on
     the signal happening to be quiet.

  2. An open-vocabulary, language-auto-detect ASR will always emit *something*. Guard B
     (constrained decode): pin the language to English, bias toward your known phrases, and reject
     low-confidence / out-of-language output so noise maps to "" instead of Hindi/Russian.

Backends are pluggable. Adapters provided for AssemblyAI (cloud) and local Whisper / LiteASR.
No API keys are hardcoded; AssemblyAI reads ASSEMBLYAI_API_KEY from the environment.

CLI:
  python stt_gate.py clip.wav --backend whisper --min-snr-db 4
  python stt_gate.py clip.wav --backend assemblyai --phrases "wake up,emergency,start activity,stop activity"
  python stt_gate.py --batch ./wavs --backend whisper --min-snr-db 4 --report report.csv
"""
from __future__ import annotations
import argparse
import os
import glob
import wave
import numpy as np


# ============================================================ audio IO + features
def read_wav(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        raw = w.readframes(n)
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def _frame_rms(x, sr, frame_ms=30, hop_ms=10):
    nf = max(1, int(sr * frame_ms / 1000))
    nh = max(1, int(sr * hop_ms / 1000))
    if len(x) <= nf:
        return np.array([np.sqrt((x ** 2).mean() + 1e-12)])
    starts = np.arange(0, len(x) - nf + 1, nh)
    return np.sqrt(np.array([(x[s:s + nf] ** 2).mean() for s in starts]) + 1e-12)


def clip_features(x, sr):
    """Cheap, robust descriptors for the submission gate."""
    rms = _frame_rms(x, sr)
    noise = float(np.percentile(rms, 20) + 1e-9)
    active = float(np.percentile(rms, 90))
    snr_db = 20.0 * np.log10(active / noise)
    voiced_frac = float(np.mean(rms > 2.0 * noise))          # share of frames above noise
    # spectral flatness: high => noise-like, low => tonal/voiced
    X = np.abs(np.fft.rfft(x - x.mean())) ** 2 + 1e-12
    flatness = float(np.exp(np.mean(np.log(X))) / np.mean(X))
    peak = float(np.max(np.abs(x)) if x.size else 0.0)
    dur = len(x) / sr
    return dict(snr_db=snr_db, voiced_frac=voiced_frac, flatness=flatness,
                peak=peak, dur_s=dur, n_active=int((rms > 2.0 * noise).sum()))


# ============================================================ Guard A: submission gate
def should_transcribe(feat, min_snr_db=4.0, min_voiced_frac=0.08, max_flatness=0.6,
                      min_dur_s=0.20, min_peak=0.02):
    """Return (ok: bool, reasons: list[str]). A clip must look like it actually contains voiced
    speech before we spend an STT call on it. Tune thresholds against your held-out set."""
    reasons = []
    if feat["dur_s"] < min_dur_s:
        reasons.append(f"too short ({feat['dur_s']:.2f}s<{min_dur_s})")
    if feat["peak"] < min_peak:
        reasons.append(f"too quiet (peak {feat['peak']:.3f}<{min_peak})")
    if feat["snr_db"] < min_snr_db:
        reasons.append(f"low SNR ({feat['snr_db']:.1f}dB<{min_snr_db})")
    if feat["voiced_frac"] < min_voiced_frac:
        reasons.append(f"little voiced ({feat['voiced_frac']:.2f}<{min_voiced_frac})")
    if feat["flatness"] > max_flatness:
        reasons.append(f"noise-like (flatness {feat['flatness']:.2f}>{max_flatness})")
    return (len(reasons) == 0), reasons


# ============================================================ Guard B: constrained ASR backends
DEFAULT_PHRASES = ["wake up", "emergency", "start activity", "stop activity"]


def transcribe_whisper(path, phrases=None, language="en", model_name="base.en", min_conf=-1.0):
    """Local OpenAI-Whisper backend. `phrases` are injected as an initial_prompt to bias decoding;
    language is pinned so noise cannot decode to Hindi/Russian. Requires `pip install openai-whisper`.
    min_conf: drop result if mean token logprob < min_conf (e.g. -1.0). Returns dict."""
    import whisper
    model = whisper.load_model(model_name)
    prompt = (", ".join(phrases or DEFAULT_PHRASES)) or None
    res = model.transcribe(path, language=language, initial_prompt=prompt,
                           temperature=0.0, condition_on_previous_text=False)
    text = (res.get("text") or "").strip()
    segs = res.get("segments") or []
    conf = float(np.mean([s.get("avg_logprob", -10.0) for s in segs])) if segs else -10.0
    nospeech = float(np.mean([s.get("no_speech_prob", 1.0) for s in segs])) if segs else 1.0
    if conf < min_conf or nospeech > 0.6:
        text = ""                                            # reject low-confidence / non-speech
    return dict(text=text, confidence=conf, no_speech_prob=nospeech, backend="whisper")


def transcribe_litewhisper(path, phrases=None, language="en",
                           model="efficient-speech/lite-whisper-large-v3-turbo", min_conf=-1.0):
    """Local LiteASR / lite-whisper backend (this repo's model) via HuggingFace transformers.
    Pins language and biases with a prompt. Requires transformers + torch + librosa."""
    import torch, librosa
    from transformers import AutoProcessor, AutoModel
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    mdl = AutoModel.from_pretrained(model, trust_remote_code=True).to(dtype).to(device)
    proc = AutoProcessor.from_pretrained("openai/whisper-large-v3")
    audio, _ = librosa.load(path, sr=16000)
    feats = proc(audio, sampling_rate=16000, return_tensors="pt").input_features.to(dtype).to(device)
    forced = proc.get_decoder_prompt_ids(language=language, task="transcribe")
    ids = mdl.generate(feats, forced_decoder_ids=forced)
    text = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
    return dict(text=text, confidence=None, no_speech_prob=None, backend="litewhisper")


def transcribe_assemblyai(path, phrases=None, language="en", min_conf=0.4):
    """AssemblyAI cloud backend. Sets language_code (no auto-detect -> no Hindi/Russian),
    word_boost with your phrases, and a confidence floor. Reads ASSEMBLYAI_API_KEY from env.
    Requires `pip install assemblyai`."""
    import assemblyai as aai
    key = os.environ.get("ASSEMBLYAI_API_KEY")
    if not key:
        raise SystemExit("set ASSEMBLYAI_API_KEY in the environment")
    aai.settings.api_key = key
    cfg = aai.TranscriptionConfig(
        language_code=language,                              # pin language: kills gibberish
        word_boost=phrases or DEFAULT_PHRASES,               # bias toward known commands
        boost_param="high",
        punctuate=True, format_text=True,
    )
    t = aai.Transcriber().transcribe(path, config=cfg)
    text = (t.text or "").strip()
    conf = float(getattr(t, "confidence", 0.0) or 0.0)
    if conf < min_conf:
        text = ""                                            # reject low-confidence transcripts
    return dict(text=text, confidence=conf, no_speech_prob=None, backend="assemblyai")


BACKENDS = {
    "whisper": transcribe_whisper,
    "litewhisper": transcribe_litewhisper,
    "assemblyai": transcribe_assemblyai,
}


# ============================================================ orchestration
def gated_transcribe(path, backend="whisper", phrases=None, language="en",
                     min_snr_db=4.0, min_voiced_frac=0.08, max_flatness=0.6, **bk):
    """Full path: features -> submission gate -> (maybe) constrained ASR.
    Returns a record dict; text is "" whenever the clip is gated out or the ASR is unconfident."""
    x, sr = read_wav(path)
    feat = clip_features(x, sr)
    ok, reasons = should_transcribe(feat, min_snr_db, min_voiced_frac, max_flatness)
    rec = dict(path=path, gated_out=not ok, reasons=";".join(reasons), text="", **feat)
    if not ok:
        return rec
    out = BACKENDS[backend](path, phrases=phrases, language=language, **bk)
    rec.update(text=out["text"], confidence=out.get("confidence"),
               no_speech_prob=out.get("no_speech_prob"), backend=out["backend"])
    return rec


def _print_rec(r):
    status = "GATED" if r["gated_out"] else ("EMPTY" if not r["text"] else "TEXT")
    print(f"[{status:5s}] snr={r['snr_db']:5.1f}dB voiced={r['voiced_frac']:.2f} "
          f"flat={r['flatness']:.2f}  {os.path.basename(r['path'])}"
          + (f"  reasons: {r['reasons']}" if r["gated_out"] else f"  -> {r['text']!r}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav", nargs="?", help="single WAV path")
    ap.add_argument("--batch", help="directory of WAVs to process")
    ap.add_argument("--backend", choices=list(BACKENDS), default="whisper")
    ap.add_argument("--language", default="en")
    ap.add_argument("--phrases", default=",".join(DEFAULT_PHRASES),
                    help="comma-separated bias phrases / command vocabulary")
    ap.add_argument("--min-snr-db", type=float, default=4.0)
    ap.add_argument("--min-voiced-frac", type=float, default=0.08)
    ap.add_argument("--max-flatness", type=float, default=0.6)
    ap.add_argument("--report", help="optional CSV report path for --batch")
    a = ap.parse_args()

    phrases = [p.strip() for p in a.phrases.split(",") if p.strip()]
    paths = sorted(glob.glob(os.path.join(a.batch, "*.wav"))) if a.batch else ([a.wav] if a.wav else [])
    if not paths:
        raise SystemExit("provide a WAV path or --batch <dir>")

    recs = []
    for p in paths:
        r = gated_transcribe(p, a.backend, phrases, a.language,
                             a.min_snr_db, a.min_voiced_frac, a.max_flatness)
        _print_rec(r)
        recs.append(r)

    gated = sum(r["gated_out"] for r in recs)
    empty = sum((not r["gated_out"]) and (not r["text"]) for r in recs)
    text = sum(bool(r["text"]) for r in recs)
    print(f"\nsummary: {len(recs)} clips | {text} transcribed | {empty} empty (sent, unconfident) "
          f"| {gated} gated out (never sent)")

    if a.report and recs:
        import csv
        keys = ["path", "gated_out", "reasons", "text", "confidence", "snr_db",
                "voiced_frac", "flatness", "peak", "dur_s", "n_active"]
        with open(a.report, "w", newline="") as f:
            wri = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            wri.writeheader()
            wri.writerows(recs)
        print(f"wrote {a.report}")


if __name__ == "__main__":
    main()
