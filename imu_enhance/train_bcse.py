#!/usr/bin/env python3
"""
train_bcse.py - supervised Bone-Conduction Speech Enhancement (BCSE) for IMU audio.

THE point of this script: classical DSP (imu_enhance.py) is bounded by Nyquist. At a fixed
3333 Hz IMU rate you physically never capture the >1.6 kHz fricative/consonant band, so no filter
can recover it -- and on normal-volume speakers the formants you DO capture sit under the noise.

But you have paired data: for every utterance, an IMU reconstruction AND a clean microphone
recording. That is a supervised training set. This model LEARNS the mapping

        IMU low-band log-mel  -->  clean-mic full-band log-mel

which does two things classical DSP cannot:
  * denoise using a learned model of what real speech looks like (it will not turn noise into
    fake harmonics the way MMSE does -> fewer STT hallucinations), and
  * bandwidth-extend: hallucinate the missing high band CONDITIONED on real speech structure,
    supervised by the mic, so the extension is speech-consistent rather than invented.

Architecture: small 1-D-time / 2-D U-Net over mel frames (frequency x time), L1 + multi-scale
spectral loss. Inference reconstructs a 16 kHz waveform via mel-pseudo-inverse + Griffin-Lim
(swap in a neural vocoder like HiFi-GAN for production quality).

This is a runnable SCAFFOLD: wire `pair_index.csv` to your data and it trains. Requires torch
(already in requirements.txt) + librosa + soundfile + numpy.

Data contract -- pair_index.csv with columns:
    imu_wav,mic_wav            # one row per aligned utterance
(imu_wav = output of imu_enhance.py at --out-rate 16000; mic_wav = clean reference, 16 kHz)

CLI:
  python train_bcse.py train --index pairs.csv --epochs 60 --out ckpt.pt
  python train_bcse.py enhance --ckpt ckpt.pt --in imu_clip.wav --out enhanced.wav
"""
from __future__ import annotations
import argparse
import os
import numpy as np

SR = 16000
N_FFT = 512
HOP = 128
N_MELS = 80
F_MIN = 0.0
F_MAX = SR / 2


# ============================================================ features (numpy/librosa)
def wav_to_logmel(path_or_array, sr=SR):
    import librosa
    if isinstance(path_or_array, str):
        y, _ = librosa.load(path_or_array, sr=sr)
    else:
        y = np.asarray(path_or_array, dtype=np.float32)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP,
                                       n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX, power=2.0)
    return np.log(S + 1e-6).astype(np.float32)            # (n_mels, T)


def logmel_to_wav(logmel, sr=SR, n_iter=60):
    """Approximate inversion: log-mel -> power-mel -> linear (NNLS mel pinv) -> Griffin-Lim.
    Good enough to feed an STT; replace with a neural vocoder for listening quality."""
    import librosa
    mel = np.exp(np.asarray(logmel, np.float32)) - 1e-6
    mel = np.maximum(mel, 0.0)
    lin = librosa.feature.inverse.mel_to_stft(mel, sr=sr, n_fft=N_FFT, power=2.0,
                                              fmin=F_MIN, fmax=F_MAX)
    y = librosa.griffinlim(lin, n_iter=n_iter, hop_length=HOP, n_fft=N_FFT)
    p = np.max(np.abs(y)) + 1e-9
    return (y / p).astype(np.float32)


# ============================================================ dataset
def _make_dataset(index_csv):
    import csv
    import torch
    from torch.utils.data import Dataset

    class PairDS(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            imu, mic = self.rows[i]["imu_wav"], self.rows[i]["mic_wav"]
            xi = wav_to_logmel(imu)
            xt = wav_to_logmel(mic)
            T = min(xi.shape[1], xt.shape[1])              # time-align (both 16 kHz, same hop)
            xi, xt = xi[:, :T], xt[:, :T]
            return torch.from_numpy(xi), torch.from_numpy(xt)

    with open(index_csv) as f:
        rows = [r for r in csv.DictReader(f) if r.get("imu_wav") and r.get("mic_wav")]
    if not rows:
        raise SystemExit(f"no usable rows in {index_csv} (need columns imu_wav, mic_wav)")
    return PairDS(rows)


def _collate(batch):
    """Pad variable-length mel frames to the longest in the batch; return a time mask."""
    import torch
    Tmax = max(b[0].shape[1] for b in batch)
    xs, ys, masks = [], [], []
    for xi, xt in batch:
        T = xi.shape[1]
        pad = Tmax - T
        xs.append(torch.nn.functional.pad(xi, (0, pad)))
        ys.append(torch.nn.functional.pad(xt, (0, pad)))
        m = torch.zeros(Tmax)
        m[:T] = 1.0
        masks.append(m)
    return torch.stack(xs), torch.stack(ys), torch.stack(masks)


# ============================================================ model: mel U-Net
def _build_model():
    import torch
    import torch.nn as nn

    class ConvBlock(nn.Module):
        def __init__(self, ci, co):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.GELU(),
                nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.GELU(),
            )

        def forward(self, x):
            return self.net(x)

    class MelUNet(nn.Module):
        """2-D U-Net over (freq x time). Input = IMU log-mel, output = predicted RESIDUAL added to
        the input (so the net learns the correction: denoise + fill the high band)."""
        def __init__(self, base=32):
            super().__init__()
            self.e1 = ConvBlock(1, base)
            self.e2 = ConvBlock(base, base * 2)
            self.e3 = ConvBlock(base * 2, base * 4)
            self.pool = nn.MaxPool2d(2)
            self.mid = ConvBlock(base * 4, base * 8)
            self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
            self.d3 = ConvBlock(base * 8, base * 4)
            self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
            self.d2 = ConvBlock(base * 4, base * 2)
            self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
            self.d1 = ConvBlock(base * 2, base)
            self.head = nn.Conv2d(base, 1, 1)

        @staticmethod
        def _cat(up, skip):
            # crop BOTH to common (min) H,W before concat: ConvTranspose on odd-pooled dims can
            # produce a feature either larger or smaller than the skip connection.
            import torch
            h = min(up.shape[-2], skip.shape[-2])
            w = min(up.shape[-1], skip.shape[-1])
            return torch.cat([up[..., :h, :w], skip[..., :h, :w]], 1)

        def forward(self, x):                              # x: (B, n_mels, T)
            import torch.nn.functional as F
            inp = x.unsqueeze(1)                           # (B,1,F,T)
            e1 = self.e1(inp)
            e2 = self.e2(self.pool(e1))
            e3 = self.e3(self.pool(e2))
            m = self.mid(self.pool(e3))
            d3 = self.d3(self._cat(self.up3(m), e3))
            d2 = self.d2(self._cat(self.up2(d3), e2))
            d1 = self.d1(self._cat(self.up1(d2), e1))
            res = self.head(d1)
            # restore exact input size (interpolate if pooling/cropping changed it)
            if res.shape[-2:] != inp.shape[-2:]:
                res = F.interpolate(res, size=inp.shape[-2:], mode="bilinear", align_corners=False)
            return (inp + res).squeeze(1)                  # residual mapping

    return MelUNet()


def _loss_fn():
    import torch
    import torch.nn as nn
    l1 = nn.L1Loss(reduction="none")

    def loss(pred, target, mask):
        m = mask.unsqueeze(1)                              # (B,1,T) over time
        per = l1(pred, target)                             # (B,F,T)
        per = per * m
        base = per.sum() / (m.sum() * pred.shape[1] + 1e-9)
        # extra weight on the high mel bands (the bandwidth-extension region the IMU lacks)
        hi = pred.shape[1] // 2
        hi_per = l1(pred[:, hi:], target[:, hi:]) * m
        hi_term = hi_per.sum() / (m.sum() * (pred.shape[1] - hi) + 1e-9)
        return base + 0.5 * hi_term

    return loss


# ============================================================ train / enhance
def train(args):
    import torch
    from torch.utils.data import DataLoader, random_split

    ds = _make_dataset(args.index)
    n_val = max(1, int(len(ds) * 0.1))
    tr, va = random_split(ds, [len(ds) - n_val, n_val],
                          generator=torch.Generator().manual_seed(0))
    dl_tr = DataLoader(tr, batch_size=args.batch, shuffle=True, collate_fn=_collate)
    dl_va = DataLoader(va, batch_size=args.batch, shuffle=False, collate_fn=_collate)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = _loss_fn()

    best = float("inf")
    for ep in range(1, args.epochs + 1):
        model.train()
        tl = 0.0
        for xi, xt, m in dl_tr:
            xi, xt, m = xi.to(device), xt.to(device), m.to(device)
            opt.zero_grad()
            out = loss_fn(model(xi), xt, m)
            out.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tl += out.item() * xi.size(0)
        sched.step()

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xi, xt, m in dl_va:
                xi, xt, m = xi.to(device), xt.to(device), m.to(device)
                vl += loss_fn(model(xi), xt, m).item() * xi.size(0)
        tl /= len(tr)
        vl /= len(va)
        print(f"epoch {ep:3d}/{args.epochs}  train {tl:.4f}  val {vl:.4f}")
        if vl < best:
            best = vl
            torch.save({"model": model.state_dict(), "cfg": dict(
                sr=SR, n_fft=N_FFT, hop=HOP, n_mels=N_MELS)}, args.out)
            print(f"  saved {args.out} (val {vl:.4f})")
    print(f"done. best val {best:.4f} -> {args.out}")


def enhance(args):
    import torch
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = _build_model()
    model.load_state_dict(ckpt["model"])
    model.eval()
    xi = wav_to_logmel(args.inp)
    with torch.no_grad():
        pred = model(torch.from_numpy(xi).unsqueeze(0)).squeeze(0).numpy()
    y = logmel_to_wav(pred, n_iter=args.griffin_iters)
    import soundfile as sf
    sf.write(args.out, y, SR)
    print(f"enhanced {args.inp} -> {args.out} ({len(y)/SR:.2f}s @ {SR}Hz, "
          f"mel {xi.shape}->{pred.shape})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train BCSE on paired (IMU, mic) data")
    t.add_argument("--index", required=True, help="pair_index.csv with imu_wav,mic_wav columns")
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--out", default="bcse_ckpt.pt")
    t.set_defaults(func=train)

    e = sub.add_parser("enhance", help="enhance one IMU wav with a trained checkpoint")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--in", dest="inp", required=True, help="IMU wav (16 kHz)")
    e.add_argument("--out", default="enhanced.wav")
    e.add_argument("--griffin-iters", type=int, default=60)
    e.set_defaults(func=enhance)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
