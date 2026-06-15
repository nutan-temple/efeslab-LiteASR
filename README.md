# Encoder-only SpQR Quantization for Moonshine

A quantization pipeline that applies the **real [SpQR](https://arxiv.org/abs/2306.03078)
algorithm** (Sparse-Quantized Representation) to the **encoder** of a
LiteASR-compressed Moonshine ASR model.

This is **not** a naive round-to-int8 scheme. The quantization engine is used
**verbatim** from the upstream [SpQR repository](https://github.com/Vahe1994/SpQR):

| File | Role |
|------|------|
| `spqr_engine.py` | `SPQRUtil` — GPTQ Hessian-aware error propagation + unstructured outlier detection |
| `quant_groups.py` | `Quantizer` — per-group scale/zero estimation and meta-quantization |
| `weight_permutation.py` | `get_permutation_order` — `act_order` / `spearman` column reordering |

These three files are copied **byte-for-byte** and are **not modified**.
`SPQRUtil.quantize()` performs, in order:

1. **Column permutation** (`act_order`) — quantize the most salient input
   features first, ordered by the Hessian diagonal.
2. **GPTQ error propagation** — each column is quantized and its rounding error
   is pushed onto the remaining columns, weighted by the inverse-Hessian
   Cholesky factor.
3. **Unstructured outlier detection** — individual weights whose quantization
   error exceeds a relative threshold are kept in **fp16** (a leave-one-out
   re-fit decides which weights are outliers); the rest are re-quantized
   without them.
4. **Meta-quantization** — the per-group scales and zero-points are themselves
   quantized (`qq_scale_bits` / `qq_zero_bits`).

## Why encoder-only

LiteASR and follow-up work show the Moonshine/Whisper **encoder** is the
runtime bottleneck (compute-bound, long sequences), while the decoder can be
compressed by other means. This pipeline therefore quantizes **only**
`model.model.encoder.layers` and leaves the decoder in full precision. The
compression report reflects this honestly (see below) — there is **no** flat
"4× total" claim.

## Pipeline

1. Load `usefulsensors/moonshine-base` via `MoonshineForConditionalGeneration`.
2. *(optional)* Load a LiteASR `.pth` and swap encoder sublayers for
   `LinearLowRank` wherever `weight1`/`weight2` keys exist.
3. Quantize each encoder layer's sublayers in **GPTQ-sequential groups** so
   later projections are calibrated against the **already-quantized**
   activations of earlier ones:

   ```
   [q_proj, k_proj, v_proj]  ->  [o_proj]  ->  [mlp.fc1]  ->  [mlp.fc2]
   ```

   * **`nn.Linear`** sublayers are quantized directly with `SPQRUtil`.
   * **`LinearLowRank`** sublayers have **both** low-rank factors quantized
     independently, each with its own Hessian: `W1`'s input is the layer input
     `x`, and `W2`'s input is `x @ W1`. The low-rank structure is preserved
     (the factors are quantized in place — not collapsed into a dense matrix).
4. Evaluate WER on LibriSpeech `test-clean` + `test-other`.
5. Report encoder-only compression.

### Calibration

Real speech from LibriSpeech `validation-clean` is run through the encoder conv
frontend to produce transformer-layer inputs; per-layer forward passes then
accumulate the Hessians used by SpQR. Variable-length clips are processed
individually (no zero-padding) to avoid polluting the Hessian.

## Correct WER evaluation

The evaluation in `eval_utils.py` **replicates the reference Moonshine recipe
exactly**, which is what makes the numbers comparable to the published baseline
of **3.38% test-clean / 9.38% test-other**:

* the model runs in **float16 on GPU** (float32 on CPU);
* inputs are moved *and cast* with `inputs.to(device, torch_dtype)`;
* generation is **token-limited**: `max_length = max(int(seq_lens * 6.5/16000), 10)`;
* hypotheses **and** references are normalized identically — lowercased,
  stripped, and **punctuation-removed** — before WER is computed.

> The last point matters: Moonshine emits cased, punctuated text while
> LibriSpeech references are upper-cased and unpunctuated. Without identical
> normalization the WER is massively inflated. (An earlier version that only
> lowercased — keeping punctuation — and ran in fp32 produced spuriously high
> WER.)

## Compression accounting

Because **only the encoder is quantized**, the script reports separate, honest
figures instead of a single headline ratio:

* the **encoder** shrinks ~3–4× — its effective bits/weight are
  `wbits + (qq_scale_bits + qq_zero_bits)/groupsize + outlier_fraction·32`
  (the conv/norm/bias tensors stay fp32);
* the **whole model** shrinks only in proportion to the encoder's share of the
  parameters, since the fp32 decoder dominates the remaining size.

## Usage

```bash
# 8-bit encoder-only SpQR with outlier detection, full LibriSpeech eval
python quantize_encoder_spqr.py \
    --pth_path lite-moonshine-moonshine-base_0.99:0.999.pth \
    --wbits 8 --groupsize 16 --perchannel \
    --qq_scale_bits 3 --qq_zero_bits 3 \
    --outlier_threshold 0.2 --permutation_order act_order \
    --nsamples 128

# Dense base encoder (no .pth), quick check on a few eval samples
python quantize_encoder_spqr.py --wbits 8 --max_eval_samples 50

# CPU smoke test: one layer, no eval
python quantize_encoder_spqr.py --nsamples 2 --skip_eval --max_layers 1
```

### Key arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--pth_path` | None | Optional LiteASR `.pth` (encoder `LinearLowRank`). Omit to quantize the dense base encoder. |
| `--wbits` | `8` | Base weight bits for SpQR. |
| `--groupsize` | `16` | Input-feature group size for scale/zero estimation. |
| `--perchannel` / `--no_perchannel` | on | Per-output-channel base quantization. |
| `--qq_scale_bits` / `--qq_zero_bits` | `3` / `3` | Meta-quantization bits for scales/zeros. |
| `--outlier_threshold` | `0.2` | `outlier_relative_threshold`; use `inf` to disable outliers. |
| `--permutation_order` | `act_order` | `identity`, `act_order`, or `spearman`. |
| `--nsamples` | `128` | Calibration clips from LibriSpeech `validation-clean`. |
| `--max_eval_samples` | None | Cap eval samples per split (None = full: 2620 clean / 2939 other). |
| `--max_layers` | None | Quantize only the first N encoder layers (smoke testing). |
| `--skip_eval` | False | Skip WER evaluation. |
| `--save` / `--output` | None | Save the quantized state dict / results JSON. |

## File structure

```
quantize_encoder_spqr.py  - Main entry point (encoder-only SpQR)
spqr_engine.py            - VERBATIM upstream SpQR: SPQRUtil (GPTQ + outliers)
quant_groups.py           - VERBATIM upstream SpQR: Quantizer
weight_permutation.py     - VERBATIM upstream SpQR: get_permutation_order
modelutils.py             - Moonshine + LiteASR .pth loading, layer utilities
datautils.py              - Calibration / eval data loading
eval_utils.py             - Correct WER evaluation (reference recipe)
requirements.txt          - Python dependencies
```

## Requirements

```bash
pip install -r requirements.txt
```

- Python >= 3.9, PyTorch >= 2.0, transformers >= 4.49.0 (for Moonshine support)
- CUDA recommended (and required to reproduce the fp16 baseline numbers).

## Acknowledgements

- [SpQR: Sparse-Quantized Representation](https://github.com/Vahe1994/SpQR) — the quantization engine used verbatim.
- [LiteASR](https://arxiv.org/abs/2502.20583) — low-rank encoder compression.
- [Moonshine](https://huggingface.co/usefulsensors/moonshine-base) — the ASR model.
