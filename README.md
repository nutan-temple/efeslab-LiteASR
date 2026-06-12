# SpQR W8A8 Quantization for Moonshine

A standalone W8A8 (8-bit weights, 8-bit activations) quantization pipeline
for LiteASR-compressed Moonshine ASR models, using GPTQ-style Hessian-aware
calibration.

## What This Does

This pipeline takes a LiteASR `.pth` file (a Moonshine model where encoder
layers have been compressed using low-rank factorization) and applies INT8
quantization to all linear layers using GPTQ calibration for optimal rounding.

### What W8A8 Means

- **W8 (Weight 8-bit):** All weight matrices are quantized to INT8 using
  per-channel symmetric quantization. Each output channel (row) gets its own
  scale factor: `scale = max(|row|) / 127`

- **A8 (Activation 8-bit):** At inference time, activations are dynamically
  quantized to INT8 using per-tensor symmetric quantization. This happens
  on-the-fly based on the observed activation range.

### Why GPTQ Calibration

Naive INT8 quantization (round to nearest) ignores the input distribution.
GPTQ uses a Hessian-based approach:

1. Collect `H = X^T X / n` from calibration data (LibriSpeech dev-clean)
2. Process weight columns sequentially using the inverse Hessian
3. Propagate quantization error from each column to subsequent columns
4. This compensates for rounding errors in a data-aware manner

The result is significantly better accuracy than naive rounding, especially
for sensitive layers.

## Architecture

The Moonshine model (`usefulsensors/moonshine-base`) has:
- **Encoder:** Conv frontend + 8 transformer layers with RoPE
  - Conv1(1, 416, k=127, s=64) -> tanh -> GroupNorm -> Conv2(416, 832, k=7, s=3) -> GELU -> Conv3(832, 416, k=3, s=2) -> GELU
  - 8 layers: self-attention (q/k/v/o_proj) + MLP (fc1, fc2)
- **Decoder:** 8 transformer layers with self-attention + cross-attention + SwiGLU MLP
  - hidden_size=416, intermediate_size=1664, 8 heads, vocab=32768

### LiteASR Compression

The `.pth` file contains encoder layers where some `nn.Linear` modules have
been replaced with `LinearLowRank`:

```python
class LinearLowRank(nn.Module):
    def __init__(self, weight1, weight2, bias):
        super().__init__()
        self.weight1 = nn.Parameter(weight1)  # (in_features, rank)
        self.weight2 = nn.Parameter(weight2)  # (rank, out_features)
        self.bias = nn.Parameter(bias)

    def forward(self, x):
        return (x @ self.weight1) @ self.weight2 + self.bias
```

The W8A8 quantization handles both `nn.Linear` and `LinearLowRank` layers.

## File Structure

```
quantize_w8a8.py   - Main entry point
modelutils.py      - Model loading (HF + LiteASR .pth) and layer utilities
quant_engine.py    - W8A8 quantization engine (GPTQ + INT8 primitives)
datautils.py       - Calibration data (LibriSpeech dev) and eval data loading
eval_utils.py      - WER evaluation on LibriSpeech test-clean/test-other
requirements.txt   - Python dependencies
README.md          - This file
```

## Usage

### Basic W8A8 Quantization

```bash
python quantize_w8a8.py \
    --pth_path /path/to/lite-moonshine-moonshine-base_0.98:0.99.pth \
    --nsamples 128 \
    --max_eval_samples 200
```

### Full Evaluation

```bash
python quantize_w8a8.py \
    --pth_path /path/to/model.pth \
    --nsamples 128 \
    --save quantized_w8a8.pth \
    --output results.json
```

### Encoder-Only Quantization

```bash
python quantize_w8a8.py \
    --pth_path /path/to/model.pth \
    --part encoder \
    --nsamples 64
```

### Quick Test (No Network Required)

```bash
python quantize_w8a8.py \
    --pth_path /path/to/model.pth \
    --use_synthetic \
    --nsamples 16 \
    --skip_eval
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--pth_path` | (required) | Path to LiteASR .pth checkpoint |
| `--model` | `usefulsensors/moonshine-base` | HuggingFace model name |
| `--part` | `both` | Which parts to quantize: encoder, decoder, or both |
| `--blocksize` | `128` | GPTQ block size for column processing |
| `--percdamp` | `0.01` | GPTQ Hessian damping factor |
| `--nsamples` | `128` | Number of calibration samples |
| `--audio_len` | `160000` | Audio length (samples at 16kHz, 160000 = 10s) |
| `--use_synthetic` | False | Use random noise instead of LibriSpeech |
| `--skip_eval` | False | Skip WER evaluation |
| `--max_eval_samples` | None | Max eval samples per split (None = all) |
| `--save` | None | Path to save quantized state_dict |
| `--output` | `w8a8_results.json` | Path to save results JSON |

## How Each File Works

### `quantize_w8a8.py` (Main Entry Point)

Orchestrates the full pipeline:
1. Parses command-line arguments
2. Loads the Moonshine model and applies LiteASR .pth weights
3. Optionally evaluates WER before quantization
4. Loads calibration data (LibriSpeech dev or synthetic)
5. Runs W8A8 quantization on encoder and/or decoder
6. Optionally evaluates WER after quantization
7. Prints a comparison table and saves results

### `modelutils.py` (Model Utilities)

Handles all model-related operations:
- `load_moonshine_model()`: Downloads Moonshine from HuggingFace
- `load_liteasr_pth()`: Replaces encoder Linear layers with LinearLowRank
- `get_encoder_layers()` / `get_decoder_layers()`: Access layer lists
- `find_sublayers()`: Find all Linear/LinearLowRank in a layer
- `get_encoder_sequential_groups()`: Groups of sublayers that share inputs
- `get_decoder_sequential_groups()`: Same for decoder (includes cross-attention)

### `quant_engine.py` (Quantization Engine)

Core quantization logic:
- `quantize_weight_int8_perchannel()`: Per-channel symmetric INT8 quantization
- `dequantize_weight_int8()`: Reverse (for simulation/evaluation)
- `quantize_activation_int8_dynamic()`: Dynamic per-tensor INT8 for activations
- `GPTQQuantizer`: Hessian-aware quantization class
  - `add_batch()`: Accumulates H = X^T X / n from calibration inputs
  - `quantize()`: Runs GPTQ algorithm targeting INT8
- `quantize_encoder_w8a8()`: Quantizes all encoder layers
- `quantize_decoder_w8a8()`: Quantizes all decoder layers
- `W8A8Linear` / `W8A8LinearLowRank`: Quantized layer wrappers

### `datautils.py` (Data Utilities)

Data loading for calibration and evaluation:
- `get_librispeech_calibration()`: Loads LibriSpeech dev-clean for GPTQ calibration
- `get_librispeech_eval()`: Loads LibriSpeech test splits for WER evaluation
- `get_synthetic_calibration()`: Random noise for testing without network

### `eval_utils.py` (Evaluation)

WER evaluation using the HuggingFace `evaluate` library:
- `evaluate_wer()`: Runs model.generate() on a dataset, computes WER
- `evaluate_model()`: Full evaluation on test-clean and test-other
- Uses `max_length = max(int(seq_lens * 6.5/16000), 10)` for generation

## Expected Results

With 128 calibration samples on a typical LiteASR model:
- Compression: ~4x (FP32 to INT8)
- WER degradation: typically < 1% absolute on test-clean
- Quantization time: ~2-5 minutes on A100 (depends on model size)

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- transformers >= 4.49.0 (for Moonshine support)
- CUDA recommended (works on CPU but slower)

Install dependencies:
```bash
pip install -r requirements.txt
```
