import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from torch.nn.functional import scaled_dot_product_attention

    SDPA_AVAILABLE = True
except (ImportError, RuntimeError, OSError):
    scaled_dot_product_attention = None
    SDPA_AVAILABLE = False

try:
    from compile import CUDAGraphed
except ImportError:
    CUDAGraphed = None


@dataclass
class MoonshineModelDimensions:
    hidden_size: int
    intermediate_size: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_head: int
    n_text_layer: int
    n_text_ctx: int
    head_dim: int
    partial_rotary_factor: float
    rope_theta: float
    pad_head_dim_to_multiple_of: int


class LayerNorm(nn.Module):
    """LayerNorm without bias, matching Moonshine architecture."""

    def __init__(self, normalized_shape: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, self.normalized_shape, self.weight.to(x.dtype), None, 1e-5)


class Linear(nn.Module):
    """Linear layer with optional bias, cast to input dtype."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class LinearLowRank(nn.Module):
    """Low-rank linear layer: x -> (x @ weight1^T) @ weight2^T + bias."""

    def __init__(self, in_features: int, out_features: int, low_rank_features: int,
                 device: str = "cpu", dtype: torch.dtype = torch.float32):
        super().__init__()
        self.weight1 = nn.Parameter(torch.empty(low_rank_features, in_features, dtype=dtype, device=device))
        self.weight2 = nn.Parameter(torch.empty(out_features, low_rank_features, dtype=dtype, device=device))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype, device=device))

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            F.linear(x, self.weight1, None), self.weight2, self.bias
        )


class Conv1d(nn.Conv1d):
    """Conv1d that casts weight/bias to input dtype."""

    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


class MoonshineRotaryEmbedding(nn.Module):
    """Rotary Position Embedding for Moonshine.

    Only the first rotary_dim dimensions get rotary embedding.
    """

    def __init__(self, head_dim: int, partial_rotary_factor: float, rope_theta: float = 10000.0,
                 device: str = "cpu"):
        super().__init__()
        self.head_dim = head_dim
        self.rotary_dim = int(head_dim * partial_rotary_factor)  # 32 for default
        self.rope_theta = rope_theta

        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32, device=device) / self.rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
        """Return (cos, sin) tensors of shape (seq_len, rotary_dim)."""
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))  # (seq_len, rotary_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, rotary_dim)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: Tensor) -> Tensor:
    """Rotate the last dimension: [-x2, x1]."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor, rotary_dim: int
) -> Tuple[Tensor, Tensor]:
    """Apply rotary embedding to first rotary_dim dimensions of q and k.

    q, k: (batch, n_heads, seq_len, head_dim)
    cos, sin: (seq_len, rotary_dim)
    """
    # Extract the rotary and pass-through portions
    q_rot = q[..., :rotary_dim]
    q_pass = q[..., rotary_dim:]
    k_rot = k[..., :rotary_dim]
    k_pass = k[..., rotary_dim:]

    # Expand cos/sin for broadcasting: (1, 1, seq_len, rotary_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Apply rotary embedding
    q_rot = q_rot * cos + _rotate_half(q_rot) * sin
    k_rot = k_rot * cos + _rotate_half(k_rot) * sin

    # Concatenate back
    q = torch.cat([q_rot, q_pass], dim=-1)
    k = torch.cat([k_rot, k_pass], dim=-1)

    return q, k


class MultiHeadAttention(nn.Module):
    """Multi-head attention with RoPE support and head_dim padding."""

    def __init__(
        self,
        hidden_size: int,
        n_head: int,
        head_dim: int,
        padded_head_dim: int,
        rotary_emb: Optional[MoonshineRotaryEmbedding] = None,
        use_rope: bool = True,
        has_bias: bool = False,
        low_rank_config: Optional[Dict[str, int]] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        self.padded_head_dim = padded_head_dim
        self.hidden_size = hidden_size
        self.use_rope = use_rope
        self.rotary_emb = rotary_emb

        if low_rank_config is not None:
            self.query = LinearLowRank(hidden_size, hidden_size, low_rank_config["q_proj"], device=device, dtype=dtype) \
                if "q_proj" in low_rank_config else Linear(hidden_size, hidden_size, bias=has_bias)
            self.key = LinearLowRank(hidden_size, hidden_size, low_rank_config["k_proj"], device=device, dtype=dtype) \
                if "k_proj" in low_rank_config else Linear(hidden_size, hidden_size, bias=has_bias)
            self.value = LinearLowRank(hidden_size, hidden_size, low_rank_config["v_proj"], device=device, dtype=dtype) \
                if "v_proj" in low_rank_config else Linear(hidden_size, hidden_size, bias=has_bias)
            self.out = LinearLowRank(hidden_size, hidden_size, low_rank_config["o_proj"], device=device, dtype=dtype) \
                if "o_proj" in low_rank_config else Linear(hidden_size, hidden_size, bias=has_bias)
        else:
            self.query = Linear(hidden_size, hidden_size, bias=has_bias)
            self.key = Linear(hidden_size, hidden_size, bias=has_bias)
            self.value = Linear(hidden_size, hidden_size, bias=has_bias)
            self.out = Linear(hidden_size, hidden_size, bias=has_bias)

    def _apply_rope_to_qk(
        self, q: Tensor, k: Tensor, position_ids: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Apply RoPE to q and k tensors. q, k: (batch, n_heads, seq_len, head_dim)."""
        rotary_dim = self.rotary_emb.rotary_dim
        max_pos = int(position_ids.max().item()) + 1
        cos, sin = self.rotary_emb(max_pos, q.device, q.dtype)

        # Select cos/sin by position_ids: (seq_len, rotary_dim)
        pos_cos = cos[position_ids]
        pos_sin = sin[position_ids]

        q, k = apply_rotary_pos_emb(q, k, pos_cos, pos_sin, rotary_dim)
        return q, k

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
        offset: Optional[int] = None,
        position_ids: Optional[Tensor] = None,
        is_prefilling: bool = False,
    ) -> Tensor:
        q = self.query(x)
        batch, seq_len, _ = q.shape

        if kv_cache is None:
            # Encoder self-attention: no KV cache
            k = self.key(x)
            v = self.value(x)

            # Reshape to (batch, n_heads, seq_len, head_dim)
            q = q.view(batch, seq_len, self.n_head, self.head_dim).permute(0, 2, 1, 3)
            k = k.view(batch, seq_len, self.n_head, self.head_dim).permute(0, 2, 1, 3)
            v = v.view(batch, seq_len, self.n_head, self.head_dim).permute(0, 2, 1, 3)

            # Apply RoPE
            if self.use_rope and self.rotary_emb is not None and position_ids is not None:
                q, k = self._apply_rope_to_qk(q, k, position_ids)

            wv = self._sdpa(q, k, v, mask=None)

        elif xa is None:
            # Decoder self-attention with KV cache
            # Apply RoPE BEFORE caching so cached keys have positional info
            k = self.key(x)
            v = self.value(x)

            # Reshape to multi-head
            q = q.view(batch, seq_len, self.n_head, self.head_dim).permute(0, 2, 1, 3)
            k = k.view(batch, seq_len, self.n_head, self.head_dim).permute(0, 2, 1, 3)
            v = v.view(batch, seq_len, self.n_head, self.head_dim).permute(0, 2, 1, 3)

            # Apply RoPE to current q, k
            if self.use_rope and self.rotary_emb is not None and position_ids is not None:
                q, k = self._apply_rope_to_qk(q, k, position_ids)

            # Flatten back to (batch, seq_len, hidden_size) for cache storage
            k_flat = k.permute(0, 2, 1, 3).contiguous().reshape(batch, seq_len, self.hidden_size)
            v_flat = v.permute(0, 2, 1, 3).contiguous().reshape(batch, seq_len, self.hidden_size)

            # Store in cache
            off = offset if isinstance(offset, int) else (offset.item() if isinstance(offset, Tensor) else int(offset))
            if is_prefilling:
                kv_cache["self-key"][:, off: off + seq_len].copy_(k_flat)
                kv_cache["self-value"][:, off: off + seq_len].copy_(v_flat)
            else:
                kv_cache["self-key"][:, off: off + 1].copy_(k_flat)
                kv_cache["self-value"][:, off: off + 1].copy_(v_flat)

            # Read from cache and reshape to multi-head for attention
            kv_len = off + seq_len
            cached_k = kv_cache["self-key"][:, :kv_len]
            cached_v = kv_cache["self-value"][:, :kv_len]
            cached_k = cached_k.view(batch, kv_len, self.n_head, self.head_dim).permute(0, 2, 1, 3)
            cached_v = cached_v.view(batch, kv_len, self.n_head, self.head_dim).permute(0, 2, 1, 3)

            wv = self._sdpa(q, cached_k, cached_v, mask=mask)

        else:
            # Decoder cross-attention (no RoPE)
            q = q.view(batch, seq_len, self.n_head, self.head_dim).permute(0, 2, 1, 3)

            if is_prefilling:
                k = self.key(xa)
                v = self.value(xa)
                kv_cache["cross-key"][:, :k.shape[1]].copy_(k)
                kv_cache["cross-value"][:, :v.shape[1]].copy_(v)

            # Get cross-attention KV from cache
            cached_k = kv_cache["cross-key"]
            cached_v = kv_cache["cross-value"]
            kv_len = cached_k.shape[1]
            cached_k = cached_k.view(batch, kv_len, self.n_head, self.head_dim).permute(0, 2, 1, 3)
            cached_v = cached_v.view(batch, kv_len, self.n_head, self.head_dim).permute(0, 2, 1, 3)

            wv = self._sdpa(q, cached_k, cached_v, mask=None)

        out = self.out(wv)
        return out

    def _sdpa(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Scaled dot-product attention with head_dim padding.

        q: (batch, n_heads, q_len, head_dim)
        k: (batch, n_heads, kv_len, head_dim)
        v: (batch, n_heads, kv_len, head_dim)
        Returns: (batch, q_len, hidden_size)
        """
        batch = q.shape[0]
        q_len = q.shape[2]

        # Pad head_dim to padded_head_dim for attention computation
        if self.padded_head_dim > self.head_dim:
            pad_size = self.padded_head_dim - self.head_dim
            q = F.pad(q, (0, pad_size))
            k = F.pad(k, (0, pad_size))
            v = F.pad(v, (0, pad_size))

        # Scaled dot-product attention
        if SDPA_AVAILABLE:
            a = scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            scale = self.padded_head_dim ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if mask is not None:
                scores = scores + mask
            scores = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            a = torch.matmul(scores, v)

        # Slice back to head_dim
        if self.padded_head_dim > self.head_dim:
            a = a[..., :self.head_dim]

        # Reshape back: (batch, n_heads, q_len, head_dim) -> (batch, q_len, hidden_size)
        out = a.permute(0, 2, 1, 3).contiguous().reshape(batch, q_len, self.hidden_size)
        return out


class EncoderBlock(nn.Module):
    """Moonshine encoder block: pre-norm with self-attention + MLP (GELU)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        n_head: int,
        head_dim: int,
        padded_head_dim: int,
        rotary_emb: MoonshineRotaryEmbedding,
        low_rank_config: Optional[Dict[str, int]] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.input_layernorm = LayerNorm(hidden_size)
        self.self_attn = MultiHeadAttention(
            hidden_size, n_head, head_dim, padded_head_dim,
            rotary_emb=rotary_emb, use_rope=True, has_bias=False,
            low_rank_config=low_rank_config, device=device, dtype=dtype,
        )
        self.post_attention_layernorm = LayerNorm(hidden_size)

        # MLP: fc1 -> GELU -> fc2
        if low_rank_config is not None and "fc1" in low_rank_config:
            self.fc1 = LinearLowRank(hidden_size, intermediate_size, low_rank_config["fc1"], device=device, dtype=dtype)
        else:
            self.fc1 = Linear(hidden_size, intermediate_size, bias=True)

        if low_rank_config is not None and "fc2" in low_rank_config:
            self.fc2 = LinearLowRank(intermediate_size, hidden_size, low_rank_config["fc2"], device=device, dtype=dtype)
        else:
            self.fc2 = Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, x: Tensor, position_ids: Optional[Tensor] = None) -> Tensor:
        # Self-attention with pre-norm
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, position_ids=position_ids)
        x = residual + x

        # MLP with pre-norm
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = residual + x

        return x


class DecoderBlock(nn.Module):
    """Moonshine decoder block: self-attn + cross-attn + SwiGLU MLP."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        n_head: int,
        head_dim: int,
        padded_head_dim: int,
        rotary_emb: MoonshineRotaryEmbedding,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        # Self-attention (with RoPE)
        self.input_layernorm = LayerNorm(hidden_size)
        self.self_attn = MultiHeadAttention(
            hidden_size, n_head, head_dim, padded_head_dim,
            rotary_emb=rotary_emb, use_rope=True, has_bias=False,
            device=device, dtype=dtype,
        )

        # Cross-attention (no RoPE)
        self.post_attention_layernorm = LayerNorm(hidden_size)
        self.cross_attn = MultiHeadAttention(
            hidden_size, n_head, head_dim, padded_head_dim,
            rotary_emb=None, use_rope=False, has_bias=False,
            device=device, dtype=dtype,
        )

        # MLP with SwiGLU: fc1(hidden->2*intermediate) -> chunk -> SiLU(gate)*x -> fc2(intermediate->hidden)
        self.final_layernorm = LayerNorm(hidden_size)
        # fc1 outputs 2*intermediate_size for SwiGLU gate+hidden split
        self.fc1 = Linear(hidden_size, intermediate_size * 2, bias=True)
        self.fc2 = Linear(intermediate_size, hidden_size, bias=True)

    def forward(
        self,
        x: Tensor,
        xa: Tensor,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
        offset: Optional[int] = None,
        position_ids: Optional[Tensor] = None,
        is_prefilling: bool = False,
    ) -> Tensor:
        # Self-attention with pre-norm
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(
            x, xa=None, mask=mask, kv_cache=kv_cache,
            offset=offset, position_ids=position_ids,
            is_prefilling=is_prefilling,
        )
        x = residual + x

        # Cross-attention with pre-norm
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.cross_attn(
            x, xa=xa, kv_cache=kv_cache, is_prefilling=is_prefilling,
        )
        x = residual + x

        # SwiGLU MLP with pre-norm
        residual = x
        x = self.final_layernorm(x)
        gate_and_hidden = self.fc1(x)
        gate, hidden = gate_and_hidden.chunk(2, dim=-1)
        x = F.silu(gate) * hidden
        x = self.fc2(x)
        x = residual + x

        return x


class AudioEncoder(nn.Module):
    """Moonshine audio encoder: Conv frontend + Transformer with RoPE."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        n_head: int,
        n_layer: int,
        head_dim: int,
        padded_head_dim: int,
        partial_rotary_factor: float,
        rope_theta: float,
        low_rank_config: List[Dict[str, int]],
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        # Conv1d frontend: raw waveform -> features
        self.conv1 = Conv1d(1, hidden_size, kernel_size=127, stride=64, bias=False)
        self.group_norm = nn.GroupNorm(1, hidden_size)
        self.conv2 = Conv1d(hidden_size, hidden_size * 2, kernel_size=7, stride=3)
        self.conv3 = Conv1d(hidden_size * 2, hidden_size, kernel_size=3, stride=2)

        # Rotary embedding shared across layers
        self.rotary_emb = MoonshineRotaryEmbedding(
            head_dim, partial_rotary_factor, rope_theta, device=device
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            EncoderBlock(
                hidden_size, intermediate_size, n_head, head_dim, padded_head_dim,
                rotary_emb=self.rotary_emb, low_rank_config=low_rank_config[i],
                device=device, dtype=dtype,
            )
            for i in range(n_layer)
        ])
        self.ln_post = LayerNorm(hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        x : torch.Tensor, shape = (batch_size, audio_len)
            Raw audio waveform.
        """
        # Add channel dimension: (batch, audio_len) -> (batch, 1, audio_len)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Conv frontend
        x = torch.tanh(self.conv1(x))
        x = self.group_norm(x)
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))

        # (batch, hidden_size, seq_len) -> (batch, seq_len, hidden_size)
        x = x.permute(0, 2, 1)

        # Position IDs for RoPE
        seq_len = x.shape[1]
        position_ids = torch.arange(seq_len, device=x.device, dtype=torch.long)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, position_ids=position_ids)

        x = self.ln_post(x)
        return x


class TextDecoder(nn.Module):
    """Moonshine text decoder: embed + Transformer with self-attn (RoPE) + cross-attn + SwiGLU."""

    def __init__(
        self,
        n_vocab: int,
        n_text_ctx: int,
        hidden_size: int,
        intermediate_size: int,
        n_head: int,
        n_layer: int,
        head_dim: int,
        padded_head_dim: int,
        partial_rotary_factor: float,
        rope_theta: float,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, hidden_size)

        # Rotary embedding for decoder self-attention
        self.rotary_emb = MoonshineRotaryEmbedding(
            head_dim, partial_rotary_factor, rope_theta, device=device
        )

        self.blocks = nn.ModuleList([
            DecoderBlock(
                hidden_size, intermediate_size, n_head, head_dim, padded_head_dim,
                rotary_emb=self.rotary_emb, device=device, dtype=dtype,
            )
            for _ in range(n_layer)
        ])
        self.ln = LayerNorm(hidden_size)

        # Causal mask
        mask = torch.empty(n_text_ctx, n_text_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(
        self,
        x: Tensor,
        xa: Tensor,
        offset: int,
        kv_cache: Optional[List[dict]] = None,
        is_prefilling: bool = False,
    ) -> Tensor:
        """
        x : torch.LongTensor, shape = (batch_size, seq_len)
            Token IDs.
        xa : torch.Tensor, shape = (batch_size, encoder_seq_len, hidden_size)
            Encoder output.
        offset : int
            Position offset for KV cache.
        """
        seq_len = x.shape[1]
        x = self.token_embedding(x)
        x = x.to(xa.dtype)

        # Position IDs for this segment
        position_ids = torch.arange(offset, offset + seq_len, device=x.device, dtype=torch.long)

        # Build causal mask for this segment
        if is_prefilling:
            sliced_mask = self.mask[offset: offset + seq_len, :offset + seq_len]
        else:
            # Single token: no mask needed (all previous positions visible)
            sliced_mask = None

        for i, block in enumerate(self.blocks):
            x = block(
                x, xa,
                mask=sliced_mask,
                kv_cache=kv_cache[i] if kv_cache is not None else None,
                offset=offset,
                position_ids=position_ids,
                is_prefilling=is_prefilling,
            )

        x = self.ln(x)

        # Project to vocab
        logits = (x @ self.token_embedding.weight.to(x.dtype).T).float()
        return logits


class Moonshine(nn.Module):
    """
    Custom optimized Moonshine inference model with static KV cache and CUDA Graph support.
    Equivalent of src/model.py Whisper class but for the Moonshine architecture.
    """

    def __init__(
        self,
        dims: MoonshineModelDimensions,
        low_rank_config: Optional[List[Dict[str, int]]] = None,
        bs: int = 1,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        eos_token_id: int = 2,
        max_encoder_len: int = 512,
    ):
        super().__init__()
        self.dims = dims
        self.bs = bs
        self.device = device
        self.dtype = dtype
        self.eos_token_id = eos_token_id

        # If low_rank_config is None, use empty dicts (uncompressed model)
        if low_rank_config is None:
            low_rank_config = [{} for _ in range(dims.n_audio_layer)]

        padded_head_dim = (
            math.ceil(dims.head_dim / dims.pad_head_dim_to_multiple_of)
            * dims.pad_head_dim_to_multiple_of
        )

        self.encoder = AudioEncoder(
            hidden_size=dims.hidden_size,
            intermediate_size=dims.intermediate_size,
            n_head=dims.n_audio_head,
            n_layer=dims.n_audio_layer,
            head_dim=dims.head_dim,
            padded_head_dim=padded_head_dim,
            partial_rotary_factor=dims.partial_rotary_factor,
            rope_theta=dims.rope_theta,
            low_rank_config=low_rank_config,
            device=device,
            dtype=dtype,
        )

        self.decoder = TextDecoder(
            n_vocab=dims.n_vocab,
            n_text_ctx=dims.n_text_ctx,
            hidden_size=dims.hidden_size,
            intermediate_size=dims.intermediate_size,
            n_head=dims.n_text_head,
            n_layer=dims.n_text_layer,
            head_dim=dims.head_dim,
            padded_head_dim=padded_head_dim,
            partial_rotary_factor=dims.partial_rotary_factor,
            rope_theta=dims.rope_theta,
            device=device,
            dtype=dtype,
        )

        # Allocate static KV cache
        self.max_encoder_len = max_encoder_len
        self.kv_cache = [
            {
                "self-key": torch.zeros(bs, dims.n_text_ctx, dims.hidden_size, device=device, dtype=dtype),
                "self-value": torch.zeros(bs, dims.n_text_ctx, dims.hidden_size, device=device, dtype=dtype),
                "cross-key": torch.zeros(bs, self.max_encoder_len, dims.hidden_size, device=device, dtype=dtype),
                "cross-value": torch.zeros(bs, self.max_encoder_len, dims.hidden_size, device=device, dtype=dtype),
            }
            for _ in range(dims.n_text_layer)
        ]
        self.offset = 0

    def reinit_kv_cache(self):
        """Reset KV cache for a new sequence."""
        for cache in self.kv_cache:
            cache["self-key"].zero_()
            cache["self-value"].zero_()
            cache["cross-key"].zero_()
            cache["cross-value"].zero_()
        self.offset = 0

    def init_cuda_graph(self):
        """Initialize CUDA Graphs for encoder and decoder. Skip on CPU."""
        if self.device == "cpu" or CUDAGraphed is None:
            print("Skipping CUDA Graph initialization (CPU mode or CUDAGraphed not available).")
            return

        with torch.no_grad():
            self._init_encoder()
            self._init_decoder_generation()
            self.reinit_kv_cache()
            print("CUDA Graph initialization complete.")

    def _init_encoder(self):
        """Warm up and graph the encoder."""
        # Warmup with a dummy waveform
        dummy_waveform = torch.randn(self.bs, 16000, device=self.device, dtype=self.dtype)
        self.encoder(dummy_waveform)
        self.encoder.forward = CUDAGraphed(self.encoder.forward)
        for _ in range(5):
            self.encoder.forward(dummy_waveform)

    def _init_decoder_generation(self):
        """Warm up and graph the decoder for single-token generation."""
        dummy_tokens = torch.randint(0, self.dims.n_vocab, (self.bs, 1), device=self.device, dtype=torch.long)
        dummy_enc_out = torch.randn(self.bs, 100, self.dims.hidden_size, device=self.device, dtype=self.dtype)

        self.decoder.forward_generation = CUDAGraphed(self.decoder.forward)
        for _ in range(5):
            self.decoder.forward_generation(
                dummy_tokens, dummy_enc_out, 0, self.kv_cache, False
            )

    def forward(
        self, waveform: Tensor, decoder_input_ids: Tensor
    ) -> List[int]:
        """
        Run inference: encode audio and decode tokens autoregressively.

        Args:
            waveform: (batch, audio_len) raw audio waveform
            decoder_input_ids: (batch, seq_len) initial decoder token IDs (e.g., [bos_token_id])

        Returns:
            List of generated token IDs.
        """
        self.reinit_kv_cache()
        ret_tokens = []

        # Encode audio
        encoded = self.encoder(waveform)

        # Check that encoder output fits in the cross-attention cache
        encoder_seq_len = encoded.shape[1]
        if encoder_seq_len > self.max_encoder_len:
            raise ValueError(
                f"Encoder output length ({encoder_seq_len}) exceeds max_encoder_len "
                f"({self.max_encoder_len}). Increase max_encoder_len or use shorter audio."
            )

        # Prefill decoder with initial tokens
        logits = self.decoder(decoder_input_ids, encoded, 0, self.kv_cache, True)
        self.offset = logits.shape[1]
        next_token = torch.argmax(logits, dim=-1)[:, -1:]

        # Autoregressive generation
        use_cuda_graph = (self.device != "cpu" and hasattr(self.decoder, "forward_generation"))

        max_gen_len = self.dims.n_text_ctx - self.offset
        for _ in range(max_gen_len):
            ret_tokens.append(next_token[0, 0].item())
            # EOS token
            if next_token[0, -1].item() == self.eos_token_id:
                break

            if use_cuda_graph:
                logits = self.decoder.forward_generation(
                    next_token, encoded, self.offset, self.kv_cache, False
                )
            else:
                logits = self.decoder(
                    next_token, encoded, self.offset, self.kv_cache, False
                )
            next_token = torch.argmax(logits, dim=-1)[:, -1:]
            self.offset += 1

        return ret_tokens
