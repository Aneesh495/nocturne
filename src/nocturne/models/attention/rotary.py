"""Continuous-time Rotary Position Embeddings (RoPE) for neural sequence attention.

Applies relative rotational position transformations to queries and keys along
continuous temporal intervals or discrete bin indices, preserving relative lag invariance.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the hidden dimension coordinates for 2D symplectic rotation.

    Transforms [x1, x2, x3, x4] -> [-x2, x1, -x4, x3].
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class ContinuousRotaryEmbedding(nn.Module):
    """Rotary Position Embedding supporting both continuous timestamps and discrete bins.

    For queries and keys of dimension d, rotates pairs of coordinates by angle theta_i * t:
        <R(q, t), R(k, s)> = g(q, k, t - s)
    ensuring relative temporal lag invariance across neural spike trains.
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        max_seq_len: int = 512,
    ):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Rotary embedding dimension must be even, got {dim}")
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len

        # Inverse frequencies: theta_i = base^(-2i / dim)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute static grid for fast integer indexing
        self._build_static_cache(max_seq_len)

    def _build_static_cache(self, max_seq_len: int) -> None:
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)  # [T, dim // 2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [T, dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        timestamps_s: Optional[torch.Tensor] = None,
        seq_dim: int = -2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cosine and sine rotational frequency tensors for input.

        Args:
            x: Input tensor containing sequence dimension [..., T, ..., d].
            timestamps_s: Optional continuous timestamps [B, T] or [T] in seconds.
            seq_dim: Dimension index corresponding to time sequence length T.

        Returns:
            Tuple of (cos_emb, sin_emb) broadcastable with x.
        """
        T = x.shape[seq_dim]

        if timestamps_s is not None:
            # Continuous timestamp formulation
            # timestamps_s: [B, T] or [T]
            t = timestamps_s.to(dtype=self.inv_freq.dtype, device=x.device)
            if t.ndim == 1:
                t = t.unsqueeze(0)  # [1, T]
            freqs = t.unsqueeze(-1) * self.inv_freq.unsqueeze(0).unsqueeze(0)  # [B, T, dim // 2]
            emb = torch.cat((freqs, freqs), dim=-1)  # [B, T, dim]
            return emb.cos(), emb.sin()

        # Discrete bin cache fast path
        if T > self.cos_cached.shape[0]:
            self._build_static_cache(max(T, self.max_seq_len * 2))

        cos = self.cos_cached[:T].to(device=x.device, dtype=x.dtype)
        sin = self.sin_cached[:T].to(device=x.device, dtype=x.dtype)
        return cos, sin

    def apply_rotary_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        timestamps_s: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embedding to query and key tensors in-place or out-of-place.

        Args:
            q: Query tensor of shape [B, num_heads, T, head_dim].
            k: Key tensor of shape [B, num_heads, T, head_dim].
            timestamps_s: Optional continuous timestamps in seconds.

        Returns:
            Tuple of rotated (q_rot, k_rot).
        """
        cos, sin = self.forward(q, timestamps_s=timestamps_s, seq_dim=-2)

        # Reshape for broadcasting with [B, num_heads, T, head_dim]
        if cos.ndim == 2:  # [T, head_dim]
            cos = cos.unsqueeze(0).unsqueeze(1)  # [1, 1, T, head_dim]
            sin = sin.unsqueeze(0).unsqueeze(1)
        elif cos.ndim == 3:  # [B, T, head_dim]
            cos = cos.unsqueeze(1)  # [B, 1, T, head_dim]
            sin = sin.unsqueeze(1)

        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)
        return q_rot, k_rot
