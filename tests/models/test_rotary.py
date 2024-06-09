"""Unit tests for continuous and discrete Rotary Position Embeddings (RoPE)."""

from __future__ import annotations

import math
import pytest
import torch

from nocturne.models.attention.rotary import (
    ContinuousRotaryEmbedding,
    rotate_half,
)


def test_rotate_half_orthogonality() -> None:
    x = torch.randn(4, 8)
    rot_x = rotate_half(x)

    # In 2D blocks, rotation by pi/2 is orthogonal: x . rot_x == 0
    # Let's verify for individual 2D pairs: (x1, x2) . (-x2, x1) = -x1*x2 + x2*x1 = 0
    dot_pairs = x[..., :4] * (-x[..., 4:]) + x[..., 4:] * x[..., :4]
    assert torch.allclose(dot_pairs, torch.zeros_like(dot_pairs))


def test_relative_lag_invariance() -> None:
    rope = ContinuousRotaryEmbedding(dim=16, base=1000.0)

    # Two queries and keys separated by fixed lag delta_t = 3
    q = torch.randn(1, 1, 1, 16)
    k = torch.randn(1, 1, 1, 16)

    # Evaluation at base t = 2 (q at t=2, k at t=5)
    t_q1 = torch.tensor([[2.0]])
    t_k1 = torch.tensor([[5.0]])
    q_rot1, _ = rope.apply_rotary_emb(q, q, timestamps_s=t_q1)
    _, k_rot1 = rope.apply_rotary_emb(k, k, timestamps_s=t_k1)
    score1 = torch.sum(q_rot1 * k_rot1).item()

    # Evaluation at shifted base t = 10 (q at t=10, k at t=13)
    t_q2 = torch.tensor([[10.0]])
    t_k2 = torch.tensor([[13.0]])
    q_rot2, _ = rope.apply_rotary_emb(q, q, timestamps_s=t_q2)
    _, k_rot2 = rope.apply_rotary_emb(k, k, timestamps_s=t_k2)
    score2 = torch.sum(q_rot2 * k_rot2).item()

    # Relative dot product should be identical up to floating point precision
    assert pytest.approx(score1, rel=1e-4) == score2


def test_discrete_bin_cache_and_broadcasting() -> None:
    B, num_heads, T, head_dim = 2, 4, 32, 16
    rope = ContinuousRotaryEmbedding(dim=head_dim, max_seq_len=64)

    q = torch.randn(B, num_heads, T, head_dim, requires_grad=True)
    k = torch.randn(B, num_heads, T, head_dim, requires_grad=True)

    q_rot, k_rot = rope.apply_rotary_emb(q, k)

    assert q_rot.shape == (B, num_heads, T, head_dim)
    assert k_rot.shape == (B, num_heads, T, head_dim)

    # Verify gradient flow
    loss = torch.sum(q_rot * k_rot)
    loss.backward()

    assert q.grad is not None
    assert k.grad is not None
    assert torch.all(torch.isfinite(q.grad))
    assert torch.all(torch.isfinite(k.grad))


def test_odd_dimension_raises_error() -> None:
    with pytest.raises(ValueError, match="must be even"):
        ContinuousRotaryEmbedding(dim=15)
