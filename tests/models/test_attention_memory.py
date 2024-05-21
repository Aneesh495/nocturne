"""Tests for factorized axial attention memory and sequence scaling."""

from __future__ import annotations

import torch
import pytest

from nocturne.models.attention import (
    AttentionConfig,
    SpatiotemporalAttentionModel,
)


def test_axial_attention_scaling():
    """Verify that factorized axial attention handles long sequences and populations."""
    B = 2
    T = 100
    N = 32
    config = AttentionConfig(
        obs_dim=N,
        d_model=32,
        num_heads=2,
        num_layers=2,
        dim_feedforward=64,
        dropout=0.0,
        bin_width_s=0.02,
        max_seq_len=128,
        causal=False,
    )
    model = SpatiotemporalAttentionModel(config, latent_dim=16)
    counts = torch.randint(0, 4, (B, T, N), dtype=torch.float32)

    rates_hz, latents, _, _ = model(counts)

    assert rates_hz.shape == (B, T, N)
    assert latents.shape == (B, T, 16)
    assert torch.isfinite(rates_hz).all()
    assert torch.isfinite(latents).all()
