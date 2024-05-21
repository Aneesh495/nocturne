"""Tests for forward execution, loss computation, and gradients in STA model."""

from __future__ import annotations

import torch
import pytest

from nocturne.models.attention import (
    AttentionConfig,
    SpatiotemporalAttentionModel,
)


def test_attention_forward_shapes():
    B, T, N = 2, 20, 16
    config = AttentionConfig(
        obs_dim=N,
        d_model=32,
        num_heads=2,
        num_layers=2,
        dim_feedforward=64,
        dropout=0.0,
        bin_width_s=0.02,
        max_seq_len=64,
        causal=False,
    )
    model = SpatiotemporalAttentionModel(config, latent_dim=8)
    counts = torch.randint(0, 5, (B, T, N), dtype=torch.float32)

    rates_hz, latents, loss, metrics = model(counts)

    assert rates_hz.shape == (B, T, N)
    assert latents.shape == (B, T, 8)
    assert loss is None
    assert (rates_hz >= 0.0).all(), "Firing rates must be strictly non-negative"


def test_attention_forward_with_loss_and_gradients():
    B, T, N = 2, 15, 12
    config = AttentionConfig(
        obs_dim=N,
        d_model=32,
        num_heads=2,
        num_layers=2,
        dim_feedforward=64,
        dropout=0.0,
        bin_width_s=0.02,
        max_seq_len=64,
        causal=False,
    )
    model = SpatiotemporalAttentionModel(config, latent_dim=8)
    counts = torch.randint(0, 5, (B, T, N), dtype=torch.float32)
    targets = torch.randint(0, 5, (B, T, N), dtype=torch.float32)
    mask = torch.rand((B, T, N)) > 0.3

    rates_hz, latents, loss, metrics = model(
        counts=counts,
        target_counts=targets,
        loss_mask=mask,
    )

    assert loss is not None
    assert torch.isfinite(loss)
    assert "poisson_loss" in metrics
    assert "mean_pred_rate" in metrics

    total_objective = loss + latents.sum()
    total_objective.backward()

    # Verify gradients flowed into embeddings, transformer blocks, rate head, and latent projection
    assert model.embedding.count_proj.weight.grad is not None
    assert model.rate_head.weight.grad is not None
    assert model.latent_proj.weight.grad is not None


