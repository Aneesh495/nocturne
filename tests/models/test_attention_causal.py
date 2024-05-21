"""Tests for causal temporal attention and prefix invariance."""

from __future__ import annotations

import torch
import pytest

from nocturne.models.attention import (
    AttentionConfig,
    SpatiotemporalAttentionModel,
)


def test_causal_prefix_invariance():
    """Verify that future inputs do not leak into past or present predictions."""
    torch.manual_seed(42)
    B, T, N = 1, 20, 8
    config = AttentionConfig(
        obs_dim=N,
        d_model=32,
        num_heads=2,
        num_layers=2,
        dim_feedforward=64,
        dropout=0.0,
        bin_width_s=0.02,
        max_seq_len=64,
        causal=True,
    )
    model = SpatiotemporalAttentionModel(config, latent_dim=8)
    model.eval()

    # Original inputs
    counts_orig = torch.randint(0, 5, (B, T, N), dtype=torch.float32)

    with torch.no_grad():
        rates_orig, latents_orig, _, _ = model(counts_orig, causal=True)

    # Modify future inputs beyond cutoff t_cutoff = 10
    t_cutoff = 10
    counts_perturbed = counts_orig.clone()
    counts_perturbed[:, t_cutoff:, :] = counts_perturbed[:, t_cutoff:, :] + 10.0

    with torch.no_grad():
        rates_pert, latents_pert, _, _ = model(counts_perturbed, causal=True)

    # Assert rates and latents for t <= t_cutoff are bitwise/numerically identical
    max_diff_rates = (rates_orig[:, :t_cutoff, :] - rates_pert[:, :t_cutoff, :]).abs().max().item()
    max_diff_latents = (latents_orig[:, :t_cutoff, :] - latents_pert[:, :t_cutoff, :]).abs().max().item()

    assert max_diff_rates < 1e-5, f"Causal leakage detected in firing rates! Max diff: {max_diff_rates}"
    assert max_diff_latents < 1e-5, f"Causal leakage detected in latents! Max diff: {max_diff_latents}"
