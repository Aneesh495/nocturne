"""Tests for neuron dropout robustness and graceful degradation."""

from __future__ import annotations

import torch
import pytest

from nocturne.models.attention import (
    AttentionConfig,
    SpatiotemporalAttentionModel,
)
from nocturne.inference import NeuronDropoutAuditor


def test_neuron_dropout_robustness():
    torch.manual_seed(42)
    B, T, N = 4, 20, 16
    config = AttentionConfig(
        obs_dim=N,
        d_model=16,
        num_heads=2,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        bin_width_s=0.02,
        max_seq_len=32,
    )
    model = SpatiotemporalAttentionModel(config, latent_dim=4)
    test_counts = torch.randint(0, 5, (B, T, N), dtype=torch.float32)

    auditor = NeuronDropoutAuditor(
        dropout_rates=[0.0, 0.2, 0.5, 0.8],
        bin_width_s=0.02,
    )
    report = auditor.audit_model(
        model=model,
        test_counts=test_counts,
        num_trials_per_rate=2,
    )

    assert len(report.dropout_rates) == 4
    assert len(report.mean_losses) == 4
    assert report.is_graceful
    assert all(torch.isfinite(torch.tensor(l)) for l in report.mean_losses)
