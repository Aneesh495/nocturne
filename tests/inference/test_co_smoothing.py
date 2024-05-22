"""Tests for offline co-smoothing inference and held-out channel extraction."""

from __future__ import annotations

import torch
import pytest

from nocturne.models.attention import (
    AttentionConfig,
    SpatiotemporalAttentionModel,
)
from nocturne.inference import (
    InferenceMode,
    InferenceRequest,
    CoSmoother,
)


def test_co_smoothing_held_out_slice():
    B, T, N = 2, 15, 8
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
    smoother = CoSmoother(model)

    counts = torch.randint(0, 5, (B, T, N), dtype=torch.float32)
    held_out = [2, 5]
    req = InferenceRequest(
        counts=counts,
        mode=InferenceMode.OFFLINE_SMOOTHING,
        held_out_indices=held_out,
    )

    res = smoother.infer(req)

    assert res.mode == InferenceMode.OFFLINE_SMOOTHING
    assert res.predicted_rates_hz.shape == (B, T, N)
    assert res.held_out_rates_hz is not None
    assert res.held_out_rates_hz.shape == (B, T, 2)
    assert (res.predicted_rates_hz >= 0.0).all()
    assert res.latency_seconds > 0.0
