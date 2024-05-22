"""Tests for autoregressive future horizon forecasting."""

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
    AutoregressiveForecaster,
)


def test_autoregressive_forecasting_shape():
    B, T_hist, N = 2, 10, 8
    H = 5
    config = AttentionConfig(
        obs_dim=N,
        d_model=16,
        num_heads=2,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        bin_width_s=0.02,
        max_seq_len=32,
        causal=True,
    )
    model = SpatiotemporalAttentionModel(config, latent_dim=4)
    forecaster = AutoregressiveForecaster(model)

    counts = torch.randint(0, 4, (B, T_hist, N), dtype=torch.float32)
    req = InferenceRequest(
        counts=counts,
        mode=InferenceMode.FORECASTING,
        forecast_horizon_bins=H,
    )

    res = forecaster.infer(req)

    assert res.mode == InferenceMode.FORECASTING
    assert res.predicted_rates_hz.shape == (B, H, N)
    assert res.latents.shape == (B, H, 4)
    assert (res.predicted_rates_hz >= 0.0).all()
