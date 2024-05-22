"""Tests for online causal filtering."""

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
    CausalFilter,
)


def test_causal_filtering_execution():
    B, T, N = 2, 20, 8
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
    cfilter = CausalFilter(model)

    counts = torch.randint(0, 4, (B, T, N), dtype=torch.float32)
    req = InferenceRequest(
        counts=counts,
        mode=InferenceMode.ONLINE_FILTERING,
    )

    res = cfilter.infer(req)

    assert res.mode == InferenceMode.ONLINE_FILTERING
    assert res.predicted_rates_hz.shape == (B, T, N)
    assert res.latents.shape == (B, T, 4)
    assert (res.predicted_rates_hz >= 0.0).all()
