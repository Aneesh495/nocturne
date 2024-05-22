"""Tests for OnlineCausalStreamer real-time chunk processing."""

from __future__ import annotations

import torch
import pytest

from nocturne.models.attention import (
    AttentionConfig,
    SpatiotemporalAttentionModel,
)
from nocturne.streaming import OnlineCausalStreamer


def test_streamer_push_chunk_shapes_and_latency():
    N = 8
    config = AttentionConfig(
        obs_dim=N,
        d_model=16,
        num_heads=2,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        bin_width_s=0.02,
        max_seq_len=64,
        causal=True,
    )
    model = SpatiotemporalAttentionModel(config, latent_dim=4)
    streamer = OnlineCausalStreamer(model=model, obs_dim=N, max_window_bins=30)

    # Push 3-bin chunk
    chunk = torch.randint(0, 4, (1, 3, N), dtype=torch.float32)
    res = streamer.push_chunk(chunk)

    assert res.predicted_rates_hz.shape == (1, 3, N)
    assert res.latents.shape == (1, 3, 4)
    assert res.latency_ms > 0.0
    assert res.total_bins_processed == 3
    assert (res.predicted_rates_hz >= 0.0).all()
