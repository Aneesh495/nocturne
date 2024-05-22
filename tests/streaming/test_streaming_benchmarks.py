"""Tests for StreamingBenchmarkRunner latency distributions and bounded memory."""

from __future__ import annotations

import torch
import pytest

from nocturne.models.attention import (
    AttentionConfig,
    SpatiotemporalAttentionModel,
)
from nocturne.streaming import (
    OnlineCausalStreamer,
    StreamingBenchmarkRunner,
)


def test_streaming_benchmark_execution():
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
    streamer = OnlineCausalStreamer(model=model, obs_dim=N, max_window_bins=20)

    runner = StreamingBenchmarkRunner(bin_duration_ms=20.0)
    metrics = runner.run_benchmark(
        streamer=streamer,
        num_chunks=50,
        chunk_size_bins=1,
        warmup_chunks=5,
    )

    assert metrics.num_chunks_evaluated == 50
    assert metrics.mean_latency_ms > 0.0
    assert metrics.throughput_chunks_per_sec > 0.0
    assert metrics.memory_bounded_verified
