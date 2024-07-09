"""Unit tests for the end-to-end streaming simulation workflow."""

from __future__ import annotations

import pytest
from nocturne.workflows.streaming_simulation import (
    StreamingSimulationResult,
    run_streaming_simulation,
)


def test_run_streaming_simulation_basic() -> None:
    res = run_streaming_simulation(
        num_neurons=8,
        latent_dim=2,
        total_bins=20,
        chunk_size_bins=2,
        bin_width_ms=20,
        rate_threshold_hz=20.0,
        circuit_name="linear_oscillator",
        seed=123,
    )

    assert isinstance(res, StreamingSimulationResult)
    assert res.total_chunks == 10
    assert res.total_bins == 20
    assert res.mean_latency_ms > 0.0
    assert res.p95_latency_ms >= res.mean_latency_ms * 0.5
    assert res.max_latency_ms >= res.p95_latency_ms
    assert res.reconstruction_rmse >= 0.0
    assert "Nocturne Online Streaming Latency Monitor" in res.ascii_summary
    assert "<svg" in res.svg_monitor
    assert "</svg>" in res.svg_monitor
    assert isinstance(res.stim_events, list)


def test_run_streaming_simulation_high_threshold_no_stim() -> None:
    res = run_streaming_simulation(
        num_neurons=8,
        latent_dim=2,
        total_bins=16,
        chunk_size_bins=4,
        bin_width_ms=20,
        rate_threshold_hz=1000.0,  # Unreachable threshold
        circuit_name="linear_oscillator",
        seed=999,
    )

    assert res.num_stim_events == 0
    assert res.num_blanked_bins == 0
    assert len(res.stim_events) == 0
