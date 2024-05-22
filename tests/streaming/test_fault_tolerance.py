"""Acceptance Gate BA11: 50 gap, 50 malformed, and 50 interruption recovery cases."""

from __future__ import annotations

import torch
import pytest

from nocturne.models.attention import (
    AttentionConfig,
    SpatiotemporalAttentionModel,
)
from nocturne.streaming import (
    OnlineCausalStreamer,
    StreamFaultHandler,
)


def test_fault_tolerance_suite_150_cases():
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

    report = StreamFaultHandler.audit_fault_resilience(
        streamer=streamer,
        num_gap_cases=50,
        num_malformed_cases=50,
        num_interruption_cases=50,
    )

    assert report.is_resilient
    assert report.gap_cases_recovered == 50
    assert report.malformed_cases_handled == 50
    assert report.interruption_cases_resumed == 50
