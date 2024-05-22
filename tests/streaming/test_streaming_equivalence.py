"""Tests verifying numerical equivalence between batch causal filtering and chunked streaming."""

from __future__ import annotations

import torch
import pytest

from nocturne.models.attention import (
    AttentionConfig,
    SpatiotemporalAttentionModel,
)
from nocturne.streaming import StreamingEquivalenceAuditor


def test_streaming_equivalence_across_chunk_sizes():
    torch.manual_seed(42)
    T, N = 20, 8
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
    counts = torch.randint(0, 4, (1, T, N), dtype=torch.float32)

    auditor = StreamingEquivalenceAuditor(tolerance=5e-5)
    report = auditor.audit_equivalence(
        model=model,
        counts=counts,
        chunk_sizes=[1, 2, 5],
    )

    assert report.is_equivalent
    for sz in [1, 2, 5]:
        assert report.max_rate_deviations[sz] <= 5e-5
        assert report.max_latent_deviations[sz] <= 5e-5
