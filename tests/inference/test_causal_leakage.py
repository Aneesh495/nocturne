"""Acceptance Gate BA07: 1,000 perturbation cases for causal prefix invariance."""

from __future__ import annotations

import torch
import pytest

from nocturne.models.attention import (
    AttentionConfig,
    SpatiotemporalAttentionModel,
)
from nocturne.inference import CausalLeakageAuditor


def test_causal_prefix_invariance_1000_cases():
    """Verify that future-input perturbations across 1,000 test cases produce zero leakage."""
    config = AttentionConfig(
        obs_dim=8,
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

    auditor = CausalLeakageAuditor(rate_tolerance=5e-5, latent_tolerance=1e-5)
    report = auditor.audit_model(
        model=model,
        num_cases=1000,
        batch_size=50,
        seq_len=25,
        obs_dim=8,
        seed=42,
    )

    assert report.total_cases == 1000
    assert report.passed_cases == 1000, f"Failed {report.failed_cases} cases! Violations: {report.violations[:3]}"
    assert report.failed_cases == 0
    assert report.is_strictly_causal
    assert report.max_rate_deviation <= 5e-5
    assert report.max_latent_deviation <= 1e-5

