"""Unit tests for the reproducibility audit workflow."""

from __future__ import annotations

import pytest
from nocturne.workflows.reproducibility_audit import (
    ReproducibilityAuditResult,
    run_reproducibility_audit,
)


def test_run_reproducibility_audit_deterministic() -> None:
    res = run_reproducibility_audit(
        seed=12345,
        num_neurons=8,
        latent_dim=2,
        num_steps=3,
        seq_len=12,
    )

    assert isinstance(res, ReproducibilityAuditResult)
    assert res.is_bitwise_identical is True
    assert res.audit_passed is True
    assert res.max_weight_discrepancy < 1e-7
    assert res.max_loss_discrepancy < 1e-7
    assert res.sha256_run1 == res.sha256_run2
    assert res.is_non_trivial is True
    assert "PASSED" in res.summary_report
