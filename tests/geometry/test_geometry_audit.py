"""Acceptance Gate BA09: 200+ held-out circuit trials for gauge-aware dynamics audit."""

from __future__ import annotations

import numpy as np
import pytest

from nocturne.geometry import PredictionStructureDivergenceAudit


def test_geometry_audit_200_held_out_trials():
    """Verify gauge-aware dynamics audit on >=200 held-out evaluation trials."""
    np.random.seed(42)
    B_train = 50
    B_test = 200  # Acceptance Gate BA09 requires >= 200 held-out trials
    T = 25
    D = 4

    # Generate ground truth latent trajectories
    true_train = np.random.randn(B_train, T, D)
    true_test = np.random.randn(B_test, T, D)

    # Inferred latents generated via fixed orthogonal transformation + mild noise
    Q, _ = np.linalg.qr(np.random.randn(D, D))
    inferred_train = true_train @ Q + 0.05 * np.random.randn(B_train, T, D)
    inferred_test = true_test @ Q + 0.05 * np.random.randn(B_test, T, D)

    audit = PredictionStructureDivergenceAudit(disparity_threshold=0.3)
    report = audit.audit_dynamics(
        true_latents_train=true_train,
        inferred_latents_train=inferred_train,
        true_latents_test=true_test,
        inferred_latents_test=inferred_test,
    )

    assert report.num_trials_evaluated == 200
    assert report.has_geometric_fidelity
    assert report.test_procrustes_disparity < 0.15
    assert report.mean_cca_correlation > 0.90
    assert len(report.canonical_correlations) == D
