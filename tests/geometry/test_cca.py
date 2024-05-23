"""Tests for CanonicalCorrelationAnalysis on coupled multivariate signals."""

from __future__ import annotations

import numpy as np
import pytest

from nocturne.geometry import CanonicalCorrelationAnalysis


def test_cca_perfectly_correlated_subspace():
    np.random.seed(42)
    N = 300
    # Common shared latent
    z1 = np.random.randn(N, 1)
    z2 = np.random.randn(N, 1)

    # X has z1 and independent noise
    X = np.hstack([z1, np.random.randn(N, 2)])
    # Y has linear combination of z1 and independent noise
    Y = np.hstack([2.5 * z1 + 0.5, np.random.randn(N, 3)])

    cca = CanonicalCorrelationAnalysis()
    res = cca.fit(X, Y)

    # First canonical correlation should be extremely close to 1.0
    assert len(res.canonical_correlations) == 3
    assert res.canonical_correlations[0] > 0.98
    assert res.mean_correlation > 0.3
