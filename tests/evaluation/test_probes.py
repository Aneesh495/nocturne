"""Tests for BehavioralDecodingProbe train-only ridge fit and out-of-sample R2."""

from __future__ import annotations

import numpy as np
import pytest

from nocturne.evaluation.probes import BehavioralDecodingProbe


def test_probe_train_only_out_of_sample_decoding():
    np.random.seed(42)
    N_train = 200
    N_test = 50
    D = 4
    K = 2

    # Latents
    Z_train = np.random.randn(N_train, D)
    Z_test = np.random.randn(N_test, D)

    # True linear projection
    W_true = np.array([
        [1.0, 0.0],
        [0.5, -1.0],
        [0.0, 2.0],
        [-0.5, 0.5],
    ])
    b_true = np.array([0.5, -0.5])

    # Behavior
    V_train = Z_train @ W_true + b_true + 0.05 * np.random.randn(N_train, K)
    V_test = Z_test @ W_true + b_true + 0.05 * np.random.randn(N_test, K)

    probe = BehavioralDecodingProbe(ridge_alpha=1e-3)
    probe.fit(Z_train, V_train)

    res = probe.evaluate(Z_test, V_test)

    assert res.macro_r2 > 0.95
    assert len(res.per_dim_r2) == K
    assert all(r > 0.95 for r in res.per_dim_r2)
    assert res.macro_pearson_r > 0.95
    assert res.mse < 0.05
