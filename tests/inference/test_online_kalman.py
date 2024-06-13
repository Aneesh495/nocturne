"""Unit tests for recursive Point-Process Extended Kalman Filter (PPEKF)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nocturne.inference.online_kalman import (
    PointProcessEKF,
    PointProcessKalmanState,
)


def test_ppekf_tracks_latent_trajectory() -> None:
    np.random.seed(42)
    T, d, N = 200, 2, 32
    dt = 0.02

    # Known transition and emission parameters
    A = np.array([[0.95, -0.1], [0.1, 0.95]])
    Q = np.eye(d) * 0.01
    C = np.random.randn(N, d) * 0.8
    b = np.full(N, 2.5)

    # Generate synthetic true latent path and spike counts
    true_x = np.zeros((T, d))
    curr_x = np.array([1.0, 0.0])
    counts = np.zeros((T, N), dtype=int)

    for t in range(T):
        curr_x = (A @ curr_x) + np.random.multivariate_normal(np.zeros(d), Q)
        true_x[t] = curr_x
        lam = np.exp(np.clip((C @ curr_x) + b, -5.0, 5.0))
        counts[t] = np.random.poisson(lam * dt)

    # Instantiate filter
    ekf = PointProcessEKF(
        latent_dim=d,
        num_channels=N,
        transition_matrix=A,
        process_noise_cov=Q,
        emission_weights=C,
        emission_biases=b,
        bin_width_s=dt,
    )

    states = ekf.filter_sequence(counts)

    assert len(states) == T
    est_x = np.array([s.latent_mean for s in states])

    # Filtered trajectory should positively correlate with true trajectory
    corr_dim0 = float(np.corrcoef(true_x[:, 0], est_x[:, 0])[0, 1])
    corr_dim1 = float(np.corrcoef(true_x[:, 1], est_x[:, 1])[0, 1])

    assert corr_dim0 > 0.50
    assert corr_dim1 > 0.50

    # Covariance matrices must remain positive definite
    for s in states:
        eigvals = np.linalg.eigvalsh(s.latent_covariance)
        assert (eigvals > 0.0).all()


def test_ppekf_step_and_reset() -> None:
    ekf = PointProcessEKF(latent_dim=3, num_channels=5)
    y1 = np.array([1, 0, 2, 0, 1])
    s1 = ekf.step(y1)

    assert isinstance(s1, PointProcessKalmanState)
    assert s1.latent_mean.shape == (3,)
    assert s1.predicted_rates_hz.shape == (5,)

    # Reset
    ekf.reset(initial_mean=np.ones(3))
    assert np.allclose(ekf.x, np.ones(3))


def test_ppekf_dimension_mismatch_raises() -> None:
    ekf = PointProcessEKF(latent_dim=2, num_channels=4)
    with pytest.raises(ValueError, match="Expected 1D count vector"):
        ekf.step(np.zeros(3))
