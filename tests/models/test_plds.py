"""Unit tests for PLDS, Laplace smoother, and comparison with Kalman reference."""

import numpy as np
import pytest
import torch

from nocturne.models.baselines.kalman_reference import KalmanSmootherReference
from nocturne.models.baselines.plds import PoissonLinearDynamicalSystem
from nocturne.models.baselines.plds_laplace import PLDSLaplaceSmoother


def test_plds_sampling():
    plds = PoissonLinearDynamicalSystem(latent_dim=3, num_neurons=12, seed=123)
    latents, rates, counts = plds.sample_trajectory(num_time_bins=20, bin_width_s=0.02)

    assert latents.shape == (20, 3)
    assert rates.shape == (20, 12)
    assert counts.shape == (20, 12)
    assert np.all(rates > 0.0)
    assert np.all(counts >= 0)
    assert np.issubdtype(counts.dtype, np.integer)


def test_plds_laplace_inference():
    plds = PoissonLinearDynamicalSystem(latent_dim=2, num_neurons=8, seed=42)
    _, _, counts_np = plds.sample_trajectory(num_time_bins=15, bin_width_s=0.02)

    counts = torch.from_numpy(counts_np).float()
    smoother = PLDSLaplaceSmoother(plds)
    map_z, post_var = smoother.infer_latents_map(counts, bin_width_s=0.02, num_iter=20)

    assert map_z.shape == (15, 2)
    assert post_var.shape == (15, 2)
    assert torch.all(post_var > 0.0)


def test_kalman_smoother_exactness():
    # Linear Gaussian system
    A = np.array([[0.8, 0.2], [-0.1, 0.85]])
    Q = np.diag([0.1, 0.1])
    C = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    R = np.diag([0.2, 0.2, 0.2])
    mu_0 = np.array([0.0, 0.0])
    P_0 = np.eye(2)

    kalman = KalmanSmootherReference(A=A, Q=Q, C=C, R=R, mu_0=mu_0, P_0=P_0)

    # Synthetic observations
    T = 25
    rng = np.random.default_rng(101)
    true_z = np.zeros((T, 2))
    obs_y = np.zeros((T, 3))
    z = rng.multivariate_normal(mu_0, P_0)
    for t in range(T):
        if t > 0:
            z = A @ z + rng.multivariate_normal([0, 0], Q)
        true_z[t] = z
        obs_y[t] = C @ z + rng.multivariate_normal([0, 0, 0], R)

    mu_smooth, P_smooth = kalman.filter_and_smooth(obs_y)
    assert mu_smooth.shape == (T, 2)
    assert P_smooth.shape == (T, 2, 2)
    # Smoothed trajectory correlation with true states should be high
    corr0 = np.corrcoef(mu_smooth[:, 0], true_z[:, 0])[0, 1]
    corr1 = np.corrcoef(mu_smooth[:, 1], true_z[:, 1])[0, 1]
    assert corr0 > 0.6
    assert corr1 > 0.6
