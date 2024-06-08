"""Unit tests for local tangent space alignment and Grassmannian drift."""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from nocturne.geometry.tangent import TangentSpaceEstimator, TangentSpaceResult


def test_2d_circle_embedded_in_high_dimensions() -> None:
    T = 200
    t = np.linspace(0, 4 * math.pi, T)
    # Circle in first 2 coordinates, 8 noise-free zero coordinates in R^10
    traj = np.zeros((T, 10))
    traj[:, 0] = np.cos(t)
    traj[:, 1] = np.sin(t)

    estimator = TangentSpaceEstimator(window_size=8, k_dim=2)
    res = estimator.analyze_trajectory(traj)

    assert isinstance(res, TangentSpaceResult)
    # The trajectory is a 1D curve in a 2D plane, so participation ratio in local windows is <= 2.0
    assert res.mean_intrinsic_dim <= 2.0
    assert res.mean_intrinsic_dim >= 1.0
    # Tangent plane rotates smoothly along the circle
    assert res.mean_grassmann_drift > 0.0
    assert res.max_grassmann_drift < 1.0


def test_high_dimensional_random_walk() -> None:
    np.random.seed(42)
    T = 150
    # High-dimensional isotropic random walk in R^16
    steps = np.random.randn(T, 16)
    traj = np.cumsum(steps, axis=0)

    estimator = TangentSpaceEstimator(window_size=10, k_dim=4)
    res = estimator.analyze_trajectory(traj)

    # In isotropic diffusion, participation ratio should be substantially higher than 1.5
    assert res.mean_intrinsic_dim > 2.0


def test_pytorch_tensor_input() -> None:
    t = torch.linspace(0, 10, 100)
    traj = torch.stack([t, t**2, torch.sin(t)], dim=1)  # [100, 3]

    estimator = TangentSpaceEstimator(window_size=5, k_dim=2)
    res = estimator.analyze_trajectory(traj)

    assert isinstance(res, TangentSpaceResult)
    assert len(res.participation_ratios) == 100 - 2 * 5


def test_window_size_validation() -> None:
    short_traj = np.ones((10, 4))
    estimator = TangentSpaceEstimator(window_size=10)
    with pytest.raises(ValueError, match="too short"):
        estimator.analyze_trajectory(short_traj)
