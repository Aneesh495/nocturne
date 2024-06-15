"""Unit tests for mutual information estimation (KSG and Gaussian bound)."""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from nocturne.evaluation.information import (
    MutualInformationEstimator,
    MutualInformationResult,
)


def test_independent_variables_near_zero_mi() -> None:
    np.random.seed(42)
    N = 400
    # Two completely independent standard Gaussian variables
    X = np.random.randn(N, 2)
    Y = np.random.randn(N, 2)

    estimator = MutualInformationEstimator(k_neighbors=4)
    res = estimator.estimate(X, Y)

    assert isinstance(res, MutualInformationResult)
    # Theoretical MI of independent variables is 0.0 bits
    assert res.mutual_information_bits < 0.20
    assert res.gaussian_linear_bits < 0.20


def test_strongly_correlated_variables_high_mi() -> None:
    np.random.seed(42)
    N = 400
    X = np.random.randn(N, 2)
    # Y is a noisy version of X: high mutual information
    Y = X + 0.1 * np.random.randn(N, 2)

    estimator = MutualInformationEstimator(k_neighbors=4)
    res = estimator.estimate(X, Y)

    assert res.mutual_information_bits > 1.5
    assert res.gaussian_linear_bits > 1.5


def test_nonlinear_relationship_captures_excess_bits() -> None:
    np.random.seed(42)
    N = 500
    # X symmetric around 0, Y = X^2 (zero linear correlation, but deterministic nonlinear dependency)
    x = np.random.uniform(-2.0, 2.0, size=(N, 1))
    y = (x**2) + 0.1 * np.random.randn(N, 1)

    # Linear correlation is near zero
    r_linear = np.corrcoef(x[:, 0], y[:, 0])[0, 1]
    assert abs(r_linear) < 0.15

    estimator = MutualInformationEstimator(k_neighbors=4)
    res = estimator.estimate(x, y)

    # KSG should detect the non-linear dependency and report non-linear excess bits
    assert res.mutual_information_bits > 0.50
    assert res.non_linear_excess_bits > 0.30


def test_pytorch_tensor_and_batched_shape() -> None:
    estimator = MutualInformationEstimator(k_neighbors=3)
    # Batched sequence tensors [B=4, T=50, d=3]
    latents = torch.randn(4, 50, 3)
    behavior = torch.randn(4, 50, 2)

    res = estimator.estimate(latents, behavior)
    assert res.sample_size == 200
    assert res.k_neighbors == 3


def test_sample_count_mismatch_raises() -> None:
    estimator = MutualInformationEstimator()
    with pytest.raises(ValueError, match="mismatch"):
        estimator.estimate(np.ones((10, 2)), np.ones((15, 2)))
