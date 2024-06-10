"""Unit tests for point-process GLM with raised-cosine history filters."""

from __future__ import annotations

import pytest
import torch

from nocturne.binning.point_process_glm import (
    PointProcessGLM,
    RaisedCosineBasis,
)


def test_raised_cosine_basis_properties() -> None:
    W, num_bases = 30, 6
    basis_gen = RaisedCosineBasis(num_bases=num_bases, history_length_bins=W)

    mat = basis_gen.basis_matrix
    assert mat.shape == (W, num_bases)
    # Basis functions are non-negative
    assert (mat >= 0.0).all()
    # Each column has unit norm
    for b in range(num_bases):
        col_norm = float(torch.norm(torch.tensor(mat[:, b])))
        assert pytest.approx(col_norm, rel=1e-3) == 1.0


def test_glm_self_history_forward_and_causality() -> None:
    B, T, N = 2, 40, 5
    glm = PointProcessGLM(
        num_channels=N,
        num_bases=4,
        history_length_bins=10,
        include_cross_coupling=False,
        bin_width_s=0.02,
    )

    counts = torch.poisson(torch.full((B, T, N), 1.0))
    rates = glm(counts)

    assert rates.shape == (B, T, N)
    assert (rates > 0.0).all()

    # Strict causality test: perturb counts at time step 25
    counts_perturbed = counts.clone()
    counts_perturbed[:, 25:, :] += 10.0

    rates_perturbed = glm(counts_perturbed)

    # Rates at time steps t <= 25 must be identical (no future leakage)
    assert torch.allclose(rates[:, :26, :], rates_perturbed[:, :26, :], atol=1e-5)


def test_glm_cross_coupling_and_gradients() -> None:
    B, T, N = 4, 30, 6
    glm = PointProcessGLM(
        num_channels=N,
        num_bases=3,
        history_length_bins=8,
        include_cross_coupling=True,
        bin_width_s=0.02,
    )

    counts = torch.poisson(torch.full((B, T, N), 0.8))
    nll = glm.compute_nll(counts)

    assert nll.item() > 0.0
    nll.backward()

    assert glm.bias.grad is not None
    assert glm.history_weights.grad is not None
    assert torch.all(torch.isfinite(glm.bias.grad))
    assert torch.all(torch.isfinite(glm.history_weights.grad))
