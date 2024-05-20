"""Unit tests for LowRankPoissonFactorModel."""

import pytest
import torch

from nocturne.models.baselines.poisson_factor import LowRankPoissonFactorModel


def test_poisson_factor_forward_and_fit():
    num_neurons = 8
    rank = 3
    model = LowRankPoissonFactorModel(num_neurons=num_neurons, rank=rank)

    # Test decode rates
    factors = torch.randn(2, 10, rank)
    rates = model.decode_rates(factors)
    assert rates.shape == (2, 10, num_neurons)
    assert torch.all(rates >= 0.0)

    # Test optimization on synthetic count matrix
    counts = torch.randint(0, 5, (2, 10, num_neurons)).float()
    fitted_factors, loss = model.fit_factors_and_loadings(
        counts=counts,
        bin_width_s=0.02,
        num_steps=10,
        lr=0.05,
    )
    assert fitted_factors.shape == (2, 10, rank)
    assert loss > 0.0
