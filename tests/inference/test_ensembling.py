"""Unit tests for deep ensemble rate fusion and uncertainty decomposition."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from nocturne.inference.ensembling import (
    DeepRateEnsemble,
    EnsemblePredictionResult,
)


class ConstantRateModel(nn.Module):
    """Mock model emitting fixed firing rates."""

    def __init__(self, rate_hz: float):
        super().__init__()
        self.rate_hz = rate_hz

    def forward(
        self,
        counts: torch.Tensor,
        observed_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, N = counts.shape
        return torch.full((B, T, N), self.rate_hz, dtype=torch.float32, device=counts.device)


def test_identical_models_zero_epistemic_variance() -> None:
    ens = DeepRateEnsemble(bin_width_s=0.02)
    ens.add_model(ConstantRateModel(rate_hz=20.0), weight=1.0)
    ens.add_model(ConstantRateModel(rate_hz=20.0), weight=1.0)

    counts = torch.zeros(2, 10, 4)
    res = ens.predict_with_uncertainty(counts)

    assert isinstance(res, EnsemblePredictionResult)
    assert res.mean_rates_hz.shape == (2, 10, 4)
    assert torch.allclose(res.mean_rates_hz, torch.full((2, 10, 4), 20.0))

    # Identical models have zero disagreement (epistemic variance = 0)
    assert torch.allclose(res.epistemic_variance, torch.zeros_like(res.epistemic_variance))

    # Point-process shot noise is non-zero: lambda / dt = 20 / 0.02 = 1000
    expected_aleatoric = 20.0 / 0.02
    assert torch.allclose(res.aleatoric_variance, torch.full_like(res.aleatoric_variance, expected_aleatoric))


def test_divergent_models_positive_epistemic_variance() -> None:
    ens = DeepRateEnsemble(bin_width_s=0.02)
    ens.add_model(ConstantRateModel(rate_hz=10.0), weight=3.0)
    ens.add_model(ConstantRateModel(rate_hz=30.0), weight=1.0)

    counts = torch.zeros(2, 5, 3)
    res = ens.predict_with_uncertainty(counts)

    # Weighted mean: 0.75 * 10 + 0.25 * 30 = 7.5 + 7.5 = 15.0
    assert torch.allclose(res.mean_rates_hz, torch.full((2, 5, 3), 15.0))

    # Epistemic variance: 0.75 * (10 - 15)^2 + 0.25 * (30 - 15)^2 = 0.75 * 25 + 0.25 * 225 = 18.75 + 56.25 = 75.0
    assert torch.allclose(res.epistemic_variance, torch.full_like(res.epistemic_variance, 75.0))
    assert (res.total_count_variance > 0).all()


def test_empty_ensemble_raises_runtime_error() -> None:
    ens = DeepRateEnsemble()
    counts = torch.zeros(1, 4, 2)
    with pytest.raises(RuntimeError, match="no registered models"):
        ens.predict_with_uncertainty(counts)
