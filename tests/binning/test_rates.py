"""Unit tests for firing rate transformations, inverse softplus, and Anscombe variance stabilization."""

import numpy as np
import pytest
import torch

from nocturne.binning.rates import RateClippingStats, RateTransformer


def test_rate_clipping_numpy():
    transformer = RateTransformer(min_rate_hz=0.1, max_rate_hz=500.0)
    rates = np.array([0.001, 10.0, 450.0, 800.0])

    clipped, stats = transformer.clip_rates_numpy(rates)
    assert clipped[0] == pytest.approx(0.1)
    assert clipped[1] == pytest.approx(10.0)
    assert clipped[2] == pytest.approx(450.0)
    assert clipped[3] == pytest.approx(500.0)

    assert stats.total_values == 4
    assert stats.clipped_low_count == 1
    assert stats.clipped_high_count == 1
    assert stats.percent_clipped_low == pytest.approx(25.0)
    assert stats.percent_clipped_high == pytest.approx(25.0)


def test_rate_clipping_torch():
    transformer = RateTransformer(min_rate_hz=1.0, max_rate_hz=100.0)
    rates = torch.tensor([0.2, 50.0, 150.0])

    clipped, stats = transformer.clip_rates_torch(rates)
    assert clipped[0].item() == pytest.approx(1.0)
    assert clipped[1].item() == pytest.approx(50.0)
    assert clipped[2].item() == pytest.approx(100.0)
    assert stats.clipped_low_count == 1
    assert stats.clipped_high_count == 1


def test_softplus_and_inverse_softplus():
    transformer = RateTransformer(softplus_beta=1.0)

    # Test round trip: f(f_inv(y)) == y
    y_vals = np.array([0.01, 0.5, 2.0, 15.0, 50.0])
    x_inv = transformer.inverse_softplus(y_vals)
    y_recovered = transformer.softplus(x_inv)
    np.testing.assert_allclose(y_recovered, y_vals, rtol=1e-6)

    # Test round trip in PyTorch
    y_torch = torch.tensor([0.01, 0.5, 2.0, 15.0, 50.0])
    x_inv_t = transformer.inverse_softplus(y_torch)
    y_rec_t = transformer.softplus(x_inv_t)
    torch.testing.assert_close(y_rec_t, y_torch, rtol=1e-6, atol=1e-6)


def test_anscombe_transform():
    # Counts array: 0, 1, 4, 9, 16
    counts = np.array([0, 1, 4, 9, 16], dtype=np.float64)
    z = RateTransformer.anscombe_transform(counts)
    recovered_counts = RateTransformer.inverse_anscombe_transform(z)

    # For counts >= 1, recovery should be close
    np.testing.assert_allclose(recovered_counts[1:], counts[1:], rtol=1e-6)

    # In PyTorch
    counts_t = torch.tensor([0.0, 1.0, 4.0, 9.0, 16.0])
    z_t = RateTransformer.anscombe_transform(counts_t)
    rec_t = RateTransformer.inverse_anscombe_transform(z_t)
    torch.testing.assert_close(rec_t[1:], counts_t[1:], rtol=1e-6, atol=1e-6)
