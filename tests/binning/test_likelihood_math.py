"""Unit tests for Poisson likelihood math, log-factorial tables, and deviance."""

import math
import numpy as np
import pytest
import scipy.special

from nocturne.binning.likelihood_math import (
    LogFactorialLUT,
    get_log_factorial,
    poisson_bits_per_spike,
    poisson_deviance,
    poisson_log_likelihood,
)


def test_log_factorial_lut():
    lut = LogFactorialLUT()

    # Small values
    assert lut.log_factorial(0) == pytest.approx(0.0)
    assert lut.log_factorial(1) == pytest.approx(0.0)
    assert lut.log_factorial(2) == pytest.approx(math.log(2.0))
    assert lut.log_factorial(5) == pytest.approx(math.log(120.0))

    # Array vectorization
    k_arr = np.array([0, 1, 2, 3, 4, 5])
    expected = scipy.special.gammaln(k_arr + 1.0)
    np.testing.assert_allclose(lut.log_factorial(k_arr), expected, rtol=1e-12)

    # Large k (> 1024) tested against scipy gammaln
    large_k = np.array([1025, 2000, 5000])
    expected_large = scipy.special.gammaln(large_k + 1.0)
    actual_large = lut.log_factorial(large_k)
    np.testing.assert_allclose(actual_large, expected_large, rtol=1e-8)

    # Negative value error
    with pytest.raises(ValueError):
        lut.log_factorial(-1)


def test_poisson_log_likelihood_analytic():
    # 1 bin, 1 neuron
    # y = 2 spikes, rate = 100 Hz, delta_t = 0.02s -> mu = 2.0 expected spikes
    # Analytic P(Y=2 | mu=2) = 2^2 * exp(-2) / 2! = 2 * exp(-2) = 0.2706705664732254
    # ln P = ln(0.2706705664732254) = -1.3068528194400546
    counts = np.array([[2]])
    rates = np.array([[100.0]])
    dt = 0.02

    ll = poisson_log_likelihood(counts, rates, dt, include_log_factorial=True)
    expected_ll = math.log(2.0 * math.exp(-2.0))
    assert ll == pytest.approx(expected_ll, rel=1e-7)

    # When y = 0, mu = 2.0 -> P(Y=0 | mu=2) = exp(-2)
    counts_zero = np.array([[0]])
    ll_zero = poisson_log_likelihood(counts_zero, rates, dt, include_log_factorial=True)
    assert ll_zero == pytest.approx(-2.0, rel=1e-7)


def test_poisson_log_likelihood_masking():
    counts = np.array([[2, 0], [1, 5]])
    rates = np.array([[100.0, 50.0], [50.0, 200.0]])
    dt = 0.02
    mask = np.array([[True, False], [False, True]])

    # Should only evaluate counts[0, 0] (y=2, rate=100) and counts[1, 1] (y=5, rate=200)
    ll_masked = poisson_log_likelihood(counts, rates, dt, mask=mask)

    ll_00 = poisson_log_likelihood(np.array([[2]]), np.array([[100.0]]), dt)
    ll_11 = poisson_log_likelihood(np.array([[5]]), np.array([[200.0]]), dt)
    assert ll_masked == pytest.approx(ll_00 + ll_11, rel=1e-7)


def test_poisson_deviance_properties():
    # When rate matches counts exactly (mu = y), deviance is 0
    counts = np.array([[0, 2, 5]])
    dt = 0.01
    rates_perfect = counts / dt  # mu = counts
    dev_zero = poisson_deviance(counts, rates_perfect, dt)
    assert dev_zero == pytest.approx(0.0, abs=1e-7)

    # When rate does not match, deviance is strictly positive
    rates_imperfect = np.array([[10.0, 10.0, 10.0]])
    dev_pos = poisson_deviance(counts, rates_imperfect, dt)
    assert dev_pos > 0.0


def test_poisson_bits_per_spike():
    counts = np.array([[2, 4], [1, 3]])
    dt = 0.02
    # Model rates closer to counts
    model_rates = counts / dt
    # Baseline reference rate: uniform mean rate
    mean_rate = np.mean(counts) / dt
    ref_rates = np.full_like(model_rates, mean_rate)

    bps = poisson_bits_per_spike(counts, model_rates, ref_rates, dt)
    assert bps is not None
    assert bps > 0.0  # Perfect model has higher likelihood than mean rate -> BPS > 0

    # When model equals reference, BPS is exactly 0
    bps_zero = poisson_bits_per_spike(counts, ref_rates, ref_rates, dt)
    assert bps_zero == pytest.approx(0.0)

    # Zero-spike denominator test: returns None instead of NaN / division by zero
    zero_counts = np.zeros_like(counts)
    bps_undefined = poisson_bits_per_spike(zero_counts, model_rates, ref_rates, dt)
    assert bps_undefined is None
