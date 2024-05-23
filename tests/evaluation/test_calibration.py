"""Tests for DispersionAndCoverageAuditor Fano factors and Poisson interval coverage."""

from __future__ import annotations

import numpy as np
import pytest

from nocturne.evaluation.calibration import DispersionAndCoverageAuditor


def test_calibration_pure_poisson_coverage():
    np.random.seed(42)
    N_samples = 500
    N_neurons = 4
    bin_width_s = 0.02

    # Firing rates in Hz
    rates_hz = np.array([10.0, 25.0, 50.0, 5.0])
    mu = rates_hz * bin_width_s

    # Generate synthetic Poisson counts
    targets = np.random.poisson(mu, size=(N_samples, N_neurons))
    pred_rates = np.tile(rates_hz, (N_samples, 1))

    auditor = DispersionAndCoverageAuditor(bin_width_s=bin_width_s)
    report = auditor.audit_calibration(pred_rates, targets)

    # For true Poisson counts, Fano factor is ~ 1.0
    assert 0.8 < report.mean_fano_factor < 1.2
    assert not report.is_overdispersed
    assert not report.is_underdispersed

    # 95% nominal interval should cover ~ 95% of true counts (+/- 3%)
    assert 0.90 < report.nominal_95_coverage < 0.99
    assert abs(report.mean_rate_bias) < 2.0
