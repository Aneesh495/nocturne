"""Tests for EvaluationMetricsCalculator deviance, NLL, and pseudo-R2."""

from __future__ import annotations

import numpy as np
import pytest

from nocturne.evaluation.metrics import EvaluationMetricsCalculator


def test_eval_metrics_perfect_predictions():
    calc = EvaluationMetricsCalculator(bin_width_s=0.02)
    # When predictions perfectly match targets, deviance must be 0.0
    targets = np.array([0, 1, 3, 5, 0], dtype=np.float64)
    pred_rates = targets / 0.02  # perfect rate match

    summary = calc.compute_metrics(pred_rates, targets)

    assert summary.total_spikes == 9
    assert summary.num_observations == 5
    assert summary.poisson_deviance == pytest.approx(0.0, abs=1e-6)
    assert summary.mae_rates == pytest.approx(0.0, abs=1e-6)


def test_eval_metrics_deviance_pseudo_r2():
    calc = EvaluationMetricsCalculator(bin_width_s=0.02)
    targets = np.array([1, 2, 4, 1, 5, 2], dtype=np.float64)
    null_rates = np.ones_like(targets) * (np.mean(targets) / 0.02)
    good_pred_rates = targets / 0.02

    summary = calc.compute_metrics(
        predicted_rates_hz=good_pred_rates,
        target_counts=targets,
        null_rates_hz=null_rates,
    )

    # Perfect model has pseudo-R2 of 1.0
    assert summary.deviance_pseudo_r2 is not None
    assert summary.deviance_pseudo_r2 == pytest.approx(1.0, abs=1e-5)
