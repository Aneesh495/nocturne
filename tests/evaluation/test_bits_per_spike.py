"""Tests for BitsPerSpikeEvaluator and zero-spike denominator handling."""

from __future__ import annotations

import math
import numpy as np
import pytest

from nocturne.evaluation.bits_per_spike import BitsPerSpikeEvaluator


def test_bps_zero_spike_denominator_contract():
    evaluator = BitsPerSpikeEvaluator(bin_width_s=0.02)
    # Zero spikes across all observations
    targets = np.zeros((10, 2), dtype=np.float64)
    pred_rates = np.ones((10, 2), dtype=np.float64) * 10.0
    null_rates = np.ones((10, 2), dtype=np.float64) * 5.0

    res = evaluator.evaluate(pred_rates, targets, null_rates)

    assert res.total_spikes == 0
    assert res.active_neurons == 0
    assert res.population_bps is None, "Zero spike denominator must return None"
    assert res.mean_neuron_bps is None
    assert res.per_neuron_bps == [None, None]


def test_bps_positive_improvement():
    evaluator = BitsPerSpikeEvaluator(bin_width_s=0.02)
    # 2 neurons: neuron 0 has spikes, neuron 1 has spikes
    # Target counts:
    targets = np.array([
        [1.0, 0.0],
        [2.0, 1.0],
        [5.0, 0.0],
        [0.0, 3.0],
    ], dtype=np.float64)

    null_rates = np.mean(targets, axis=0) / 0.02
    pred_rates = targets / 0.02 + 0.1  # Accurate predictions

    res = evaluator.evaluate(pred_rates, targets, null_rates)

    assert res.total_spikes == 12
    assert res.active_neurons == 2
    assert res.population_bps is not None
    assert res.population_bps > 0.0, "Model better than null must have positive BPS"
