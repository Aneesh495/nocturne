"""Unit tests for MeanRateBaseline model."""

import numpy as np
import pytest
import torch

from nocturne.models.baselines.mean_rate import MeanRateBaseline


def test_mean_rate_baseline_fit():
    baseline = MeanRateBaseline(num_neurons=3)
    assert not baseline.is_fitted

    # 10 bins of 20 ms -> 0.2s duration
    dt = 0.02
    # Neuron 0: 4 spikes (20 Hz), Neuron 1: 0 spikes (floor 1e-4), Neuron 2: 10 spikes (50 Hz)
    counts = np.zeros((10, 3), dtype=np.int64)
    counts[:, 0] = [1, 0, 1, 0, 1, 0, 1, 0, 0, 0]  # 4 spikes
    counts[:, 2] = 1                                 # 10 spikes

    baseline.fit_from_counts(counts, bin_width_s=dt)
    assert baseline.is_fitted

    rates = baseline.predict_numpy(num_time_bins=5)
    assert rates.shape == (5, 3)
    assert rates[0, 0] == pytest.approx(4.0 / 0.2)
    assert rates[0, 2] == pytest.approx(10.0 / 0.2)

    # Tensor forward pass
    t_out = baseline(batch_size=2, num_time_bins=5)
    assert t_out.shape == (2, 5, 3)
