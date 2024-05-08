"""Unit tests for continuous spike timestamp accumulation into discrete integer counts."""

import numpy as np
import pytest

from nocturne.binning.accumulator import BinnedSpikeCounts, SpikeAccumulator
from nocturne.contracts.neurons import NeuronMetadata, PopulationMetadata
from nocturne.contracts.time import BinConfig, TemporalGrid, TimeInterval


@pytest.fixture
def test_population():
    n0 = NeuronMetadata(neuron_id="n0", channel_index=0, brain_area="M1")
    n1 = NeuronMetadata(neuron_id="n1", channel_index=1, brain_area="PMd")
    return PopulationMetadata(session_id="test_ses", neurons=(n0, n1))


def test_spike_accumulator_basic(test_population):
    bin_config = BinConfig(bin_width_s=0.1, align_to_zero=True)
    interval = TimeInterval(start=0.0, stop=1.0)  # 10 bins: [0.0, 0.1), [0.1, 0.2), ...
    accumulator = SpikeAccumulator(bin_config)

    # Unit 0 fires at 0.05, 0.15, 0.18, 0.95, and 1.5 (out of bounds)
    # Unit 1 fires at -0.1 (out of bounds), 0.55
    spikes = {
        0: [0.05, 0.15, 0.18, 0.95, 1.5],
        1: [-0.1, 0.55],
    }

    binned = accumulator.bin_timestamps(spikes, interval, test_population)

    assert binned.num_bins == 10
    assert binned.num_neurons == 2
    assert binned.total_spikes == 5  # 4 from unit 0, 1 from unit 1

    # Check neuron 0 counts
    c0 = binned.get_neuron_counts(0)
    assert c0[0] == 1  # 0.05
    assert c0[1] == 2  # 0.15, 0.18
    assert c0[9] == 1  # 0.95
    assert np.sum(c0) == 4

    # Check neuron 1 counts
    c1 = binned.get_neuron_counts(1)
    assert c1[5] == 1  # 0.55
    assert np.sum(c1) == 1

    # Firing rates in Hz over 1.0 s
    rates = binned.compute_mean_rates_hz()
    np.testing.assert_allclose(rates, [4.0, 1.0])


def test_binned_counts_slicing(test_population):
    bin_config = BinConfig(bin_width_s=0.05, align_to_zero=True)
    interval = TimeInterval(start=0.0, stop=1.0)
    grid = TemporalGrid.from_interval(interval, bin_config)

    counts = np.ones((grid.num_bins, test_population.num_neurons), dtype=np.int64)
    binned = BinnedSpikeCounts(grid=grid, population=test_population, counts=counts)

    assert binned.num_bins == 20
    sub = binned.slice_time(5, 15)
    assert sub.num_bins == 10
    assert sub.grid.interval.start == pytest.approx(0.25)
    assert sub.grid.interval.stop == pytest.approx(0.75)


def test_binned_counts_invalid_values(test_population):
    bin_config = BinConfig(bin_width_s=0.1)
    grid = TemporalGrid.from_interval(TimeInterval(start=0.0, stop=0.5), bin_config)

    # Negative counts rejected
    with pytest.raises(ValueError, match="cannot be negative"):
        bad_counts = np.array([[-1, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
        BinnedSpikeCounts(grid=grid, population=test_population, counts=bad_counts)

    # Non-integer floats rejected
    with pytest.raises(ValueError, match="exact non-negative integers"):
        float_counts = np.array([[1.5, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        BinnedSpikeCounts(grid=grid, population=test_population, counts=float_counts)
