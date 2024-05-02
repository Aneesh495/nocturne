"""Unit tests for temporal contracts, binning configurations, and temporal grids."""

import math
import numpy as np
import pytest

from nocturne.contracts.time import BinConfig, TemporalGrid, TimeInterval


def test_time_interval_valid():
    interval = TimeInterval(start=0.0, stop=1.5)
    assert interval.duration == pytest.approx(1.5)
    assert interval.contains(0.0) is True
    assert interval.contains(1.499) is True
    assert interval.contains(1.5) is False
    assert interval.contains(-0.1) is False


def test_time_interval_invalid():
    with pytest.raises(ValueError, match="must be greater than or equal to start"):
        TimeInterval(start=2.0, stop=1.0)

    with pytest.raises(ValueError, match="must be finite"):
        TimeInterval(start=float("nan"), stop=1.0)

    with pytest.raises(ValueError, match="must be finite"):
        TimeInterval(start=0.0, stop=float("inf"))


def test_time_interval_overlap_and_intersection():
    i1 = TimeInterval(start=0.0, stop=2.0)
    i2 = TimeInterval(start=1.0, stop=3.0)
    i3 = TimeInterval(start=2.5, stop=4.0)

    assert i1.overlaps(i2) is True
    assert i1.overlaps(i3) is False

    inter12 = i1.intersection(i2)
    assert inter12 is not None
    assert inter12.start == pytest.approx(1.0)
    assert inter12.stop == pytest.approx(2.0)

    inter13 = i1.intersection(i3)
    assert inter13 is None


def test_bin_config_rates_and_counts():
    config = BinConfig(bin_width_s=0.02)  # 20 ms bins -> 50 Hz
    assert config.bin_width_ms == pytest.approx(20.0)
    assert config.sample_rate_hz == pytest.approx(50.0)

    # 100 spikes/sec over 20 ms -> 2 expected spikes
    expected_count = config.rate_to_count(100.0)
    assert expected_count == pytest.approx(2.0)

    # 5 spikes in a 20 ms bin -> 250 spikes/sec
    rate_hz = config.count_to_rate(5.0)
    assert rate_hz == pytest.approx(250.0)

    # Array vectorization
    rates = np.array([0.0, 50.0, 100.0])
    counts = config.rate_to_count(rates)
    np.testing.assert_allclose(counts, np.array([0.0, 1.0, 2.0]))
    recovered_rates = config.count_to_rate(counts)
    np.testing.assert_allclose(recovered_rates, rates)


def test_bin_config_invalid():
    with pytest.raises(ValueError):
        BinConfig(bin_width_s=0.0)
    with pytest.raises(ValueError):
        BinConfig(bin_width_s=-0.01)


def test_temporal_grid_creation():
    interval = TimeInterval(start=0.0, stop=0.1)  # 100 ms
    config = BinConfig(bin_width_s=0.02)  # 20 ms bins -> 5 bins
    grid = TemporalGrid.from_interval(interval, config)

    assert grid.num_bins == 5
    assert len(grid.bin_edges) == 6
    np.testing.assert_allclose(grid.bin_edges, [0.0, 0.02, 0.04, 0.06, 0.08, 0.1])

    centers = grid.bin_centers
    assert len(centers) == 5
    np.testing.assert_allclose(centers, [0.01, 0.03, 0.05, 0.07, 0.09])


def test_temporal_grid_bin_indexing():
    interval = TimeInterval(start=1.0, stop=1.1)
    config = BinConfig(bin_width_s=0.02, align_to_zero=False)
    grid = TemporalGrid.from_interval(interval, config)

    assert grid.get_bin_index(1.000) == 0
    assert grid.get_bin_index(1.019) == 0
    assert grid.get_bin_index(1.020) == 1
    assert grid.get_bin_index(1.099) == 4
    assert grid.get_bin_index(0.999) is None
    assert grid.get_bin_index(1.100) is None


def test_temporal_grid_subgrid():
    interval = TimeInterval(start=0.0, stop=1.0)
    config = BinConfig(bin_width_s=0.1)
    grid = TemporalGrid.from_interval(interval, config)
    assert grid.num_bins == 10

    sub = grid.slice_subgrid(2, 5)
    assert sub.num_bins == 3
    assert sub.interval.start == pytest.approx(0.2)
    assert sub.interval.stop == pytest.approx(0.5)
    np.testing.assert_allclose(sub.bin_edges, [0.2, 0.3, 0.4, 0.5])
