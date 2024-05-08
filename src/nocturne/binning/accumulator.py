"""Spike count accumulator converting continuous action potential timestamps to discrete non-negative integer counts."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from nocturne.contracts.neurons import PopulationMetadata
from nocturne.contracts.time import BinConfig, TemporalGrid, TimeInterval


class BinnedSpikeCounts(BaseModel):
    """Container holding discrete non-negative integer spike counts on a temporal grid."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    grid: TemporalGrid = Field(..., description="Temporal grid defining bin edges and intervals")
    population: PopulationMetadata = Field(..., description="Recorded neural population metadata")
    counts: np.ndarray = Field(..., description="Non-negative integer array of shape [num_bins, num_neurons]")

    def model_post_init(self, __context: object) -> None:
        T = self.grid.num_bins
        N = self.population.num_neurons

        if self.counts.shape != (T, N):
            raise ValueError(f"counts shape must be ({T}, {N}), got {self.counts.shape}")

        if not np.issubdtype(self.counts.dtype, np.integer):
            # Check if float values are all integer-valued
            if not np.all(np.equal(np.mod(self.counts, 1), 0)):
                raise ValueError("Spike counts must be exact non-negative integers")
            # Cast to int64
            object.__setattr__(self, "counts", self.counts.astype(np.int64))

        if np.any(self.counts < 0):
            raise ValueError("Spike counts cannot be negative")

    @property
    def num_bins(self) -> int:
        return self.grid.num_bins

    @property
    def num_neurons(self) -> int:
        return self.population.num_neurons

    @property
    def total_spikes(self) -> int:
        """Total number of action potentials recorded across all neurons and bins."""
        return int(np.sum(self.counts))

    def get_neuron_counts(self, channel_index: int) -> np.ndarray:
        """Return 1D array of counts for a specific neuron across all bins."""
        if channel_index < 0 or channel_index >= self.num_neurons:
            raise IndexError(f"Channel index {channel_index} out of range [0, {self.num_neurons})")
        return self.counts[:, channel_index]

    def compute_mean_rates_hz(self) -> np.ndarray:
        """Compute empirical mean firing rate in spikes/second (Hz) for each neuron."""
        total_time_s = self.grid.interval.duration
        if total_time_s <= 0:
            raise ValueError("Interval duration must be positive to compute firing rates")
        return np.sum(self.counts, axis=0) / total_time_s

    def slice_time(self, start_bin: int, stop_bin: int) -> BinnedSpikeCounts:
        """Extract a sub-matrix of counts covering bins [start_bin, stop_bin)."""
        sub_grid = self.grid.slice_subgrid(start_bin, stop_bin)
        sub_counts = self.counts[start_bin:stop_bin, :]
        return BinnedSpikeCounts(
            grid=sub_grid,
            population=self.population,
            counts=sub_counts,
        )


class SpikeAccumulator:
    """Accumulates continuous spike timestamps into discrete non-negative integer bins."""

    def __init__(self, bin_config: BinConfig):
        self.bin_config = bin_config

    def bin_timestamps(
        self,
        spike_times_by_channel: Sequence[Sequence[float]] | Dict[int, Sequence[float]],
        interval: TimeInterval,
        population: PopulationMetadata,
    ) -> BinnedSpikeCounts:
        """Discretize continuous spike timestamp series into integer counts.

        Args:
            spike_times_by_channel: Map or sequence of spike timestamp arrays per channel index.
            interval: Continuous time interval to bin.
            population: Target population metadata.

        Returns:
            BinnedSpikeCounts container with exact integer matrix.
        """
        grid = TemporalGrid.from_interval(interval, self.bin_config)
        T = grid.num_bins
        N = population.num_neurons
        bin_edges = np.array(grid.bin_edges, dtype=np.float64)

        counts = np.zeros((T, N), dtype=np.int64)

        for channel_idx in range(N):
            if isinstance(spike_times_by_channel, dict):
                st = spike_times_by_channel.get(channel_idx, [])
            else:
                st = spike_times_by_channel[channel_idx] if channel_idx < len(spike_times_by_channel) else []

            if len(st) == 0:
                continue

            st_arr = np.asarray(st, dtype=np.float64)
            # Filter to [interval.start, interval.stop)
            valid_mask = (st_arr >= bin_edges[0]) & (st_arr < bin_edges[-1])
            st_valid = st_arr[valid_mask]

            if len(st_valid) > 0:
                # np.histogram with bins=bin_edges: [edge_i, edge_{i+1})
                hist, _ = np.histogram(st_valid, bins=bin_edges)
                counts[:, channel_idx] = hist.astype(np.int64)

        return BinnedSpikeCounts(
            grid=grid,
            population=population,
            counts=counts,
        )
