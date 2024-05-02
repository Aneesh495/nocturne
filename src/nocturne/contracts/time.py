"""Temporal specifications, bin configurations, and continuous-to-discrete time contracts."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


class TimeInterval(BaseModel):
    """Continuous half-open time interval [start, stop) in seconds."""

    model_config = ConfigDict(frozen=True)

    start: float = Field(..., description="Start timestamp in seconds (inclusive)")
    stop: float = Field(..., description="Stop timestamp in seconds (exclusive)")

    @field_validator("start", "stop")
    @classmethod
    def validate_finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError(f"Interval bounds must be finite real numbers, got {v}")
        return v

    def model_post_init(self, __context: object) -> None:
        if self.stop < self.start:
            raise ValueError(f"stop ({self.stop}) must be greater than or equal to start ({self.start})")

    @property
    def duration(self) -> float:
        """Duration of the interval in seconds."""
        return self.stop - self.start

    def contains(self, t: float) -> bool:
        """Check whether timestamp t falls within [start, stop)."""
        return self.start <= t < self.stop

    def overlaps(self, other: TimeInterval) -> bool:
        """Check whether this interval overlaps with another interval."""
        return max(self.start, other.start) < min(self.stop, other.stop)

    def intersection(self, other: TimeInterval) -> TimeInterval | None:
        """Compute the intersection of two intervals, or None if disjoint."""
        s = max(self.start, other.start)
        e = min(self.stop, other.stop)
        if s < e:
            return TimeInterval(start=s, stop=e)
        return None


class BinConfig(BaseModel):
    """Discrete binning specification defining bin width and temporal discretization."""

    model_config = ConfigDict(frozen=True)

    bin_width_s: float = Field(..., gt=0.0, description="Bin width in seconds (must be strictly positive)")
    align_to_zero: bool = Field(default=True, description="Whether bin edges align with integer multiples of bin_width_s")

    @property
    def bin_width_ms(self) -> float:
        """Bin width in milliseconds."""
        return self.bin_width_s * 1000.0

    @property
    def sample_rate_hz(self) -> float:
        """Equivalent sampling frequency in Hertz."""
        return 1.0 / self.bin_width_s

    def rate_to_count(self, rate_hz: float | np.ndarray) -> float | np.ndarray:
        """Convert rate in spikes/second to expected spike count per bin.

        Formula: lambda = rate * Delta_t
        """
        return rate_hz * self.bin_width_s

    def count_to_rate(self, count: float | np.ndarray) -> float | np.ndarray:
        """Convert expected spike count per bin to rate in spikes/second.

        Formula: rate = count / Delta_t
        """
        return count / self.bin_width_s


class TemporalGrid(BaseModel):
    """Discrete temporal grid of bin edges and bin centers for a continuous recording interval."""

    model_config = ConfigDict(frozen=True)

    interval: TimeInterval = Field(..., description="Underlying continuous time interval")
    bin_config: BinConfig = Field(..., description="Binning configuration")
    num_bins: int = Field(..., ge=1, description="Number of discrete temporal bins")
    bin_edges: Tuple[float, ...] = Field(..., description="Monotonically increasing bin edge timestamps")

    @classmethod
    def from_interval(cls, interval: TimeInterval, bin_config: BinConfig) -> TemporalGrid:
        """Construct a regular temporal grid spanning the specified interval."""
        if interval.duration <= 0:
            raise ValueError("Interval duration must be strictly positive to construct a TemporalGrid")

        start = interval.start
        stop = interval.stop
        bin_width = bin_config.bin_width_s

        epsilon = 1e-12
        if bin_config.align_to_zero:
            first_edge = math.floor((start + epsilon) / bin_width) * bin_width
            raw_bins = (stop - first_edge) / bin_width
            num_bins = int(math.ceil(raw_bins - epsilon))
        else:
            first_edge = start
            raw_bins = (stop - start) / bin_width
            num_bins = int(math.ceil(raw_bins - epsilon))

        num_bins = max(1, num_bins)
        edges = [round(first_edge + i * bin_width, 12) for i in range(num_bins + 1)]
        return cls(
            interval=interval,
            bin_config=bin_config,
            num_bins=num_bins,
            bin_edges=tuple(edges)
        )

    @property
    def bin_centers(self) -> np.ndarray:
        """Return NumPy array of bin center timestamps."""
        edges = np.array(self.bin_edges, dtype=np.float64)
        return 0.5 * (edges[:-1] + edges[1:])

    def get_bin_index(self, timestamp: float) -> int | None:
        """Return the discrete bin index containing the timestamp, or None if outside the grid."""
        if timestamp < self.bin_edges[0] or timestamp >= self.bin_edges[-1]:
            return None
        idx = int(math.floor((timestamp - self.bin_edges[0]) / self.bin_config.bin_width_s))
        if 0 <= idx < self.num_bins:
            return idx
        return None

    def slice_subgrid(self, start_idx: int, stop_idx: int) -> TemporalGrid:
        """Extract a subgrid covering bins [start_idx, stop_idx)."""
        if start_idx < 0 or stop_idx > self.num_bins or start_idx >= stop_idx:
            raise IndexError(f"Invalid subgrid bounds [{start_idx}, {stop_idx}) for grid of size {self.num_bins}")
        sub_edges = self.bin_edges[start_idx : stop_idx + 1]
        sub_interval = TimeInterval(start=sub_edges[0], stop=sub_edges[-1])
        return TemporalGrid(
            interval=sub_interval,
            bin_config=self.bin_config,
            num_bins=stop_idx - start_idx,
            bin_edges=sub_edges
        )
