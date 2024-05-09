"""Valid interval cutter, gap detector, and continuous timeline integrity verifier."""

from __future__ import annotations

from typing import List, Sequence, Tuple
from pydantic import BaseModel, ConfigDict, Field

from nocturne.contracts.time import TimeInterval


class RecordingGap(BaseModel):
    """Identified break or non-recording interval in an electrophysiological session."""

    model_config = ConfigDict(frozen=True)

    interval: TimeInterval = Field(..., description="Continuous bounds of the recording gap [start, stop)")
    reason: str = Field(default="pause", description="Diagnostic reason for gap (e.g. pause, dropped_packets)")

    @property
    def duration_s(self) -> float:
        return self.interval.duration


class GapDetector:
    """Detects gaps, hardware pauses, and dropped intervals across an experimental recording."""

    def __init__(self, min_gap_duration_s: float = 0.5):
        if min_gap_duration_s <= 0.0:
            raise ValueError(f"min_gap_duration_s must be positive, got {min_gap_duration_s}")
        self.min_gap_duration_s = min_gap_duration_s

    def detect_gaps_between_intervals(self, intervals: Sequence[TimeInterval]) -> List[RecordingGap]:
        """Identify intervals where recording was interrupted between successive valid epochs.

        Args:
            intervals: Sequence of TimeIntervals, assumed sorted by start time.

        Returns:
            List of detected RecordingGap objects.
        """
        if len(intervals) <= 1:
            return []

        sorted_intervals = sorted(intervals, key=lambda x: x.start)
        gaps: List[RecordingGap] = []

        for i in range(len(sorted_intervals) - 1):
            curr_stop = sorted_intervals[i].stop
            next_start = sorted_intervals[i + 1].start

            if next_start > curr_stop:
                gap_duration = next_start - curr_stop
                if gap_duration >= self.min_gap_duration_s:
                    gaps.append(
                        RecordingGap(
                            interval=TimeInterval(start=curr_stop, stop=next_start),
                            reason=f"inter_interval_gap_{gap_duration:.3f}s",
                        )
                    )

        return gaps


class IntervalCutter:
    """Slices valid recording periods into bounded duration windows without crossing gaps."""

    def __init__(
        self,
        window_duration_s: float,
        step_duration_s: float | None = None,
        min_window_duration_s: float | None = None,
    ):
        if window_duration_s <= 0.0:
            raise ValueError(f"window_duration_s must be positive, got {window_duration_s}")

        self.window_duration_s = window_duration_s
        self.step_duration_s = step_duration_s if step_duration_s is not None else window_duration_s
        if self.step_duration_s <= 0.0:
            raise ValueError(f"step_duration_s must be positive, got {self.step_duration_s}")

        self.min_window_duration_s = min_window_duration_s if min_window_duration_s is not None else window_duration_s

    def cut_valid_windows(self, valid_intervals: Sequence[TimeInterval]) -> List[TimeInterval]:
        """Partition valid epochs into sliding or non-overlapping windows of bounded duration.

        Guarantees that no window bridges a recording gap or exceeds valid boundaries.
        """
        windows: List[TimeInterval] = []

        for interval in sorted(valid_intervals, key=lambda x: x.start):
            duration = interval.duration
            if duration < self.min_window_duration_s:
                # Sub-threshold fragment discarded to preserve window contract
                continue

            current_start = interval.start
            while current_start + self.window_duration_s <= interval.stop + 1e-9:
                win_stop = min(current_start + self.window_duration_s, interval.stop)
                windows.append(TimeInterval(start=current_start, stop=win_stop))
                current_start += self.step_duration_s

        return windows
