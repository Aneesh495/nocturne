"""Trial boundaries, event alignments, behavioral trajectories, and condition contracts."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from nocturne.contracts.time import TimeInterval


class TrialType(str, Enum):
    """Behavioral task paradigm or trial condition classification."""

    STANDARD = "standard"
    DELAYED_REACH = "delayed_reach"
    RANDOM_TRACKING = "random_tracking"
    PERTURBATION_BUMP = "perturbation_bump"
    ABORTED = "aborted"


class BehavioralTrajectory(BaseModel):
    """Continuous behavioral time-series (e.g. hand position, velocity, force)."""

    model_config = ConfigDict(frozen=True)

    variable_names: Tuple[str, ...] = Field(..., description="Names of kinematic/behavioral channels")
    timestamps: Tuple[float, ...] = Field(..., description="Sample timestamps in seconds")
    values: Tuple[Tuple[float, ...], ...] = Field(..., description="2D array [time_steps, channels]")

    def model_post_init(self, __context: object) -> None:
        T = len(self.timestamps)
        if len(self.values) != T:
            raise ValueError(f"Values length ({len(self.values)}) must match timestamps length ({T})")
        C = len(self.variable_names)
        for row in self.values:
            if len(row) != C:
                raise ValueError(f"Each values row length ({len(row)}) must match variable_names length ({C})")

    @property
    def num_timepoints(self) -> int:
        return len(self.timestamps)

    @property
    def num_channels(self) -> int:
        return len(self.variable_names)

    def as_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (timestamps, values) as NumPy ndarrays."""
        return np.array(self.timestamps, dtype=np.float64), np.array(self.values, dtype=np.float64)


class TrialMetadata(BaseModel):
    """Metadata, event timings, and condition labels for a single behavioral trial."""

    model_config = ConfigDict(frozen=True)

    trial_id: int = Field(..., ge=0, description="0-indexed trial number within session")
    session_id: str = Field(..., description="Parent session identifier")
    interval: TimeInterval = Field(..., description="Continuous bounds [start, stop) in session time")
    events: Dict[str, float] = Field(default_factory=dict, description="Event name to timestamp in seconds")
    trial_type: TrialType = Field(default=TrialType.STANDARD, description="Trial task classification")
    is_valid: bool = Field(default=True, description="Whether trial met task inclusion criteria without abort")
    condition_id: Optional[int] = Field(default=None, description="Categorical condition ID (e.g. target index)")
    behavior: Optional[BehavioralTrajectory] = Field(default=None, description="Kinematic trajectory for this trial")

    def model_post_init(self, __context: object) -> None:
        for ev_name, ev_time in self.events.items():
            if not self.interval.contains(ev_time) and ev_time != self.interval.stop:
                # Events should fall within trial boundaries
                pass

    def get_event_time(self, event_name: str) -> float:
        """Get timestamp of specified event, raising KeyError if absent."""
        if event_name not in self.events:
            raise KeyError(f"Event '{event_name}' not present in trial {self.trial_id} (events: {list(self.events.keys())})")
        return self.events[event_name]

    def align_window(self, event_name: str, pre_s: float, post_s: float) -> TimeInterval:
        """Extract a continuous time interval relative to an alignment event.

        Window: [event_time - pre_s, event_time + post_s)
        """
        if pre_s < 0.0 or post_s <= 0.0:
            raise ValueError(f"pre_s ({pre_s}) must be >= 0 and post_s ({post_s}) must be > 0")
        ev_t = self.get_event_time(event_name)
        return TimeInterval(start=ev_t - pre_s, stop=ev_t + post_s)


class TrialCollection(BaseModel):
    """Collection of trials comprising an experimental session."""

    model_config = ConfigDict(frozen=True)

    session_id: str = Field(..., description="Session identifier")
    trials: Tuple[TrialMetadata, ...] = Field(..., description="Tuple of trials")

    def model_post_init(self, __context: object) -> None:
        t_ids = set()
        for t in self.trials:
            if t.trial_id in t_ids:
                raise ValueError(f"Duplicate trial_id {t.trial_id} in session {self.session_id}")
            t_ids.add(t.trial_id)

    @property
    def num_trials(self) -> int:
        return len(self.trials)

    @property
    def valid_trials(self) -> List[TrialMetadata]:
        """Return list of valid trials that passed quality checks."""
        return [t for t in self.trials if t.is_valid]

    def get_by_id(self, trial_id: int) -> TrialMetadata:
        """Retrieve trial by trial_id."""
        for t in self.trials:
            if t.trial_id == trial_id:
                return t
        raise KeyError(f"Trial {trial_id} not found in session {self.session_id}")

    def filter_by_condition(self, condition_id: int) -> List[TrialMetadata]:
        """Filter trials matching a specific condition ID."""
        return [t for t in self.trials if t.condition_id == condition_id]
