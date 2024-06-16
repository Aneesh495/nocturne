"""Closed-loop neurostimulation trigger simulator and safety interlock engine.

Evaluates responsive neuromodulation conditions on streaming latent states,
simulating hardware latency jitter, refractory blanking, and charge limits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch


@dataclass(frozen=True)
class StimulationEvent:
    """Record of a dispatched closed-loop stimulation pulse."""

    timestamp_s: float
    trigger_type: str
    detected_latent_state: List[float]
    latency_ms: float
    energy_units: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert event record to dictionary."""
        return asdict(self)


class ClosedLoopStimulator:
    """Simulates real-time closed-loop stimulation triggers with safety interlocks."""

    def __init__(
        self,
        rate_threshold_hz: float = 40.0,
        phase_trigger_target: Optional[float] = None,
        min_inter_stim_interval_s: float = 0.05,  # Max 20 Hz stimulation
        blanking_duration_s: float = 0.02,
        mean_hardware_latency_ms: float = 3.5,
        latency_jitter_std_ms: float = 0.8,
        max_pulses_per_session: int = 1000,
    ):
        self.rate_threshold_hz = rate_threshold_hz
        self.phase_trigger_target = phase_trigger_target
        self.min_isi_s = min_inter_stim_interval_s
        self.blanking_duration_s = blanking_duration_s
        self.mean_lat_ms = mean_hardware_latency_ms
        self.jitter_std_ms = latency_jitter_std_ms
        self.max_pulses = max_pulses_per_session

        self.last_stim_timestamp_s: float = -1e9
        self.event_history: List[StimulationEvent] = []
        self._rng = np.random.RandomState(42)

    def is_blanking_active(self, current_time_s: float) -> bool:
        """Check whether the sensor is currently inside an electrical artifact blanking window."""
        return (current_time_s - self.last_stim_timestamp_s) < self.blanking_duration_s

    def evaluate_step(
        self,
        current_time_s: float,
        latent_state: np.ndarray | torch.Tensor,
        predicted_rates_hz: Optional[np.ndarray | torch.Tensor] = None,
    ) -> Optional[StimulationEvent]:
        """Evaluate trigger condition and dispatch pulse if safety interlocks pass.

        Args:
            current_time_s: Continuous stream timestamp in seconds.
            latent_state: 1D latent state vector [d].
            predicted_rates_hz: Optional 1D predicted firing rate vector [N].

        Returns:
            StimulationEvent if a pulse was triggered, else None.
        """
        if len(self.event_history) >= self.max_pulses:
            return None  # Total session limit reached

        # Refractory inter-stimulus interval interlock
        if (current_time_s - self.last_stim_timestamp_s) < self.min_isi_s:
            return None

        if isinstance(latent_state, torch.Tensor):
            z = latent_state.detach().cpu().numpy().astype(np.float64)
        else:
            z = np.asarray(latent_state, dtype=np.float64)

        trigger_type: Optional[str] = None

        # 1. Check rate threshold crossing
        if predicted_rates_hz is not None:
            if isinstance(predicted_rates_hz, torch.Tensor):
                r = predicted_rates_hz.detach().cpu().numpy().astype(np.float64)
            else:
                r = np.asarray(predicted_rates_hz, dtype=np.float64)
            if np.max(r) >= self.rate_threshold_hz:
                trigger_type = "rate_threshold_crossing"

        # 2. Check phase target crossing in first 2 latent dimensions
        if trigger_type is None and self.phase_trigger_target is not None and len(z) >= 2:
            phase = math.atan2(z[1], z[0])  # [-pi, pi]
            diff = abs(phase - self.phase_trigger_target)
            diff = min(diff, 2 * math.pi - diff)
            if diff < 0.20:  # Within 0.20 rad tolerance
                trigger_type = "phase_attractor_crossing"

        if trigger_type is None:
            return None

        # Sample hardware latency with positive clipping
        lat_ms = max(0.5, float(self._rng.normal(self.mean_lat_ms, self.jitter_std_ms)))

        event = StimulationEvent(
            timestamp_s=current_time_s + (lat_ms / 1000.0),
            trigger_type=trigger_type,
            detected_latent_state=[float(val) for val in z],
            latency_ms=lat_ms,
            energy_units=1.0,
        )

        self.last_stim_timestamp_s = current_time_s
        self.event_history.append(event)
        return event

    def get_summary(self) -> Dict[str, Any]:
        """Return diagnostic metrics of closed-loop stimulation history."""
        latencies = [e.latency_ms for e in self.event_history]
        return {
            "total_pulses": len(self.event_history),
            "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
            "max_latency_ms": float(np.max(latencies)) if latencies else 0.0,
            "min_latency_ms": float(np.min(latencies)) if latencies else 0.0,
        }
