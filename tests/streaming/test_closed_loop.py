"""Unit tests for closed-loop neurostimulation trigger simulator."""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from nocturne.streaming.closed_loop import (
    ClosedLoopStimulator,
    StimulationEvent,
)


def test_rate_threshold_crossing_and_isi_interlock() -> None:
    stim = ClosedLoopStimulator(
        rate_threshold_hz=30.0,
        min_inter_stim_interval_s=0.10,  # 100 ms minimum ISI
        blanking_duration_s=0.03,        # 30 ms blanking
    )

    z = np.array([0.5, -0.5])

    # 1. Below threshold -> No trigger
    low_rates = np.full(8, 15.0)
    ev0 = stim.evaluate_step(current_time_s=1.0, latent_state=z, predicted_rates_hz=low_rates)
    assert ev0 is None
    assert stim.is_blanking_active(1.0) is False

    # 2. Above threshold -> Trigger emitted
    high_rates = np.full(8, 35.0)
    ev1 = stim.evaluate_step(current_time_s=1.02, latent_state=z, predicted_rates_hz=high_rates)
    assert isinstance(ev1, StimulationEvent)
    assert ev1.trigger_type == "rate_threshold_crossing"
    assert ev1.latency_ms > 0.0

    # 3. Blanking active at 1.03 (1.02 + 0.01 < 1.02 + 0.03)
    assert stim.is_blanking_active(1.03) is True
    assert stim.is_blanking_active(1.06) is False

    # 4. Immediate repeat at 1.05 (< 1.02 + 0.10 ISI) -> Suppressed by ISI interlock
    ev2 = stim.evaluate_step(current_time_s=1.05, latent_state=z, predicted_rates_hz=high_rates)
    assert ev2 is None

    # 5. Repeat after ISI expired (at 1.15) -> Trigger allowed
    ev3 = stim.evaluate_step(current_time_s=1.15, latent_state=z, predicted_rates_hz=high_rates)
    assert isinstance(ev3, StimulationEvent)


def test_phase_attractor_crossing() -> None:
    target_phase = 0.0  # Positive x-axis (phase = 0)
    stim = ClosedLoopStimulator(
        rate_threshold_hz=100.0,  # High rate threshold so rate does not trigger
        phase_trigger_target=target_phase,
        min_inter_stim_interval_s=0.05,
    )

    # State at phase pi (negative x-axis: [-1, 0]) -> No trigger
    z_pi = np.array([-1.0, 0.0])
    ev_pi = stim.evaluate_step(current_time_s=0.1, latent_state=z_pi)
    assert ev_pi is None

    # State at phase 0 (positive x-axis: [1, 0]) -> Triggers phase target
    z_0 = np.array([1.0, 0.0])
    ev_0 = stim.evaluate_step(current_time_s=0.2, latent_state=z_0)
    assert isinstance(ev_0, StimulationEvent)
    assert ev_0.trigger_type == "phase_attractor_crossing"


def test_session_pulse_limit() -> None:
    stim = ClosedLoopStimulator(
        rate_threshold_hz=10.0,
        min_inter_stim_interval_s=0.001,
        max_pulses_per_session=2,
    )
    rates = np.full(4, 25.0)
    z = np.zeros(2)

    ev1 = stim.evaluate_step(0.01, z, rates)
    ev2 = stim.evaluate_step(0.02, z, rates)
    ev3 = stim.evaluate_step(0.03, z, rates)

    assert ev1 is not None
    assert ev2 is not None
    assert ev3 is None  # Hit session limit of 2 pulses

    summary = stim.get_summary()
    assert summary["total_pulses"] == 2
    assert summary["mean_latency_ms"] > 0.0
