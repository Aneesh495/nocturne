"""Unit tests for MC_Maze dataset adapter and delayed reach trial alignment."""

import numpy as np
import pytest

from nocturne.contracts.neurons import NeuronRole
from nocturne.recordings.mc_maze import MCMazeAdapter


def test_mc_maze_adapter_fixture():
    adapter = MCMazeAdapter()
    recording = adapter.create_synthetic_maze_fixture(
        num_trials=50,
        num_neurons=30,
        trial_duration_s=1.0,
        bin_width_s=0.02,
        seed=123,
    )

    assert recording.num_neurons == 30
    assert recording.trials is not None
    assert recording.trials.num_trials == 50

    # Verify held-in / held-out distribution (approx 70% / 30%)
    pop = recording.metadata.population
    assert len(pop.held_in_indices) == 21
    assert len(pop.held_out_indices) == 9

    # Verify counts are non-negative integers
    assert np.issubdtype(recording.binned_counts.counts.dtype, np.integer)
    assert np.all(recording.binned_counts.counts >= 0)
    assert recording.binned_counts.total_spikes > 0

    # Verify behavioral trajectories
    t0 = recording.trials.get_by_id(0)
    assert t0.behavior is not None
    assert "vel_x" in t0.behavior.variable_names
    assert "vel_y" in t0.behavior.variable_names

    # Verify trial alignment to move_onset
    win = t0.align_window("move_onset", pre_s=0.25, post_s=0.45)
    assert win.duration == pytest.approx(0.70)

    # Verify official split disjointness
    official_split = recording.get_split("official")
    official_split.verify_zero_overlap()
    assert official_split.train.num_trials == 30
    assert official_split.val.num_trials == 10
    assert official_split.test.num_trials == 10
