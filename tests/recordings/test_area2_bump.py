"""Unit tests for Area2_Bump somatosensory perturbation adapter."""

import numpy as np
import pytest

from nocturne.recordings.area2_bump import Area2BumpAdapter


def test_area2_bump_adapter_fixture():
    adapter = Area2BumpAdapter()
    recording = adapter.create_synthetic_bump_fixture(
        num_trials=40,
        num_neurons=20,
        trial_duration_s=0.8,
        bin_width_s=0.02,
        seed=101,
    )

    assert recording.num_neurons == 20
    assert recording.trials is not None
    assert recording.trials.num_trials == 40

    # Verify Area 2 anatomical label
    for n in recording.metadata.population.neurons:
        assert n.brain_area == "Area2"

    # Verify bump_onset events
    t0 = recording.trials.get_by_id(0)
    assert "bump_onset" in t0.events
    assert t0.condition_id in [0, 1, 2, 3]

    # Verify force trajectory
    assert t0.behavior is not None
    assert "force_x" in t0.behavior.variable_names
    assert "force_y" in t0.behavior.variable_names

    # Verify counts
    assert np.issubdtype(recording.binned_counts.counts.dtype, np.integer)
    assert np.all(recording.binned_counts.counts >= 0)
    assert recording.binned_counts.total_spikes > 0

    # Verify zero-overlap split
    split = recording.get_split("official")
    split.verify_zero_overlap()
