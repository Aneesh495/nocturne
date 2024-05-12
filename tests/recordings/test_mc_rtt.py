"""Unit tests for MC_RTT continuous tracking adapter and window slicing."""

import numpy as np
import pytest

from nocturne.recordings.mc_rtt import MCRTTAdapter


def test_mc_rtt_adapter_fixture():
    adapter = MCRTTAdapter()
    recording = adapter.create_synthetic_rtt_fixture(
        duration_s=30.0,
        num_neurons=25,
        bin_width_s=0.02,
        window_duration_s=1.0,
        seed=42,
    )

    assert recording.num_neurons == 25
    assert recording.metadata.duration_s == pytest.approx(30.0)

    # 30 seconds sliced into 1.0 second windows -> 30 windows
    assert recording.trials is not None
    assert recording.trials.num_trials == 30

    # Verify counts are non-negative integers
    assert np.issubdtype(recording.binned_counts.counts.dtype, np.integer)
    assert np.all(recording.binned_counts.counts >= 0)

    # Verify chronological split
    chron_split = recording.get_split("chronological")
    chron_split.verify_zero_overlap()

    # Train: ~21 windows, Val: ~4 windows, Test: ~5 windows
    assert chron_split.train.num_trials + chron_split.val.num_trials + chron_split.test.num_trials == 30
    # Strict temporal ordering: train < val < test
    assert max(chron_split.train.trial_ids) < min(chron_split.val.trial_ids)
    assert max(chron_split.val.trial_ids) < min(chron_split.test.trial_ids)
