"""Unit tests for local NLB test evaluation adapter and scoring bridge."""

import numpy as np
import pytest

from nocturne.recordings.mc_maze import MCMazeAdapter
from nocturne.recordings.nlb_eval import NLBEvaluationAdapter, NLBEvaluationTarget


def test_nlb_eval_adapter_extraction():
    adapter = MCMazeAdapter()
    recording = adapter.create_synthetic_maze_fixture(num_trials=30, num_neurons=20, seed=777)

    target = NLBEvaluationAdapter.extract_evaluation_target(recording, split_name="official")

    pop = recording.metadata.population
    assert target.dataset_name == "mc_maze"
    assert target.num_held_out_units == len(pop.held_out_indices)
    assert len(target.test_trial_ids) == 6  # 20% of 30 trials = 6 test trials
    assert target.total_test_spikes > 0

    # Test rate evaluation with perfect rates
    perfect_rates = target.test_counts / target.bin_width_s
    results_perfect = target.evaluate_predictions(perfect_rates, use_official_nlb_tools=False)

    assert results_perfect["bits_per_spike"] > 0.0
    assert results_perfect["total_test_spikes"] == target.total_test_spikes
    assert results_perfect["num_evaluated_trials"] == 6.0

    # Test rate evaluation with baseline mean rates -> BPS should be ~ 0.0
    baseline_expanded = np.broadcast_to(
        target.baseline_rates_hz[None, None, :],
        target.test_counts.shape,
    )
    results_baseline = target.evaluate_predictions(baseline_expanded, use_official_nlb_tools=False)
    assert results_baseline["bits_per_spike"] == pytest.approx(0.0, abs=1e-5)


def test_nlb_eval_shape_mismatch():
    adapter = MCMazeAdapter()
    recording = adapter.create_synthetic_maze_fixture(num_trials=20, num_neurons=10, seed=12)
    target = NLBEvaluationAdapter.extract_evaluation_target(recording, split_name="official")

    bad_shape = np.zeros((target.test_counts.shape[0], target.test_counts.shape[1], 999))
    with pytest.raises(ValueError, match="!= target shape"):
        target.evaluate_predictions(bad_shape, use_official_nlb_tools=False)
