"""Unit tests for out-of-distribution cross-condition generalization."""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from nocturne.evaluation.cross_condition import (
    CrossConditionEvaluator,
    CrossConditionResult,
)


def test_four_target_reach_loco_evaluation() -> None:
    np.random.seed(42)
    conditions = ["target_0", "target_90", "target_180", "target_270"]
    angles = [0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi]

    latents_by_cond = {}
    behavior_by_cond = {}

    B_c, T, d, K = 8, 25, 4, 2
    # Linear projection matrix from latents to velocity
    W = np.random.randn(K, d) * 0.5

    for c, ang in zip(conditions, angles):
        # Latents modulated by target angle direction
        base = np.array([math.cos(ang), math.sin(ang), 0.5, -0.5])
        lats = np.tile(base, (B_c, T, 1)) + 0.1 * np.random.randn(B_c, T, d)
        # Kinematics linearly generated from latents
        beh = (lats @ W.T) + 0.05 * np.random.randn(B_c, T, K)

        latents_by_cond[c] = lats
        behavior_by_cond[c] = beh

    evaluator = CrossConditionEvaluator(ridge_alpha=1.0)
    res = evaluator.evaluate_loco(latents_by_cond, behavior_by_cond)

    assert isinstance(res, CrossConditionResult)
    assert res.num_conditions == 4
    assert len(res.condition_records) == 4
    assert res.in_distribution_mean_r2 > 0.60
    assert res.out_of_distribution_mean_r2 > 0.40
    assert res.generalization_gap >= 0.0
    assert isinstance(res.is_robust_generalizer, bool)

    as_dict = res.to_dict()
    assert as_dict["num_conditions"] == 4
    assert "summary" in as_dict


def test_pytorch_tensor_conditions() -> None:
    evaluator = CrossConditionEvaluator()
    latents = {
        "cond_a": torch.randn(6, 10, 3),
        "cond_b": torch.randn(6, 10, 3),
    }
    behavior = {
        "cond_a": torch.randn(6, 10, 2),
        "cond_b": torch.randn(6, 10, 2),
    }
    res = evaluator.evaluate_loco(latents, behavior)
    assert res.num_conditions == 2


def test_insufficient_conditions_raises() -> None:
    evaluator = CrossConditionEvaluator()
    single_cond_lat = {"only_one": np.ones((4, 10, 2))}
    single_cond_beh = {"only_one": np.ones((4, 10, 1))}
    with pytest.raises(ValueError, match="at least 2 distinct conditions"):
        evaluator.evaluate_loco(single_cond_lat, single_cond_beh)
