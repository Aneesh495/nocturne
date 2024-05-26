"""Tests for 5-family ablation matrix execution."""

from __future__ import annotations

import pytest
from nocturne.campaign.ablations import (
    AblationConfig,
    AblationFamily,
    AblationResult,
    AblationRunner,
)


def test_family_1_objective_ablation():
    results = AblationRunner.run_objective_ablation(
        dataset_name="mc_maze",
        seeds=(42,),
        num_steps=5,
    )
    assert len(results) == 3
    for r in results:
        assert isinstance(r, AblationResult)
        assert r.config.family == AblationFamily.OBJECTIVE
        assert r.poisson_deviance > 0.0


def test_family_2_masking_ablation():
    results = AblationRunner.run_masking_ablation(
        dataset_name="mc_maze",
        seeds=(42,),
        num_steps=5,
    )
    assert len(results) == 3
    for r in results:
        assert isinstance(r, AblationResult)
        assert r.config.family == AblationFamily.INPUT_MASKING


def test_family_3_capacity_ablation():
    results = AblationRunner.run_capacity_ablation(
        dataset_name="mc_maze",
        seeds=(42,),
        num_steps=5,
    )
    assert len(results) == 3
    for r in results:
        assert isinstance(r, AblationResult)
        assert r.config.family == AblationFamily.DEPTH_CAPACITY


def test_family_4_causality_ablation():
    results = AblationRunner.run_causality_ablation(
        dataset_name="mc_maze",
        seeds=(42,),
        num_steps=5,
    )
    assert len(results) == 2
    for r in results:
        assert isinstance(r, AblationResult)
        assert r.config.family == AblationFamily.TEMPORAL_CAUSALITY


def test_family_5_dynamics_ablation():
    results = AblationRunner.run_dynamics_ablation(
        dataset_name="linear_circuit",
        seeds=(42,),
        num_steps=5,
    )
    assert len(results) == 2
    for r in results:
        assert isinstance(r, AblationResult)
        assert r.config.family == AblationFamily.DYNAMICS_DRIVE
        assert r.poisson_deviance > 0.0
