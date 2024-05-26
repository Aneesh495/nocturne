"""Tests for BenchmarkSuiteRunner across models, datasets, and seeds."""

from __future__ import annotations

import pytest
from nocturne.campaign.benchmark import (
    BenchmarkRunConfig,
    BenchmarkRunResult,
    BenchmarkSuiteRunner,
)


def test_load_dataset_shapes():
    tr_counts, te_counts, held_in, held_out, kin = BenchmarkSuiteRunner.load_dataset("mc_maze", seed=42)
    assert tr_counts.ndim == 3
    assert te_counts.ndim == 3
    assert len(held_in) > 0
    assert len(held_out) > 0
    assert set(held_in).isdisjoint(set(held_out))


def test_benchmark_run_baselines():
    # Mean Rate
    cfg_mean = BenchmarkRunConfig(
        dataset_name="mc_maze",
        model_family="mean_rate",
        seed=42,
        num_steps=5,
    )
    res_mean = BenchmarkSuiteRunner.run_single(cfg_mean)
    assert isinstance(res_mean, BenchmarkRunResult)
    assert res_mean.poisson_deviance > 0.0

    # Poisson Factor
    cfg_pf = BenchmarkRunConfig(
        dataset_name="mc_maze",
        model_family="poisson_factor",
        seed=42,
        num_steps=5,
    )
    res_pf = BenchmarkSuiteRunner.run_single(cfg_pf)
    assert isinstance(res_pf, BenchmarkRunResult)
    assert res_pf.poisson_deviance > 0.0


def test_benchmark_run_attention():
    cfg_att = BenchmarkRunConfig(
        dataset_name="linear_circuit",
        model_family="attention",
        seed=42,
        num_steps=5,
    )
    res_att = BenchmarkSuiteRunner.run_single(cfg_att)
    assert isinstance(res_att, BenchmarkRunResult)
    assert res_att.poisson_deviance > 0.0
    assert res_att.duration_s > 0.0


def test_benchmark_multi_seed():
    results = BenchmarkSuiteRunner.run_multi_seed(
        dataset_name="linear_circuit",
        model_family="mean_rate",
        seeds=(42, 43, 44),
        num_steps=5,
    )
    assert len(results) == 3
    for r in results:
        assert isinstance(r, BenchmarkRunResult)
        assert r.config.dataset_name == "linear_circuit"
