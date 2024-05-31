"""Unit tests for scientific matrix aggregator and artifact exporter."""

from __future__ import annotations

import csv
from pathlib import Path
import pytest

from nocturne.campaign.ablations import (
    AblationConfig,
    AblationFamily,
    AblationResult,
)
from nocturne.campaign.benchmark import (
    BenchmarkRunConfig,
    BenchmarkRunResult,
)
from nocturne.campaign.matrix import (
    AggregatedMetric,
    BenchmarkGroupSummary,
    MatrixAggregator,
    SignificanceResult,
)


@pytest.fixture
def mock_benchmark_runs() -> list[BenchmarkRunResult]:
    runs = []
    # MC_Maze with 2 models across 3 seeds
    for seed in [42, 43, 44]:
        runs.append(
            BenchmarkRunResult(
                config=BenchmarkRunConfig(
                    dataset_name="mc_maze",
                    model_family="attention",
                    seed=seed,
                ),
                bps_held_in=0.35 + (seed - 42) * 0.01,
                bps_held_out=0.28 + (seed - 42) * 0.01,
                poisson_deviance=120.0 - (seed - 42) * 2.0,
                pseudo_r2=0.45 + (seed - 42) * 0.02,
                probe_r2=0.72 + (seed - 42) * 0.01,
                duration_s=5.2,
                model_sha256="abc123sha",
            )
        )
        runs.append(
            BenchmarkRunResult(
                config=BenchmarkRunConfig(
                    dataset_name="mc_maze",
                    model_family="mean_rate",
                    seed=seed,
                ),
                bps_held_in=0.0,
                bps_held_out=0.0,
                poisson_deviance=200.0,
                pseudo_r2=0.0,
                probe_r2=None,
                duration_s=0.1,
                model_sha256="meanratesha",
            )
        )
    return runs


@pytest.fixture
def mock_ablation_runs() -> list[AblationResult]:
    runs = []
    for seed in [42, 43]:
        runs.append(
            AblationResult(
                config=AblationConfig(
                    family=AblationFamily.OBJECTIVE,
                    variant_name="baseline_poisson",
                    dataset_name="synthetic",
                    seed=seed,
                ),
                bps_held_in=0.30,
                bps_held_out=0.25,
                poisson_deviance=100.0,
                pseudo_r2=0.50,
                duration_s=2.0,
            )
        )
        runs.append(
            AblationResult(
                config=AblationConfig(
                    family=AblationFamily.OBJECTIVE,
                    variant_name="variant_mse",
                    dataset_name="synthetic",
                    seed=seed,
                ),
                bps_held_in=0.10,
                bps_held_out=0.08,
                poisson_deviance=180.0,
                pseudo_r2=0.15,
                duration_s=2.0,
            )
        )
    return runs


def test_summarize_benchmarks(mock_benchmark_runs: list[BenchmarkRunResult]) -> None:
    aggregator = MatrixAggregator(benchmark_results=mock_benchmark_runs)
    summaries = aggregator.summarize_benchmarks()

    assert len(summaries) == 2
    att_summary = [s for s in summaries if s.model_family == "attention"][0]
    assert att_summary.dataset_name == "mc_maze"
    assert att_summary.bps_held_out is not None
    assert pytest.approx(att_summary.bps_held_out.mean) == 0.29
    assert att_summary.bps_held_out.n == 3
    assert pytest.approx(att_summary.poisson_deviance.mean) == 118.0


def test_summarize_ablations(mock_ablation_runs: list[AblationResult]) -> None:
    aggregator = MatrixAggregator(ablation_results=mock_ablation_runs)
    summaries = aggregator.summarize_ablations()

    assert len(summaries) == 2
    mse_summary = [s for s in summaries if s.variant_name == "variant_mse"][0]
    assert mse_summary.relative_deviance_delta > 0.0  # Deviance got worse by 80%
    assert pytest.approx(mse_summary.relative_deviance_delta) == 0.80


def test_markdown_and_latex_tables(
    mock_benchmark_runs: list[BenchmarkRunResult],
    mock_ablation_runs: list[AblationResult],
) -> None:
    aggregator = MatrixAggregator(
        benchmark_results=mock_benchmark_runs,
        ablation_results=mock_ablation_runs,
    )
    md_bench = aggregator.to_markdown_benchmark_table()
    assert "| Dataset | Model Family |" in md_bench
    assert "mc_maze" in md_bench
    assert "attention" in md_bench

    tex_bench = aggregator.to_latex_benchmark_table()
    assert r"\begin{tabular}" in tex_bench
    assert r"\toprule" in tex_bench
    assert r"\bottomrule" in tex_bench

    md_abl = aggregator.to_markdown_ablation_table()
    assert "| Family | Variant | Dataset |" in md_abl
    assert "variant_mse" in md_abl

    tex_abl = aggregator.to_latex_ablation_table()
    assert r"\begin{table}" in tex_abl
    assert r"\caption" in tex_abl


def test_significance_test(mock_benchmark_runs: list[BenchmarkRunResult]) -> None:
    aggregator = MatrixAggregator(benchmark_results=mock_benchmark_runs)
    res = aggregator.test_significance(
        model_a="attention",
        model_b="mean_rate",
        dataset_name="mc_maze",
        metric="poisson_deviance",
    )

    assert isinstance(res, SignificanceResult)
    assert res.n_samples == 3
    assert res.mean_difference < 0.0  # Attention deviance is lower
    assert res.p_value < 0.05  # Strongly significant


def test_export_csv_and_artifacts(
    mock_benchmark_runs: list[BenchmarkRunResult],
    mock_ablation_runs: list[AblationResult],
    tmp_path: Path,
) -> None:
    aggregator = MatrixAggregator(
        benchmark_results=mock_benchmark_runs,
        ablation_results=mock_ablation_runs,
    )
    csv_file = aggregator.export_csv(tmp_path / "benchmarks.csv", kind="benchmark")
    assert csv_file.exists()

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 6
        assert rows[0]["model_family"] in ["attention", "mean_rate"]

    artifacts = aggregator.export_artifacts(tmp_path / "artifacts")
    assert len(artifacts) == 3
    for art in artifacts:
        assert Path(art.file_path).exists()
        assert art.sha256_hash is not None
