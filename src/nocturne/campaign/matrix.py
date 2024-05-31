"""Scientific matrix aggregator, publication table generator, and artifact exporter.

Compiles empirical evidence across multi-seed benchmarks and 5-family ablation matrices
into Markdown, publication-grade LaTeX tables, CSV files, and cryptographic artifacts.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats

from nocturne.artifacts.schema import ArtifactMetadata, ArtifactRegistry
from nocturne.campaign.ablations import AblationResult
from nocturne.campaign.benchmark import BenchmarkRunResult


@dataclass(frozen=True)
class AggregatedMetric:
    """Mean and sample dispersion statistics for an experimental metric."""

    mean: float
    std: float
    median: float
    n: int

    def format_mean_std(self, precision: int = 4) -> str:
        """Format as string: mean ± std."""
        return f"{self.mean:.{precision}f} +/- {self.std:.{precision}f}"


@dataclass(frozen=True)
class SignificanceResult:
    """Outcome of paired statistical hypothesis testing across seeds."""

    metric_name: str
    model_a: str
    model_b: str
    n_samples: int
    mean_a: float
    mean_b: float
    mean_difference: float
    t_statistic: float
    p_value: float
    is_significant: bool  # True if p_value < 0.05


@dataclass(frozen=True)
class BenchmarkGroupSummary:
    """Multi-seed summary for a specific dataset and model architecture."""

    dataset_name: str
    model_family: str
    bps_held_in: Optional[AggregatedMetric]
    bps_held_out: Optional[AggregatedMetric]
    poisson_deviance: AggregatedMetric
    pseudo_r2: AggregatedMetric
    probe_r2: Optional[AggregatedMetric]
    duration_s: AggregatedMetric


@dataclass(frozen=True)
class AblationGroupSummary:
    """Multi-seed summary for an ablation family variant."""

    family: str
    variant_name: str
    dataset_name: str
    bps_held_in: Optional[AggregatedMetric]
    bps_held_out: Optional[AggregatedMetric]
    poisson_deviance: AggregatedMetric
    pseudo_r2: AggregatedMetric
    relative_deviance_delta: float


def _compute_agg(values: List[Optional[float]]) -> Optional[AggregatedMetric]:
    valid = [v for v in values if v is not None and not np.isnan(v)]
    if not valid:
        return None
    arr = np.array(valid, dtype=np.float64)
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return AggregatedMetric(
        mean=float(np.mean(arr)),
        std=std,
        median=float(np.median(arr)),
        n=len(arr),
    )


class MatrixAggregator:
    """Aggregates multi-seed benchmark and ablation results into scientific tables."""

    def __init__(
        self,
        benchmark_results: Optional[List[BenchmarkRunResult]] = None,
        ablation_results: Optional[List[AblationResult]] = None,
    ):
        self.benchmark_results: List[BenchmarkRunResult] = list(benchmark_results or [])
        self.ablation_results: List[AblationResult] = list(ablation_results or [])

    def add_benchmark_results(self, results: List[BenchmarkRunResult]) -> None:
        """Append benchmark runs to aggregator."""
        self.benchmark_results.extend(results)

    def add_ablation_results(self, results: List[AblationResult]) -> None:
        """Append ablation runs to aggregator."""
        self.ablation_results.extend(results)

    def summarize_benchmarks(self) -> List[BenchmarkGroupSummary]:
        """Group benchmark results by dataset and model family across seeds."""
        groups: Dict[Tuple[str, str], List[BenchmarkRunResult]] = {}
        for r in self.benchmark_results:
            key = (r.config.dataset_name, r.config.model_family)
            groups.setdefault(key, []).append(r)

        summaries: List[BenchmarkGroupSummary] = []
        for (dataset, model), runs in sorted(groups.items()):
            bps_in = _compute_agg([r.bps_held_in for r in runs])
            bps_out = _compute_agg([r.bps_held_out for r in runs])
            dev = _compute_agg([r.poisson_deviance for r in runs])
            r2 = _compute_agg([r.pseudo_r2 for r in runs])
            probe = _compute_agg([r.probe_r2 for r in runs])
            dur = _compute_agg([r.duration_s for r in runs])

            assert dev is not None and r2 is not None and dur is not None

            summaries.append(
                BenchmarkGroupSummary(
                    dataset_name=dataset,
                    model_family=model,
                    bps_held_in=bps_in,
                    bps_held_out=bps_out,
                    poisson_deviance=dev,
                    pseudo_r2=r2,
                    probe_r2=probe,
                    duration_s=dur,
                )
            )
        return summaries

    def summarize_ablations(self) -> List[AblationGroupSummary]:
        """Group ablation results by family and variant across seeds."""
        groups: Dict[Tuple[str, str, str], List[AblationResult]] = {}
        for r in self.ablation_results:
            key = (r.config.family.value, r.config.variant_name, r.config.dataset_name)
            groups.setdefault(key, []).append(r)

        # Find baseline deviance per family for delta computation
        family_baselines: Dict[Tuple[str, str], float] = {}
        for (fam, var, dset), runs in groups.items():
            if "baseline" in var.lower() or "standard" in var.lower():
                dev_agg = _compute_agg([r.poisson_deviance for r in runs])
                if dev_agg:
                    family_baselines[(fam, dset)] = dev_agg.mean

        summaries: List[AblationGroupSummary] = []
        for (fam, var, dset), runs in sorted(groups.items()):
            bps_in = _compute_agg([r.bps_held_in for r in runs])
            bps_out = _compute_agg([r.bps_held_out for r in runs])
            dev = _compute_agg([r.poisson_deviance for r in runs])
            r2 = _compute_agg([r.pseudo_r2 for r in runs])
            assert dev is not None and r2 is not None

            base_dev = family_baselines.get((fam, dset), dev.mean)
            rel_delta = (dev.mean - base_dev) / (abs(base_dev) + 1e-8)

            summaries.append(
                AblationGroupSummary(
                    family=fam,
                    variant_name=var,
                    dataset_name=dset,
                    bps_held_in=bps_in,
                    bps_held_out=bps_out,
                    poisson_deviance=dev,
                    pseudo_r2=r2,
                    relative_deviance_delta=rel_delta,
                )
            )
        return summaries

    def to_markdown_benchmark_table(self) -> str:
        """Produce formatted Markdown table summarizing benchmark comparisons."""
        summaries = self.summarize_benchmarks()
        lines = [
            "| Dataset | Model Family | Held-In BPS | Held-Out BPS | Poisson Deviance | Pseudo-R2 | Kinematic R2 | Runtime (s) |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for s in summaries:
            b_in = s.bps_held_in.format_mean_std() if s.bps_held_in else "-"
            b_out = s.bps_held_out.format_mean_std() if s.bps_held_out else "-"
            dev = s.poisson_deviance.format_mean_std(precision=2)
            r2 = s.pseudo_r2.format_mean_std(precision=4)
            probe = s.probe_r2.format_mean_std(precision=4) if s.probe_r2 else "-"
            dur = f"{s.duration_s.mean:.2f}s"

            row = f"| {s.dataset_name} | {s.model_family} | {b_in} | {b_out} | {dev} | {r2} | {probe} | {dur} |"
            lines.append(row)
        return "\n".join(lines)

    def to_latex_benchmark_table(self) -> str:
        """Produce publication-quality LaTeX table for multi-seed benchmarks."""
        summaries = self.summarize_benchmarks()
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{llcccccc}",
            r"\toprule",
            r"Dataset & Model & Held-In BPS $\uparrow$ & Held-Out BPS $\uparrow$ & Deviance $\downarrow$ & Pseudo-$R^2 \uparrow$ & Kinematic $R^2 \uparrow$ & Time (s) \\",
            r"\midrule",
        ]

        curr_dataset = ""
        for s in summaries:
            d_label = s.dataset_name if s.dataset_name != curr_dataset else ""
            curr_dataset = s.dataset_name

            b_in = f"${s.bps_held_in.mean:.3f} \\pm {s.bps_held_in.std:.3f}$" if s.bps_held_in else "-"
            b_out = f"${s.bps_held_out.mean:.3f} \\pm {s.bps_held_out.std:.3f}$" if s.bps_held_out else "-"
            dev = f"${s.poisson_deviance.mean:.1f} \\pm {s.poisson_deviance.std:.1f}$"
            r2 = f"${s.pseudo_r2.mean:.3f} \\pm {s.pseudo_r2.std:.3f}$"
            probe = f"${s.probe_r2.mean:.3f} \\pm {s.probe_r2.std:.3f}$" if s.probe_r2 else "-"
            dur = f"{s.duration_s.mean:.1f}"

            row = f"{d_label} & {s.model_family} & {b_in} & {b_out} & {dev} & {r2} & {probe} & {dur} \\\\"
            lines.append(row)

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Neural population dynamics benchmark results across multiple seeds (mean $\pm$ std).}",
            r"\label{tab:nocturne_benchmark}",
            r"\end{table}",
        ])
        return "\n".join(lines)

    def to_markdown_ablation_table(self) -> str:
        """Produce formatted Markdown table summarizing ablation matrix."""
        summaries = self.summarize_ablations()
        lines = [
            "| Family | Variant | Dataset | Held-Out BPS | Deviance | Pseudo-R2 | Rel. Dev. Delta |",
            "| :--- | :--- | :--- | :---: | :---: | :---: | :---: |",
        ]
        for s in summaries:
            b_out = s.bps_held_out.format_mean_std() if s.bps_held_out else "-"
            dev = s.poisson_deviance.format_mean_std(precision=2)
            r2 = s.pseudo_r2.format_mean_std(precision=4)
            delta = f"{s.relative_deviance_delta * 100.0:+.1f}%"
            row = f"| {s.family} | {s.variant_name} | {s.dataset_name} | {b_out} | {dev} | {r2} | {delta} |"
            lines.append(row)
        return "\n".join(lines)

    def to_latex_ablation_table(self) -> str:
        """Produce publication-quality LaTeX table for ablation studies."""
        summaries = self.summarize_ablations()
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{lllcccc}",
            r"\toprule",
            r"Family & Variant & Dataset & Held-Out BPS $\uparrow$ & Deviance $\downarrow$ & Pseudo-$R^2 \uparrow$ & $\Delta$ Dev (\%) \\",
            r"\midrule",
        ]

        curr_fam = ""
        for s in summaries:
            fam_label = s.family if s.family != curr_fam else ""
            curr_fam = s.family

            b_out = f"${s.bps_held_out.mean:.3f} \\pm {s.bps_held_out.std:.3f}$" if s.bps_held_out else "-"
            dev = f"${s.poisson_deviance.mean:.1f} \\pm {s.poisson_deviance.std:.1f}$"
            r2 = f"${s.pseudo_r2.mean:.3f} \\pm {s.pseudo_r2.std:.3f}$"
            delta = f"{s.relative_deviance_delta * 100.0:+.1f}\\%"

            row = f"{fam_label} & {s.variant_name} & {s.dataset_name} & {b_out} & {dev} & {r2} & {delta} \\\\"
            lines.append(row)

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Ablation matrix results evaluating objective, masking, capacity, causality, and dynamics.}",
            r"\label{tab:nocturne_ablations}",
            r"\end{table}",
        ])
        return "\n".join(lines)

    def test_significance(
        self,
        model_a: str,
        model_b: str,
        dataset_name: str,
        metric: str = "poisson_deviance",
    ) -> SignificanceResult:
        """Perform paired t-test comparing two models across matched seeds on a dataset."""
        runs_a = [
            r for r in self.benchmark_results
            if r.config.model_family == model_a and r.config.dataset_name == dataset_name
        ]
        runs_b = [
            r for r in self.benchmark_results
            if r.config.model_family == model_b and r.config.dataset_name == dataset_name
        ]

        seeds_a = {r.config.seed: getattr(r, metric) for r in runs_a if getattr(r, metric) is not None}
        seeds_b = {r.config.seed: getattr(r, metric) for r in runs_b if getattr(r, metric) is not None}

        common_seeds = sorted(set(seeds_a.keys()) & set(seeds_b.keys()))
        if len(common_seeds) < 2:
            raise ValueError(
                f"Need at least 2 common seeds for paired testing, found {len(common_seeds)}."
            )

        vals_a = [seeds_a[s] for s in common_seeds]
        vals_b = [seeds_b[s] for s in common_seeds]

        arr_a = np.array(vals_a, dtype=np.float64)
        arr_b = np.array(vals_b, dtype=np.float64)
        diff = arr_a - arr_b

        if np.allclose(diff, 0.0):
            t_stat = 0.0
            p_val = 1.0
        else:
            t_stat, p_val = stats.ttest_rel(arr_a, arr_b)
            t_stat = float(t_stat)
            p_val = float(p_val)

        return SignificanceResult(
            metric_name=metric,
            model_a=model_a,
            model_b=model_b,
            n_samples=len(common_seeds),
            mean_a=float(np.mean(arr_a)),
            mean_b=float(np.mean(arr_b)),
            mean_difference=float(np.mean(diff)),
            t_statistic=t_stat,
            p_value=p_val,
            is_significant=bool(p_val < 0.05),
        )

    def export_csv(self, filepath: Path | str, kind: str = "benchmark") -> Path:
        """Export individual raw run results to CSV."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        if kind == "benchmark":
            fields = [
                "dataset_name", "model_family", "seed", "bps_held_in",
                "bps_held_out", "poisson_deviance", "pseudo_r2",
                "probe_r2", "duration_s", "model_sha256",
            ]
            rows = []
            for r in self.benchmark_results:
                rows.append({
                    "dataset_name": r.config.dataset_name,
                    "model_family": r.config.model_family,
                    "seed": r.config.seed,
                    "bps_held_in": r.bps_held_in,
                    "bps_held_out": r.bps_held_out,
                    "poisson_deviance": r.poisson_deviance,
                    "pseudo_r2": r.pseudo_r2,
                    "probe_r2": r.probe_r2,
                    "duration_s": r.duration_s,
                    "model_sha256": r.model_sha256,
                })
        elif kind == "ablation":
            fields = [
                "family", "variant_name", "dataset_name", "seed",
                "bps_held_in", "bps_held_out", "poisson_deviance",
                "pseudo_r2", "duration_s",
            ]
            rows = []
            for r in self.ablation_results:
                rows.append({
                    "family": r.config.family.value,
                    "variant_name": r.config.variant_name,
                    "dataset_name": r.config.dataset_name,
                    "seed": r.config.seed,
                    "bps_held_in": r.bps_held_in,
                    "bps_held_out": r.bps_held_out,
                    "poisson_deviance": r.poisson_deviance,
                    "pseudo_r2": r.pseudo_r2,
                    "duration_s": r.duration_s,
                })
        else:
            raise ValueError(f"Unknown kind {kind}, expected 'benchmark' or 'ablation'.")

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        return path

    def export_artifacts(
        self,
        output_dir: Path | str,
        registry: Optional[ArtifactRegistry] = None,
    ) -> List[ArtifactMetadata]:
        """Save Markdown, LaTeX, and CSV summaries as verifiable registered artifacts."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        reg = registry or ArtifactRegistry(out_dir)

        artifacts: List[ArtifactMetadata] = []

        # 1. Benchmark Markdown
        bm_md = out_dir / "benchmark_matrix.md"
        bm_md.write_text(self.to_markdown_benchmark_table(), encoding="utf-8")
        h_md = hashlib.sha256(bm_md.read_bytes()).hexdigest()
        meta_md = ArtifactMetadata(
            artifact_id="benchmark_matrix_md",
            artifact_type="metrics",
            dataset_name="all",
            split_name="test",
            model_name="all",
            seed=0,
            metrics={"runs": float(len(self.benchmark_results))},
            file_path=str(bm_md),
            sha256_hash=h_md,
        )
        reg.register(meta_md)
        artifacts.append(meta_md)

        # 2. Benchmark LaTeX
        bm_tex = out_dir / "benchmark_table.tex"
        bm_tex.write_text(self.to_latex_benchmark_table(), encoding="utf-8")
        h_tex = hashlib.sha256(bm_tex.read_bytes()).hexdigest()
        meta_tex = ArtifactMetadata(
            artifact_id="benchmark_table_latex",
            artifact_type="figure",
            dataset_name="all",
            split_name="test",
            model_name="all",
            seed=0,
            metrics={"runs": float(len(self.benchmark_results))},
            file_path=str(bm_tex),
            sha256_hash=h_tex,
        )
        reg.register(meta_tex)
        artifacts.append(meta_tex)

        # 3. Benchmark CSV
        bm_csv = out_dir / "benchmark_runs.csv"
        self.export_csv(bm_csv, kind="benchmark")
        h_csv = hashlib.sha256(bm_csv.read_bytes()).hexdigest()
        meta_csv = ArtifactMetadata(
            artifact_id="benchmark_runs_csv",
            artifact_type="metrics",
            dataset_name="all",
            split_name="test",
            model_name="all",
            seed=0,
            metrics={"total_rows": float(len(self.benchmark_results))},
            file_path=str(bm_csv),
            sha256_hash=h_csv,
        )
        reg.register(meta_csv)
        artifacts.append(meta_csv)

        return artifacts
