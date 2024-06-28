"""Multi-seed empirical benchmark campaign CLI subcommand."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

from nocturne.campaign.benchmark import (
    BenchmarkRunConfig,
    BenchmarkRunResult,
    BenchmarkSuiteRunner,
)


def run_benchmark_command(
    dataset: str = "mc_maze",
    model: str = "mean_rate",
    seeds: Tuple[int, ...] = (42, 43, 44),
    num_steps: int = 15,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> int:
    """Execute multi-seed benchmark run and emit results."""
    if verbose:
        print(f"=== Nocturne Benchmark Campaign ===")
        print(f"Dataset: {dataset} | Model: {model} | Seeds: {list(seeds)} | Steps: {num_steps}\n")

    results: List[BenchmarkRunResult] = BenchmarkSuiteRunner.run_multi_seed(
        dataset_name=dataset,
        model_family=model,
        seeds=seeds,
        num_steps=num_steps,
    )

    if verbose:
        print("| Seed | Held-in BPS | Held-out BPS | Pseudo R2 | Probe R2 | Latency (s) |")
        print("| :--- | :---------: | :----------: | :-------: | :------: | :---------: |")
        for r in results:
            hi = f"{r.bps_held_in:.4f}" if r.bps_held_in is not None else "N/A"
            ho = f"{r.bps_held_out:.4f}" if r.bps_held_out is not None else "N/A"
            pr = f"{r.probe_r2:.4f}" if r.probe_r2 is not None else "N/A"
            print(f"| {r.config.seed:4d} | {hi:11s} | {ho:12s} | {r.pseudo_r2:9.4f} | {pr:8s} | {r.duration_s:11.2f} |")

        # Mean and std
        bps_in_vals = [r.bps_held_in for r in results if r.bps_held_in is not None]
        bps_out_vals = [r.bps_held_out for r in results if r.bps_held_out is not None]
        if bps_in_vals and bps_out_vals:
            print("-" * 75)
            print(f"Held-in BPS:  {sum(bps_in_vals)/len(bps_in_vals):.4f}")
            print(f"Held-out BPS: {sum(bps_out_vals)/len(bps_out_vals):.4f}")

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        if verbose:
            print(f"\nBenchmark results saved to: {p}")

    return 0


def add_benchmark_arguments(parser: argparse.ArgumentParser) -> None:
    """Register CLI arguments for the benchmark subcommand."""
    parser.add_argument(
        "--dataset",
        type=str,
        default="mc_maze",
        choices=["mc_maze", "mc_rtt", "area2_bump", "linear_circuit", "nonlinear_circuit"],
        help="Target benchmark dataset or synthetic circuit",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mean_rate",
        choices=["mean_rate", "poisson_factor", "vldm", "attention"],
        help="Target model family architecture",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44",
        help="Comma-separated random seeds",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=15,
        help="Optimization steps per seed",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional path to write benchmark results JSON",
    )
