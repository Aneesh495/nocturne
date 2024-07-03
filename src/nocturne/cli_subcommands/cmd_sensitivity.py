"""Hyperparameter sensitivity sweep CLI subcommand."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from nocturne.campaign.sensitivity import (
    HyperparameterSensitivitySweep,
    SensitivityCurve,
    SensitivityPoint,
)


def run_sensitivity_command(
    parameter: str = "all",
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> int:
    """Execute hyperparameter sensitivity curves and dimensionality saturation sweeps."""
    if verbose:
        print("=== Nocturne Hyperparameter Sensitivity Sweep ===")
        print(f"Target parameter axis: {parameter}\n")

    sweep = HyperparameterSensitivitySweep(random_seed=42)
    curves = []

    if parameter in ("latent_dimension", "all"):
        dim_curve = sweep.sweep_latent_dimension(dimensions=[2, 4, 8, 16], T=100, N=20)
        curves.append(dim_curve)
        if verbose:
            print(sweep.format_markdown_table(dim_curve))
            elbow = dim_curve.find_elbow_point()
            print(f"Optimal Dimensionality Saturation Knee: d* = {int(elbow.parameter_value)} (BPS={elbow.co_smoothing_bps:.4f}, R2={elbow.decoding_r2:.4f})\n")

    if parameter in ("bin_width_ms", "all"):
        bw_curve = sweep.sweep_bin_width(bin_widths_ms=[10, 20, 50], total_time_s=2.0, N=20)
        curves.append(bw_curve)
        if verbose:
            print(sweep.format_markdown_table(bw_curve))

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            curve.parameter_name: [
                {
                    "parameter_name": pt.parameter_name,
                    "parameter_value": pt.parameter_value,
                    "co_smoothing_bps": pt.co_smoothing_bps,
                    "decoding_r2": pt.decoding_r2,
                    "runtime_seconds": pt.runtime_seconds,
                }
                for pt in curve.points
            ]
            for curve in curves
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        if verbose:
            print(f"Sensitivity sweep data saved to: {p}")

    return 0


def add_sensitivity_arguments(parser: argparse.ArgumentParser) -> None:
    """Register CLI arguments for the sensitivity subcommand."""
    parser.add_argument(
        "--parameter",
        type=str,
        default="all",
        choices=["all", "latent_dimension", "bin_width_ms"],
        help="Target hyperparameter parameter sweep axis",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional path to write sensitivity metrics JSON",
    )
