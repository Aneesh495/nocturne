"""Neural spike train raster visualization CLI subcommand."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional
import numpy as np

from nocturne.campaign.benchmark import BenchmarkSuiteRunner
from nocturne.viewer.raster import RasterVisualizer


def run_raster_command(
    dataset: str = "mc_maze",
    format_type: str = "both",
    max_bins: int = 50,
    max_neurons: int = 16,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> int:
    """Render neural spike raster plot in ASCII and/or SVG format."""
    # Load dataset sample
    train_counts, _, _, _, _ = BenchmarkSuiteRunner.load_dataset(dataset, seed=42)
    # Take first trial: [T, N]
    sample_counts = train_counts[0]
    T, N = sample_counts.shape
    neuron_indices = list(range(min(N, max_neurons)))

    if format_type in ("ascii", "both"):
        ascii_repr = RasterVisualizer.render_ascii(
            counts=sample_counts,
            neuron_indices=neuron_indices,
            max_time_bins=max_bins,
        )
        if verbose:
            print(ascii_repr)

    if format_type in ("svg", "both") or output_path is not None:
        svg_xml = RasterVisualizer.render_svg(
            counts=sample_counts[:max_bins],
            neuron_indices=neuron_indices,
            bin_width_s=0.02,
            title=f"Spike Raster: {dataset} (T={min(T, max_bins)} bins)",
        )
        if output_path is not None:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(svg_xml)
            if verbose:
                print(f"\nVector SVG raster saved to: {p}")

    return 0


def add_raster_arguments(parser: argparse.ArgumentParser) -> None:
    """Register CLI arguments for the raster subcommand."""
    parser.add_argument(
        "--dataset",
        type=str,
        default="mc_maze",
        choices=["mc_maze", "mc_rtt", "area2_bump", "linear_circuit", "nonlinear_circuit"],
        help="Dataset from which to extract spike recordings",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="both",
        choices=["ascii", "svg", "both"],
        help="Output visualization format",
    )
    parser.add_argument(
        "--max-bins",
        type=int,
        default=50,
        help="Maximum number of temporal bins to display",
    )
    parser.add_argument(
        "--max-neurons",
        type=int,
        default=16,
        help="Maximum number of neurons to display",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional path to write SVG output",
    )
