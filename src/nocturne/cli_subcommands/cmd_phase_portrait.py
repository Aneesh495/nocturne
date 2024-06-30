"""Latent phase portrait and vector field visualization CLI subcommand."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional
import numpy as np

from nocturne.synthetic.generator import SyntheticDatasetGenerator
from nocturne.viewer.phase_portrait import PhasePortraitVisualizer


def run_phase_portrait_command(
    regime: str = "nonlinear_oscillator",
    format_type: str = "both",
    num_trials: int = 4,
    duration_s: float = 1.0,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> int:
    """Generate and render continuous latent state phase portraits."""
    generator = SyntheticDatasetGenerator(seed=42)
    circuit = generator.create_circuit(regime)

    # Generate simulations using generate_regime_corpus
    sims, _ = generator.generate_regime_corpus(
        regime_name=regime,
        num_trials=num_trials,
        trial_duration_s=duration_s,
    )
    traj_arr = np.stack([s.latent.states for s in sims], axis=0)  # [K, T, D]



    if format_type in ("ascii", "both"):
        ascii_art = PhasePortraitVisualizer.render_ascii(traj_arr)
        if verbose:
            print(f"=== Latent Phase Portrait: {regime} ===")
            print(ascii_art)

    if format_type in ("svg", "both") or output_path is not None:
        svg_xml = PhasePortraitVisualizer.render_svg(
            trajectories=traj_arr,
            title=f"Latent Phase Portrait: {regime} ({num_trials} trials)",
        )
        if output_path is not None:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(svg_xml)
            if verbose:
                print(f"\nVector SVG phase portrait saved to: {p}")

    return 0


def add_phase_portrait_arguments(parser: argparse.ArgumentParser) -> None:
    """Register CLI arguments for the phase portrait subcommand."""
    parser.add_argument(
        "--regime",
        type=str,
        default="nonlinear_oscillator",
        choices=["linear_oscillator", "nonlinear_oscillator", "bistable_memory", "driven_recurrent"],
        help="Dynamical system circuit regime",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="both",
        choices=["ascii", "svg", "both"],
        help="Output visualization format",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=4,
        help="Number of trajectory realizations to overlay",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Duration of trajectories in seconds",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional path to write SVG phase portrait XML",
    )
