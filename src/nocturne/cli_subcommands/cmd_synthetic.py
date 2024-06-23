"""Synthetic circuit generation and benchmark simulation CLI subcommand."""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional
import numpy as np

from nocturne.synthetic.generator import SyntheticDatasetGenerator


def run_synthetic_command(
    regime: str = "all",
    num_trials: int = 10,
    num_neurons: int = 24,
    bin_width_ms: int = 20,
    duration_s: float = 1.0,
    seed: int = 42,
    verbose: bool = True,
) -> int:
    """Generate controlled synthetic neural population circuits."""
    generator = SyntheticDatasetGenerator(seed=seed)
    bin_width_s = bin_width_ms / 1000.0

    if regime == "all":
        regimes_to_run = generator.SUPPORTED_REGIMES
    elif regime in generator.SUPPORTED_REGIMES:
        regimes_to_run = [regime]
    else:
        if verbose:
            print(f"Error: Unknown regime '{regime}'. Supported: {generator.SUPPORTED_REGIMES}")
        return 1

    if verbose:
        print(f"=== Generating Synthetic Neural Circuits (Seed={seed}) ===")
        print(f"Parameters: {num_trials} trials/regime, {num_neurons} neurons, {bin_width_ms}ms bins, {duration_s}s duration\n")

    for reg_name in regimes_to_run:
        sims, processed_rec = generator.generate_regime_corpus(
            regime_name=reg_name,
            num_trials=num_trials,
            num_neurons=num_neurons,
            bin_width_s=bin_width_s,
            trial_duration_s=duration_s,
        )

        total_spikes = sum(float(sim.spike_counts.sum()) for sim in sims)
        mean_rate = total_spikes / (num_trials * num_neurons * duration_s)


        if verbose:
            print(f"  [{reg_name:22s}] {len(sims)} trials, {num_neurons} neurons | {int(total_spikes):6d} spikes ({mean_rate:5.2f} Hz avg)")

    if verbose:
        print("\nSynthetic circuit simulation successfully completed.")
    return 0


def add_synthetic_arguments(parser: argparse.ArgumentParser) -> None:
    """Register command line arguments for the synthetic subcommand."""
    parser.add_argument(
        "--regime",
        type=str,
        default="all",
        choices=["all", "linear_oscillator", "nonlinear_oscillator", "bistable_memory", "driven_recurrent"],
        help="Target dynamical circuit regime or 'all'",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of trials per regime",
    )
    parser.add_argument(
        "--neurons",
        type=int,
        default=24,
        help="Number of observed neurons in the population",
    )
    parser.add_argument(
        "--bin-width-ms",
        type=int,
        default=20,
        help="Discretization bin width in milliseconds",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Trial duration in seconds",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
