#!/usr/bin/env python3
"""Quickstart example: Simulating controlled synthetic neural dynamics across four regimes."""

from __future__ import annotations

import sys
import numpy as np

from nocturne.synthetic.generator import SyntheticDatasetGenerator
from nocturne.viewer.raster import RasterVisualizer


def main() -> int:
    print("=" * 60)
    print("Nocturne Neural Dynamics: Synthetic Circuits Quickstart")
    print("=" * 60)

    seed = 42
    generator = SyntheticDatasetGenerator(seed=seed)
    num_neurons = 12
    duration_s = 0.5
    bin_width_s = 0.02

    regimes = [
        "linear_oscillator",
        "nonlinear_oscillator",
        "bistable_memory",
        "driven_recurrent",
    ]

    for regime in regimes:
        print(f"\n--- Simulating Regime: {regime} ---")
        sims, split = generator.generate_regime_corpus(
            regime_name=regime,
            num_trials=2,
            num_neurons=num_neurons,
            bin_width_s=bin_width_s,
            trial_duration_s=duration_s,
        )
        sim = sims[0]
        latent_states = sim.latent.states  # [T, D]
        spikes = sim.spike_counts          # [T, N]
        rates = sim.true_rates_hz          # [T, N]

        total_spikes = int(np.sum(spikes))
        mean_rate = float(np.mean(rates))
        latent_dim = latent_states.shape[1]

        print(f"  Latent dimension     : {latent_dim}")
        print(f"  Temporal bins (T)    : {spikes.shape[0]}")
        print(f"  Monitored neurons (N): {num_neurons}")
        print(f"  Total spikes emitted : {total_spikes}")
        print(f"  Mean population rate : {mean_rate:.2f} Hz")

        # Display ASCII raster for the first 6 neurons over first 25 bins
        ascii_raster = RasterVisualizer.render_ascii(
            counts=spikes[:25, :6],
            max_time_bins=25,
        )
        print("  Sample Spike Raster (ASCII):")
        for line in ascii_raster.splitlines():
            print(f"    {line}")

    print("\nQuickstart simulation completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
