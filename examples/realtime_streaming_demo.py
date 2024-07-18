#!/usr/bin/env python3
"""Real-time 50 Hz causal streaming and closed-loop neurostimulation demonstration."""

from __future__ import annotations

import sys
from nocturne.workflows.streaming_simulation import run_streaming_simulation


def main() -> int:
    print("=" * 60)
    print("Nocturne Neural Dynamics: 50 Hz Closed-Loop Streaming Demo")
    print("=" * 60)

    num_neurons = 16
    latent_dim = 4
    total_bins = 60
    chunk_size_bins = 1  # Single-bin streaming tick: 50 Hz updates
    bin_width_ms = 20    # 20 ms bin duration
    rate_threshold_hz = 30.0

    print(f"Sampling configuration:")
    print(f"  Neural channels          : {num_neurons}")
    print(f"  Bin width                : {bin_width_ms} ms (50 Hz streaming loop)")
    print(f"  Chunk size               : {chunk_size_bins} bin per hardware tick")
    print(f"  Total session bins       : {total_bins} ({total_bins * bin_width_ms / 1000.0:.1f} s)")
    print(f"  Closed-loop trigger rate : {rate_threshold_hz:.1f} Hz")

    print("\nRunning causal streaming simulation with neurostimulation feedback...")
    result = run_streaming_simulation(
        num_neurons=num_neurons,
        latent_dim=latent_dim,
        total_bins=total_bins,
        chunk_size_bins=chunk_size_bins,
        bin_width_ms=bin_width_ms,
        rate_threshold_hz=rate_threshold_hz,
        min_isi_s=0.06,
        blanking_duration_s=0.02,
        circuit_name="nonlinear_oscillator",
        seed=42,
    )

    print("\n" + result.ascii_summary)
    print("\nClosed-Loop Stimulation Metrics:")
    print(f"  Stimulation pulses triggered: {result.num_stim_events}")
    print(f"  Artifact blanked bins       : {result.num_blanked_bins}")
    print(f"  Deadline budget (20ms) hits : {result.deadline_violations} violations")
    print(f"  Reconstruction RMSE         : {result.reconstruction_rmse:.4f} Hz")

    if result.stim_events:
        print("\nFirst 3 Dispatched Stimulation Events:")
        for idx, ev in enumerate(result.stim_events[:3]):
            print(
                f"  [{idx + 1}] Time: {ev.timestamp_s:.3f} s | Type: {ev.trigger_type} | "
                f"Latency: {ev.latency_ms:.2f} ms | Energy: {ev.energy_units:.2f} uJ"
            )

    print("\n50 Hz closed-loop streaming demonstration completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
