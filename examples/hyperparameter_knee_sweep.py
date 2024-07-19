#!/usr/bin/env python3
"""Latent dimensionality sensitivity sweep and saturation elbow knee detection example."""

from __future__ import annotations

import sys

from nocturne.campaign.sensitivity import (
    HyperparameterSensitivitySweep,
    SensitivityCurve,
    SensitivityPoint,
)


def main() -> int:
    print("=" * 60)
    print("Nocturne Neural Dynamics: Latent Dimensionality Knee Sweep")
    print("=" * 60)

    sweep_engine = HyperparameterSensitivitySweep(random_seed=42)
    dimensions = [2, 4, 8, 16, 24, 32]

    print(f"Sweeping latent dimensionality D in {dimensions}...")
    curve: SensitivityCurve = sweep_engine.sweep_latent_dimension(
        dimensions=dimensions,
        T=150,
        N=20,
    )

    print("\nDimensionality Sensitivity Results:")
    print(f"{'Dim (D)':<8} | {'Co-Smoothing BPS':<18} | {'Decoding R2':<14} | {'Runtime (ms)':<12}")
    print("-" * 58)

    for pt in curve.points:
        runtime_ms = pt.runtime_seconds * 1000.0
        print(
            f"{int(pt.parameter_value):<8} | {pt.co_smoothing_bps:<18.4f} | "
            f"{pt.decoding_r2:<14.4f} | {runtime_ms:<12.2f}"
        )

    # Detect saturation elbow point via maximum secant line distance
    elbow: SensitivityPoint = curve.find_elbow_point()
    print("\n" + "=" * 60)
    print("Saturation Knee Detection Analysis:")
    print(f"  Detected Optimal Latent Dimension: D = {int(elbow.parameter_value)}")
    print(f"  Elbow Co-Smoothing BPS           : {elbow.co_smoothing_bps:.4f}")
    print(f"  Elbow Behavioral Decoding R2     : {elbow.decoding_r2:.4f}")
    print("=" * 60)

    print("\nHyperparameter knee sweep completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
