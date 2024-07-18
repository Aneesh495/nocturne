#!/usr/bin/env python3
"""Five-family ablation matrix execution and comparative degradation analysis."""

from __future__ import annotations

import sys
from typing import List

from nocturne.campaign.ablations import AblationResult, AblationRunner


def main() -> int:
    print("=" * 60)
    print("Nocturne Neural Dynamics: 5-Family Ablation Sweep")
    print("=" * 60)

    seeds = (42,)
    num_steps = 5

    print(f"Executing 5-family ablation matrix (seeds={seeds}, steps={num_steps}):\n")

    all_results: List[AblationResult] = []

    # Family 1: Loss Objective
    print("Running Family 1: Objective Formulation (Poisson vs MSE vs Gaussian)...")
    res_obj = AblationRunner.run_objective_ablation(seeds=seeds, num_steps=num_steps)
    all_results.extend(res_obj)

    # Family 2: Input Masking
    print("Running Family 2: Input Masking (Co-Smoothing vs Unmasked vs Dropout)...")
    res_mask = AblationRunner.run_masking_ablation(seeds=seeds, num_steps=num_steps)
    all_results.extend(res_mask)

    # Family 3: Depth & Capacity
    print("Running Family 3: Depth & Hidden Capacity Scaling...")
    res_cap = AblationRunner.run_capacity_ablation(seeds=seeds, num_steps=num_steps)
    all_results.extend(res_cap)

    # Family 4: Temporal Causality
    print("Running Family 4: Temporal Causality (Causal vs Bidirectional)...")
    res_caus = AblationRunner.run_causality_ablation(seeds=seeds, num_steps=num_steps)
    all_results.extend(res_caus)

    # Family 5: Dynamics Drive
    print("Running Family 5: Latent Dynamics Drive (Autonomous vs Inferred Drive)...")
    res_dyn = AblationRunner.run_dynamics_ablation(seeds=seeds, num_steps=num_steps)
    all_results.extend(res_dyn)

    print("\n" + "=" * 60)
    print("Ablation Study Summary Table")
    print("=" * 60)
    print(f"{'Family':<24} | {'Variant':<20} | {'BPS Held-Out':<12} | {'Poisson Dev':<12}")
    print("-" * 76)

    for r in all_results:
        bps_str = f"{r.bps_held_out:.4f}" if r.bps_held_out is not None else "N/A"
        dev_str = f"{r.poisson_deviance:.4f}"
        print(f"{r.config.family.value:<24} | {r.config.variant_name:<20} | {bps_str:<12} | {dev_str:<12}")

    print("\nAblation sweep across all 5 families completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
