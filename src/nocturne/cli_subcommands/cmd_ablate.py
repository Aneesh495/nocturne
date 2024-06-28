"""Five-family ablation matrix execution CLI subcommand."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

from nocturne.campaign.ablations import (
    AblationFamily,
    AblationResult,
    AblationRunner,
)


def run_ablation_command(
    family: str = "all",
    dataset: str = "mc_maze",
    seeds: Tuple[int, ...] = (42,),
    num_steps: int = 15,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> int:
    """Execute ablation experiments and report comparative metrics."""
    if verbose:
        print(f"=== Nocturne Five-Family Ablation Engine ===")
        print(f"Family: {family} | Dataset: {dataset} | Seeds: {list(seeds)} | Steps: {num_steps}\n")

    results: List[AblationResult] = []

    if family == "all":
        results.extend(AblationRunner.run_objective_ablation(dataset_name=dataset, seeds=seeds, num_steps=num_steps))
        results.extend(AblationRunner.run_masking_ablation(dataset_name=dataset, seeds=seeds, num_steps=num_steps))
        results.extend(AblationRunner.run_depth_ablation(dataset_name=dataset, seeds=seeds, num_steps=num_steps))
        results.extend(AblationRunner.run_causality_ablation(dataset_name=dataset, seeds=seeds, num_steps=num_steps))
        results.extend(AblationRunner.run_dynamics_ablation(dataset_name=dataset, seeds=seeds, num_steps=num_steps))
    elif family in ("1", "objective", "family_1_objective"):
        results = AblationRunner.run_objective_ablation(dataset_name=dataset, seeds=seeds, num_steps=num_steps)
    elif family in ("2", "masking", "family_2_input_masking"):
        results = AblationRunner.run_masking_ablation(dataset_name=dataset, seeds=seeds, num_steps=num_steps)
    elif family in ("3", "depth", "family_3_depth_capacity"):
        results = AblationRunner.run_depth_ablation(dataset_name=dataset, seeds=seeds, num_steps=num_steps)
    elif family in ("4", "causality", "family_4_temporal_causality"):
        results = AblationRunner.run_causality_ablation(dataset_name=dataset, seeds=seeds, num_steps=num_steps)
    elif family in ("5", "dynamics", "family_5_dynamics_drive"):
        results = AblationRunner.run_dynamics_ablation(dataset_name=dataset, seeds=seeds, num_steps=num_steps)
    else:
        if verbose:
            print(f"Error: Unknown ablation family '{family}'")
        return 1

    if verbose:
        print("| Family | Variant | Seed | Held-in BPS | Held-out BPS | Pseudo R2 | Duration (s) |")
        print("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
        for r in results:
            hi = f"{r.bps_held_in:.4f}" if r.bps_held_in is not None else "N/A"
            ho = f"{r.bps_held_out:.4f}" if r.bps_held_out is not None else "N/A"
            fam_short = r.config.family.value.replace("family_", "F")
            print(f"| {fam_short:15s} | {r.config.variant_name:18s} | {r.config.seed:4d} | {hi:11s} | {ho:12s} | {r.pseudo_r2:9.4f} | {r.duration_s:12.2f} |")

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        if verbose:
            print(f"\nAblation results saved to: {p}")

    return 0


def add_ablation_arguments(parser: argparse.ArgumentParser) -> None:
    """Register CLI arguments for the ablate subcommand."""
    parser.add_argument(
        "--family",
        type=str,
        default="all",
        choices=["all", "objective", "masking", "depth", "causality", "dynamics"],
        help="Target ablation family",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mc_maze",
        help="Dataset name for ablation execution",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42",
        help="Comma-separated random seeds",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=15,
        help="Optimization steps per ablation variant",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional path to write ablation results JSON",
    )
