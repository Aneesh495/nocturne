"""Adversarial attractor falsification and surrogate testing CLI subcommand."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from nocturne.verification.synthetic_falsification import (
    AdversarialAttractorFalsifier,
    FalsificationReport,
    FalsificationResult,
)


def run_falsify_command(
    num_seeds: int = 15,
    surrogate_shuffles: int = 30,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> int:
    """Execute adversarial attractor falsification suite."""
    if verbose:
        print("=== Nocturne Adversarial Attractor Falsification Suite ===")
        print(f"Seeds: {num_seeds} | Fourier Surrogates: {surrogate_shuffles} per null regime\n")

    falsifier = AdversarialAttractorFalsifier(
        num_seeds=num_seeds,
        surrogate_shuffles=surrogate_shuffles,
        random_seed=42,
    )

    report: FalsificationReport = falsifier.run_falsification_suite()

    if verbose:
        print("| Regime | Passed | Spurious Attractor? | Fixed Points | Surrogate p-value |")
        print("| :--- | :---: | :---: | :---: | :---: |")
        for r in report.results:
            p_val_str = f"{r.surrogate_p_value:.3f}" if r.surrogate_p_value < 1.0 else "N/A (Linear)"
            print(f"| {r.regime_name:30s} | {'YES' if r.passed else 'NO':6s} | {'YES' if r.spurious_attractor_detected else 'NO':19s} | {r.detected_fixed_points_count:12d} | {p_val_str:17s} |")

        print("-" * 75)
        print(f"Falsification Verdict: {'ALL NULL CONTROLS CONFIRMED (Zero spurious attractors)' if report.all_passed else 'FALSIFICATION FAILED'}")

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "all_passed": report.all_passed,
            "results": [
                {
                    "regime_name": r.regime_name,
                    "passed": r.passed,
                    "spurious_attractor_detected": r.spurious_attractor_detected,
                    "detected_fixed_points_count": r.detected_fixed_points_count,
                    "surrogate_p_value": r.surrogate_p_value,
                    "details": r.details,
                }
                for r in report.results
            ],
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        if verbose:
            print(f"\nFalsification audit report saved to: {p}")

    return 0 if report.all_passed else 1


def add_falsify_arguments(parser: argparse.ArgumentParser) -> None:
    """Register CLI arguments for the falsify subcommand."""
    parser.add_argument(
        "--seeds",
        type=int,
        default=15,
        help="Number of initial candidate state seeds for fixed-point optimization",
    )
    parser.add_argument(
        "--shuffles",
        type=int,
        default=30,
        help="Number of phase-randomized Fourier surrogate realizations",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional path to write falsification report JSON",
    )
