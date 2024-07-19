#!/usr/bin/env python3
"""Adversarial attractor falsification and surrogate null-hypothesis testing example."""

from __future__ import annotations

import sys
import numpy as np

from nocturne.verification.synthetic_falsification import (
    AdversarialAttractorFalsifier,
    FalsificationReport,
    FalsificationResult,
)


def main() -> int:
    print("=" * 60)
    print("Nocturne Neural Dynamics: Adversarial Attractor Falsification")
    print("=" * 60)

    falsifier = AdversarialAttractorFalsifier(
        dt=0.01,
        num_seeds=20,
        surrogate_shuffles=30,
        random_seed=42,
    )

    # 1. Test 1: Linear Spiral Sink Null Control
    print("\n--- Test 1: Linear Spiral Sink Null Control ---")
    print("Hypothesis: Vector field contains strictly 1 origin sink and 0 non-trivial attractors.")
    res_linear: FalsificationResult = falsifier.test_linear_sink_falsification()
    print(f"  Regime                : {res_linear.regime_name}")
    print(f"  Fixed points detected : {res_linear.detected_fixed_points_count}")
    print(f"  Spurious attractors   : {res_linear.spurious_attractor_detected}")
    print(f"  Falsification Verdict : {'PASSED (Null Verified)' if res_linear.passed else 'FAILED'}")

    # 2. Test 2: Stochastic Ornstein-Uhlenbeck Diffusion Surrogate Test
    print("\n--- Test 2: Stochastic OU Diffusion Fourier Surrogate Test ---")
    print("Hypothesis: Phase-randomized Fourier surrogates reject circular limit cycle claims on noise.")
    res_diff: FalsificationResult = falsifier.test_stochastic_diffusion_surrogate()
    base_circ = res_diff.details.get("base_circularity", 0.0)
    surr_circ = res_diff.details.get("surrogate_mean_circularity", 0.0)
    print(f"  Regime                : {res_diff.regime_name}")
    print(f"  Base circularity      : {base_circ:.4f}")
    print(f"  Surrogate circularity : {surr_circ:.4f}")
    print(f"  Empirical p-value     : {res_diff.surrogate_p_value:.4f}")
    print(f"  Falsification Verdict : {'PASSED (Null Rejected False Claim)' if res_diff.passed else 'FAILED'}")

    # 3. Overall Consolidated Report
    print("\n" + "=" * 60)
    report: FalsificationReport = falsifier.run_falsification_suite()
    print(f"All Adversarial Falsification Controls Passed: {report.all_passed}")
    print("=" * 60)

    print("\nAdversarial falsification audit completed successfully.")
    return 0 if report.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
