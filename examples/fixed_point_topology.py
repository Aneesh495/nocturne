#!/usr/bin/env python3
"""Fixed point discovery, Jacobian eigenspectra, and dynamical topology example."""

from __future__ import annotations

import sys
from typing import Callable, List
import numpy as np
import torch

from nocturne.geometry.fixed_points import FixedPoint, FixedPointFinder
from nocturne.geometry.jacobians import JacobianAnalyzer, JacobianDiagnostics


def bistable_vector_field(z: torch.Tensor, alpha: float = 2.0, decay: float = 1.0) -> torch.Tensor:
    """Bistable double-well vector field dz1/dt = alpha*z1 - z1^3, dz2/dt = -decay*z2."""
    z1 = z[:, 0]
    z2 = z[:, 1]
    dz1 = alpha * z1 - (z1 ** 3)
    dz2 = -decay * z2
    return torch.stack([dz1, dz2], dim=-1)


def hopf_vector_field(z: torch.Tensor, mu: float = 1.0, omega: float = 1.0, b: float = 0.5) -> torch.Tensor:
    """Supercritical Hopf limit cycle vector field."""
    z1 = z[:, 0]
    z2 = z[:, 1]
    r2 = z1**2 + z2**2
    radial = mu - r2
    angular = omega + b * r2
    dz1 = radial * z1 - angular * z2
    dz2 = angular * z1 + radial * z2
    return torch.stack([dz1, dz2], dim=-1)


def main() -> int:
    print("=" * 60)
    print("Nocturne Neural Dynamics: Fixed Point Topology & Jacobians")
    print("=" * 60)

    finder = FixedPointFinder(
        speed_tol=1e-3,
        slow_tol=1e-1,
        max_iters=150,
        lr=0.05,
        cluster_dist=0.2,
    )

    # 1. Analyze Bistable Memory Attractor System
    print("\n--- System 1: Bistable Double-Well Memory ---")
    seeds_bistable = torch.tensor([
        [-1.8, 0.2],
        [-1.0, -0.1],
        [0.05, 0.05],
        [1.2, -0.2],
        [1.8, 0.1],
    ], dtype=torch.float32)

    fps_bistable: List[FixedPoint] = finder.find_fixed_points(
        bistable_vector_field,
        seeds_bistable,
    )
    print(f"Discovered {len(fps_bistable)} unique equilibria:")

    for idx, fp in enumerate(fps_bistable, start=1):
        diag: JacobianDiagnostics = JacobianAnalyzer.analyze_fixed_point(
            bistable_vector_field,
            fp.location,
        )
        eigs_str = ", ".join([f"{e.real:.2f}+{e.imag:.2f}j" if abs(e.imag) > 1e-4 else f"{e.real:.2f}" for e in diag.eigenvalues])
        print(
            f"  [{idx}] z*=[{fp.location[0]:6.3f}, {fp.location[1]:6.3f}] | "
            f"Speed={fp.speed:.2e} | Type={diag.topological_type:<14} | "
            f"Stable={diag.is_stable} | Eigs=[{eigs_str}]"
        )

    # 2. Analyze Supercritical Hopf Limit Cycle System
    print("\n--- System 2: Supercritical Hopf Oscillator ---")
    seeds_hopf = torch.tensor([
        [0.01, 0.01],
        [-0.05, 0.02],
        [0.0, 0.0],
    ], dtype=torch.float32)

    fps_hopf: List[FixedPoint] = finder.find_fixed_points(
        hopf_vector_field,
        seeds_hopf,
    )
    print(f"Discovered {len(fps_hopf)} unique equilibria:")

    for idx, fp in enumerate(fps_hopf, start=1):
        diag = JacobianAnalyzer.analyze_fixed_point(
            hopf_vector_field,
            fp.location,
        )
        eigs_str = ", ".join([f"{e.real:.2f}+{e.imag:.2f}j" if abs(e.imag) > 1e-4 else f"{e.real:.2f}" for e in diag.eigenvalues])
        print(
            f"  [{idx}] z*=[{fp.location[0]:6.3f}, {fp.location[1]:6.3f}] | "
            f"Speed={fp.speed:.2e} | Type={diag.topological_type:<14} | "
            f"Stable={diag.is_stable} | Eigs=[{eigs_str}]"
        )

    print("\nFixed point topological audit completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
