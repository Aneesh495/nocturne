"""Tests for FixedPointFinder identifying equilibria in nonlinear vector fields."""

from __future__ import annotations

import numpy as np
import torch
import pytest

from nocturne.geometry import FixedPointFinder


def test_fixed_point_finder_bistable_potential():
    # Bistable 1D double-well vector field: dz/dt = -4 * z * (z^2 - 1)
    # Fixed points at z = -1 (stable), z = 0 (unstable), z = +1 (stable)
    def bistable_vf(z: torch.Tensor) -> torch.Tensor:
        return -4.0 * z * (z ** 2 - 1.0)

    finder = FixedPointFinder(
        speed_tol=1e-3,
        slow_tol=1e-1,
        max_iters=100,
        lr=0.05,
        cluster_dist=0.3,
    )

    # Seeds near -1, 0, +1
    seeds = torch.tensor([[-1.2], [-0.8], [0.1], [0.9], [1.3]], dtype=torch.float32)

    fixed_points = finder.find_fixed_points(bistable_vf, seeds)

    assert len(fixed_points) >= 2
    locations = [fp.location[0] for fp in fixed_points]

    # Verify that points near -1 and +1 were found
    assert any(np.isclose(loc, -1.0, atol=0.05) for loc in locations)
    assert any(np.isclose(loc, 1.0, atol=0.05) for loc in locations)
