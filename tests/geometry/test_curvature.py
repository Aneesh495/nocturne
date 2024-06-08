"""Unit tests for trajectory curvature and manifold smoothness metrics."""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from nocturne.geometry.curvature import CurvatureAnalyzer, TrajectoryCurvatureResult


def test_straight_line_zero_curvature() -> None:
    dt = 0.01
    t = np.linspace(0, 1, 101)
    # Uniform linear motion in 3D: z(t) = [2*t, 3*t, -t]
    traj = np.column_stack([2.0 * t, 3.0 * t, -1.0 * t])

    analyzer = CurvatureAnalyzer()
    res = analyzer.compute_trajectory_curvature(traj, dt=dt)

    assert isinstance(res, TrajectoryCurvatureResult)
    # Theoretical curvature of a straight line is 0
    assert res.mean_curvature < 1e-4
    assert res.max_curvature < 1e-4
    assert res.mean_turning_angle_rad < 1e-4
    expected_speed = math.sqrt(2.0**2 + 3.0**2 + (-1.0)**2)
    assert pytest.approx(res.mean_speed, rel=1e-3) == expected_speed


def test_circular_orbit_known_radius() -> None:
    dt = 0.005
    R = 2.5
    omega = 4.0
    t = np.arange(0, 2.0 * math.pi / omega, dt)
    # 2D circle with radius R: x = R cos(omega t), y = R sin(omega t)
    traj = np.column_stack([R * np.cos(omega * t), R * np.sin(omega * t)])

    analyzer = CurvatureAnalyzer()
    res = analyzer.compute_trajectory_curvature(traj, dt=dt)

    # Theoretical curvature of circle radius R is kappa = 1 / R = 1 / 2.5 = 0.4
    expected_curvature = 1.0 / R
    assert pytest.approx(res.mean_curvature, rel=0.05) == expected_curvature
    # Theoretical speed is v = omega * R = 4.0 * 2.5 = 10.0
    assert pytest.approx(res.mean_speed, rel=0.05) == 10.0


def test_pytorch_tensor_and_batch_comparison() -> None:
    dt = 0.02
    T = 50
    # Batch A: smooth slow sine wave
    t = torch.linspace(0, 2 * math.pi, T)
    smooth_traj = torch.stack([torch.sin(t), torch.cos(t)], dim=1).unsqueeze(0).repeat(3, 1, 1)

    # Batch B: jittery fast trajectory (higher curvature)
    jittery_traj = smooth_traj + 0.1 * torch.randn_like(smooth_traj)

    analyzer = CurvatureAnalyzer()
    cmp = analyzer.compare_smoothness(smooth_traj, jittery_traj, dt=dt)

    assert cmp["mean_curvature_b"] > cmp["mean_curvature_a"]
    assert cmp["smoothness_index_b"] > cmp["smoothness_index_a"]
    assert cmp["curvature_ratio_b_to_a"] > 1.0


def test_insufficient_time_points_raises() -> None:
    short_traj = np.ones((2, 4))
    analyzer = CurvatureAnalyzer()
    with pytest.raises(ValueError, match="at least 3 points"):
        analyzer.compute_trajectory_curvature(short_traj)
