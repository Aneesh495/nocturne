"""Riemannian trajectory curvature, speed profiles, and manifold smoothness metrics.

Calculates extrinsic path curvature, instantaneous acceleration, integrated path length,
and turning angles in neural latent dynamical state spaces according to differential geometry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch


@dataclass(frozen=True)
class TrajectoryCurvatureResult:
    """Quantitative curvature, velocity, and smoothness metrics for a latent path."""

    path_length: float
    mean_speed: float
    max_speed: float
    mean_curvature: float
    max_curvature: float
    mean_turning_angle_rad: float
    smoothness_index: float
    speeds: List[float]
    curvatures: List[float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert result container to dictionary."""
        return asdict(self)


class CurvatureAnalyzer:
    """Computes geometric path curvature and smoothness along neural trajectories."""

    def __init__(self, eps: float = 1e-9):
        self.eps = eps

    def compute_trajectory_curvature(
        self,
        trajectory: np.ndarray | torch.Tensor,
        dt: float = 0.02,
    ) -> TrajectoryCurvatureResult:
        """Compute differential geometric curvature and speed along a trajectory.

        Args:
            trajectory: Latent state trajectory array of shape [T, d] or [1, T, d].
            dt: Temporal sampling interval in seconds.

        Returns:
            TrajectoryCurvatureResult with velocity, acceleration, and curvature metrics.
        """
        if isinstance(trajectory, torch.Tensor):
            z = trajectory.detach().cpu().numpy().astype(np.float64)
        else:
            z = np.asarray(trajectory, dtype=np.float64)

        if z.ndim == 3:
            if z.shape[0] != 1:
                raise ValueError(f"Expected single trajectory [T, d], got batched shape {z.shape}")
            z = z[0]

        T, d = z.shape
        if T < 3:
            raise ValueError(f"Trajectory requires at least 3 points for curvature, got {T}")

        # Integrated Euclidean path length
        deltas = np.diff(z, axis=0)  # [T-1, d]
        step_lengths = np.linalg.norm(deltas, axis=1)  # [T-1]
        path_length = float(np.sum(step_lengths))

        # Central difference velocity: v(t) = (z(t+1) - z(t-1)) / (2 * dt)
        v = (z[2:] - z[:-2]) / (2.0 * dt)  # [T-2, d]
        speeds = np.linalg.norm(v, axis=1)  # [T-2]

        # Second difference acceleration: a(t) = (z(t+1) - 2*z(t) + z(t-1)) / (dt^2)
        a = (z[2:] - 2.0 * z[1:-1] + z[:-2]) / (dt**2)  # [T-2, d]
        acc_norms = np.linalg.norm(a, axis=1)  # [T-2]

        # Dot product v(t) . a(t)
        v_dot_a = np.sum(v * a, axis=1)  # [T-2]

        # Extrinsic Frenet-Serret curvature:
        # kappa(t) = sqrt(||v||^2 ||a||^2 - (v . a)^2) / (||v||^3 + eps)
        cross_term = np.maximum(0.0, (speeds**2) * (acc_norms**2) - (v_dot_a**2))
        numerator = np.sqrt(cross_term)
        denominator = (speeds**3) + self.eps
        curvatures = numerator / denominator  # [T-2]

        # Successive turning angles in radians: theta_t between delta_{t-1} and delta_t
        dot_consecutive = np.sum(deltas[:-1] * deltas[1:], axis=1)
        norms_consecutive = np.maximum(step_lengths[:-1] * step_lengths[1:], self.eps)
        cosine_sim = np.clip(dot_consecutive / norms_consecutive, -1.0, 1.0)
        turning_angles = np.arccos(cosine_sim)  # [T-2]

        # Smoothness index: average norm of acceleration
        smoothness_index = float(np.mean(acc_norms))

        return TrajectoryCurvatureResult(
            path_length=path_length,
            mean_speed=float(np.mean(speeds)),
            max_speed=float(np.max(speeds)),
            mean_curvature=float(np.mean(curvatures)),
            max_curvature=float(np.max(curvatures)),
            mean_turning_angle_rad=float(np.mean(turning_angles)),
            smoothness_index=smoothness_index,
            speeds=[float(s) for s in speeds],
            curvatures=[float(c) for c in curvatures],
        )

    def compare_smoothness(
        self,
        trajectories_a: np.ndarray | torch.Tensor,
        trajectories_b: np.ndarray | torch.Tensor,
        dt: float = 0.02,
    ) -> Dict[str, float]:
        """Compare mean curvature and acceleration across two sets of trajectories.

        Args:
            trajectories_a: Batch of trajectories [B, T, d].
            trajectories_b: Batch of trajectories [B, T, d].
            dt: Temporal step in seconds.

        Returns:
            Dictionary with comparative smoothness ratios and curvature metrics.
        """
        if isinstance(trajectories_a, torch.Tensor):
            arr_a = trajectories_a.detach().cpu().numpy().astype(np.float64)
        else:
            arr_a = np.asarray(trajectories_a, dtype=np.float64)

        if isinstance(trajectories_b, torch.Tensor):
            arr_b = trajectories_b.detach().cpu().numpy().astype(np.float64)
        else:
            arr_b = np.asarray(trajectories_b, dtype=np.float64)

        curvs_a = [self.compute_trajectory_curvature(traj, dt).mean_curvature for traj in arr_a]
        curvs_b = [self.compute_trajectory_curvature(traj, dt).mean_curvature for traj in arr_b]

        smooth_a = [self.compute_trajectory_curvature(traj, dt).smoothness_index for traj in arr_a]
        smooth_b = [self.compute_trajectory_curvature(traj, dt).smoothness_index for traj in arr_b]

        mean_c_a = float(np.mean(curvs_a))
        mean_c_b = float(np.mean(curvs_b))
        mean_s_a = float(np.mean(smooth_a))
        mean_s_b = float(np.mean(smooth_b))

        return {
            "mean_curvature_a": mean_c_a,
            "mean_curvature_b": mean_c_b,
            "curvature_ratio_b_to_a": mean_c_b / (mean_c_a + self.eps),
            "smoothness_index_a": mean_s_a,
            "smoothness_index_b": mean_s_b,
            "smoothness_ratio_b_to_a": mean_s_b / (mean_s_a + self.eps),
        }
