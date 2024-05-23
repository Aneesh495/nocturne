"""Numerical identification and clustering of fixed points and slow points in vector fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional
import numpy as np
import torch


@dataclass(frozen=True)
class FixedPoint:
    """Identified fixed point or slow point in continuous latent space."""

    location: np.ndarray       # [D] coordinate vector z*
    speed: float               # ||f(z*)|| magnitude
    is_true_fixed_point: bool  # True if speed <= speed_tol
    is_slow_point: bool        # True if speed <= slow_tol


class FixedPointFinder:
    """Finds equilibria and slow points where velocity field f(z) vanishes or slows.

    Minimizes kinetic energy objective:
        q(z) = 0.5 * ||f_theta(z)||^2
    via gradient descent from multiple trajectory seed points.
    """

    def __init__(
        self,
        speed_tol: float = 1e-3,
        slow_tol: float = 1e-1,
        max_iters: int = 150,
        lr: float = 0.05,
        cluster_dist: float = 0.1,
    ):
        self.speed_tol = speed_tol
        self.slow_tol = slow_tol
        self.max_iters = max_iters
        self.lr = lr
        self.cluster_dist = cluster_dist

    def find_fixed_points(
        self,
        vector_field: Callable[[torch.Tensor], torch.Tensor],
        initial_seeds: torch.Tensor,
    ) -> List[FixedPoint]:
        """Optimize candidate seeds to locate fixed points and slow points.

        Args:
            vector_field: Callable f(z) taking tensor [K, D] and returning velocities [K, D].
            initial_seeds: Tensor [K, D] of initial coordinate guesses.

        Returns:
            List of unique, clustered FixedPoint instances.
        """
        device = initial_seeds.device
        z = initial_seeds.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=self.lr)

        for _ in range(self.max_iters):
            optimizer.zero_grad()
            vel = vector_field(z)
            q = 0.5 * (vel ** 2).sum(dim=-1).mean()
            q.backward()
            optimizer.step()

        # Evaluate final speeds
        with torch.no_grad():
            final_vel = vector_field(z)
            speeds = torch.norm(final_vel, p=2, dim=-1).cpu().numpy()
            locations = z.detach().cpu().numpy()

        # Filter and cluster unique points
        candidates: List[FixedPoint] = []
        for i in range(len(locations)):
            loc = locations[i]
            spd = float(speeds[i])
            is_fp = spd <= self.speed_tol
            is_sp = spd <= self.slow_tol

            if is_sp:
                candidates.append(
                    FixedPoint(
                        location=loc,
                        speed=spd,
                        is_true_fixed_point=is_fp,
                        is_slow_point=is_sp,
                    )
                )

        # Cluster duplicates within cluster_dist
        clustered: List[FixedPoint] = []
        for cand in candidates:
            is_duplicate = False
            for existing in clustered:
                dist = np.linalg.norm(cand.location - existing.location)
                if dist < self.cluster_dist:
                    is_duplicate = True
                    break
            if not is_duplicate:
                clustered.append(cand)

        return clustered
