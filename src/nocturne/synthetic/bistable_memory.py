"""Regime 3: Bistable double-well memory circuit with discrete attractors and input switching."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple
import numpy as np

from nocturne.synthetic.base import SyntheticCircuit


class BistableMemoryCircuit(SyntheticCircuit):
    """Two-dimensional bistable double-well circuit modeling working memory persistence and switching.

    Dynamics:
        dz1/dt = alpha * (z1 - z1^3) + u1(t)
        dz2/dt = -decay2 * z2 + u2(t)

    Analytic properties:
        Three fixed points for u = 0:
            1. Stable attractor A: z* = (+1.0, 0.0), J = [[-2*alpha, 0], [0, -decay2]]
            2. Stable attractor B: z* = (-1.0, 0.0), J = [[-2*alpha, 0], [0, -decay2]]
            3. Saddle point:      z* = (0.0, 0.0),  J = [[+alpha, 0], [0, -decay2]]
        Switching threshold: A positive or negative input pulse in u1 pushes the trajectory
        across the separatrix at z1 = 0 into the alternate attractor basin.
    """

    def __init__(self, alpha: float = 2.0, decay2: float = 1.0):
        super().__init__(latent_dim=2, regime_name="bistable_memory_circuit")
        if alpha <= 0.0 or decay2 <= 0.0:
            raise ValueError("alpha and decay2 must be strictly positive")
        self.alpha = alpha
        self.decay2 = decay2

    @property
    def stable_fixed_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the two stable attractor states."""
        return np.array([1.0, 0.0], dtype=np.float64), np.array([-1.0, 0.0], dtype=np.float64)

    @property
    def saddle_fixed_point(self) -> np.ndarray:
        """Return the unstable saddle fixed point on the separatrix."""
        return np.array([0.0, 0.0], dtype=np.float64)

    def vector_field(self, z: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
        """Evaluate dz/dt for bistable memory circuit."""
        z1, z2 = z[0], z[1]
        dz1 = self.alpha * (z1 - z1**3)
        dz2 = -self.decay2 * z2

        dz = np.array([dz1, dz2], dtype=np.float64)
        if u is not None:
            dz = dz + u[:2]
        return dz

    def jacobian_at(self, z: np.ndarray) -> np.ndarray:
        """Analytic Jacobian df/dz at state z."""
        z1 = z[0]
        # d(dz1)/dz1 = alpha * (1 - 3 * z1^2)
        j11 = self.alpha * (1.0 - 3.0 * (z1**2))
        j12 = 0.0
        j21 = 0.0
        j22 = -self.decay2
        return np.array([[j11, j12], [j21, j22]], dtype=np.float64)

    def potential(self, z1: float | np.ndarray) -> float | np.ndarray:
        """Double-well potential V(z1) = alpha * (z1^4 / 4 - z1^2 / 2)."""
        return self.alpha * (0.25 * (z1**4) - 0.5 * (z1**2))
