"""Regime 2: Nonlinear Stuart-Landau limit cycle oscillator with known attractor geometry."""

from __future__ import annotations

import math
from typing import Optional, Tuple
import numpy as np

from nocturne.synthetic.base import SyntheticCircuit


class NonlinearLimitCycleOscillator(SyntheticCircuit):
    """Stuart-Landau nonlinear oscillator exhibiting an isolated stable limit cycle attractor.

    Dynamics in Cartesian coordinates z = (z1, z2):
        r^2 = z1^2 + z2^2
        dz1/dt = (mu - r^2) * z1 - (omega + b * r^2) * z2
        dz2/dt = (omega + b * r^2) * z1 + (mu - r^2) * z2

    Analytic properties:
        Bifurcation parameter mu > 0 gives supercritical Hopf bifurcation.
        Stable limit cycle radius: r* = sqrt(mu)
        Oscillation frequency on limit cycle: f* = (omega + b * mu) / (2 * pi)
        Origin (0, 0) is an unstable spiral fixed point (eigenvalues mu +/- i * omega).
        Radial Lyapunov exponent (Floquet multiplier exponent): lambda_r = -2 * mu.
    """

    def __init__(self, mu: float = 1.0, omega: float = 2.0 * math.pi, b: float = 0.5):
        super().__init__(latent_dim=2, regime_name="nonlinear_limit_cycle_oscillator")
        if mu <= 0.0:
            raise ValueError(f"Bifurcation parameter mu must be positive for limit cycle, got {mu}")
        self.mu = mu
        self.omega = omega
        self.b = b

    @property
    def limit_cycle_radius(self) -> float:
        """Analytic radius of the stable limit cycle manifold."""
        return math.sqrt(self.mu)

    @property
    def limit_cycle_frequency_hz(self) -> float:
        """Analytic rotation frequency on the limit cycle."""
        return (self.omega + self.b * self.mu) / (2.0 * math.pi)

    def vector_field(self, z: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
        """Evaluate nonlinear vector field dz/dt."""
        z1, z2 = z[0], z[1]
        r2 = z1**2 + z2**2
        radial = self.mu - r2
        angular = self.omega + self.b * r2

        dz1 = radial * z1 - angular * z2
        dz2 = angular * z1 + radial * z2

        dz = np.array([dz1, dz2], dtype=np.float64)
        if u is not None:
            dz = dz + u[:2]
        return dz

    def jacobian_at(self, z: np.ndarray) -> np.ndarray:
        """Analytic Jacobian df/dz evaluated at state z."""
        z1, z2 = z[0], z[1]
        r2 = z1**2 + z2**2
        mu, omega, b = self.mu, self.omega, self.b

        # dz1 = (mu - z1^2 - z2^2)*z1 - (omega + b*(z1^2 + z2^2))*z2
        # d(dz1)/dz1 = mu - 3*z1^2 - z2^2 - 2*b*z1*z2
        # d(dz1)/dz2 = -2*z1*z2 - (omega + 3*b*z2^2 + b*z1^2)
        j11 = mu - 3.0 * (z1**2) - (z2**2) - 2.0 * b * z1 * z2
        j12 = -2.0 * z1 * z2 - (omega + 3.0 * b * (z2**2) + b * (z1**2))

        # dz2 = (omega + b*(z1^2 + z2^2))*z1 + (mu - z1^2 - z2^2)*z2
        # d(dz2)/dz1 = (omega + 3*b*z1^2 + b*z2^2) - 2*z1*z2
        # d(dz2)/dz2 = 2*b*z1*z2 + mu - z1^2 - 3*z2^2
        j21 = (omega + 3.0 * b * (z1**2) + b * (z2**2)) - 2.0 * z1 * z2
        j22 = 2.0 * b * z1 * z2 + mu - (z1**2) - 3.0 * (z2**2)

        return np.array([[j11, j12], [j21, j22]], dtype=np.float64)
