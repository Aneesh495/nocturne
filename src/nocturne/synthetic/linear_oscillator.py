"""Regime 1: Damped linear harmonic oscillator circuit with known analytic eigenspectrum."""

from __future__ import annotations

import math
from typing import Optional, Tuple
import numpy as np
import scipy.linalg

from nocturne.synthetic.base import SyntheticCircuit


class LinearDampedOscillator(SyntheticCircuit):
    """Controlled 2D linear damped oscillator: dz/dt = A * z + B * u.

    Dynamics:
        dz1/dt = -gamma * z1 + omega * z2
        dz2/dt = -omega * z1 - gamma * z2

    Analytic properties:
        Transition matrix A = [[-gamma, omega], [-omega, -gamma]]
        Eigenvalues: lambda_{1,2} = -gamma +/- i * omega
        Oscillation frequency: f = omega / (2 * pi)
        Decay timescale: tau = 1 / gamma
        Fixed point: z* = [0, 0] (stable focus for gamma > 0)
    """

    def __init__(self, damping_gamma: float = 0.5, frequency_omega: float = 2.0 * math.pi):
        super().__init__(latent_dim=2, regime_name="linear_damped_oscillator")
        if damping_gamma < 0.0:
            raise ValueError(f"damping_gamma must be non-negative, got {damping_gamma}")
        self.damping_gamma = damping_gamma
        self.frequency_omega = frequency_omega

    @property
    def system_matrix(self) -> np.ndarray:
        """Return analytic 2x2 transition matrix A."""
        g = self.damping_gamma
        w = self.frequency_omega
        return np.array([[-g, w], [-w, -g]], dtype=np.float64)

    @property
    def analytic_eigenvalues(self) -> Tuple[complex, complex]:
        """Return analytic complex conjugate eigenvalues."""
        g = self.damping_gamma
        w = self.frequency_omega
        return complex(-g, w), complex(-g, -w)

    def vector_field(self, z: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
        """Evaluate dz/dt = A * z (+ u if present)."""
        A = self.system_matrix
        dz = A @ z
        if u is not None:
            dz = dz + u[:2]
        return dz

    def analytic_solution(self, z0: np.ndarray, t: float | np.ndarray) -> np.ndarray:
        """Evaluate closed-form exact solution z(t) = exp(A * t) * z0."""
        z0_arr = np.asarray(z0, dtype=np.float64)
        g = self.damping_gamma
        w = self.frequency_omega

        if isinstance(t, (int, float, np.floating)):
            decay = math.exp(-g * t)
            rot = np.array([[math.cos(w * t), math.sin(w * t)], [-math.sin(w * t), math.cos(w * t)]])
            return decay * (rot @ z0_arr)

        t_arr = np.asarray(t, dtype=np.float64)
        decay = np.exp(-g * t_arr)[:, None]  # [T, 1]
        c = np.cos(w * t_arr)
        s = np.sin(w * t_arr)
        z1 = decay[:, 0] * (c * z0_arr[0] + s * z0_arr[1])
        z2 = decay[:, 0] * (-s * z0_arr[0] + c * z0_arr[1])
        return np.column_stack([z1, z2])

    def jacobian_at(self, z: np.ndarray) -> np.ndarray:
        """Jacobian df/dz is constant and equals system matrix A everywhere."""
        return self.system_matrix
