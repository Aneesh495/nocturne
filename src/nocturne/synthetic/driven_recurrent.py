"""Regime 4: Driven continuous-time recurrent neural network with input-dependent flow fields."""

from __future__ import annotations

import math
from typing import Optional, Tuple
import numpy as np

from nocturne.synthetic.base import SyntheticCircuit


class DrivenRecurrentCircuit(SyntheticCircuit):
    """Continuous-time recurrent neural network (CTRNN) with external driving inputs.

    Dynamics:
        tau * dz/dt = -z + W * tanh(z) + B * u(t)

    Analytic properties:
        Jacobian matrix: J(z) = (1 / tau) * [ -I + W * diag(1 - tanh^2(z)) ]
        External inputs u(t) perturb the flow field and drive transitions between low-velocity manifolds.
        Explicitly designed to contrast autonomous flow from input-driven state transitions.
    """

    def __init__(
        self,
        latent_dim: int = 4,
        input_dim: int = 2,
        tau: float = 0.05,
        spectral_radius: float = 1.2,
        seed: int = 42,
    ):
        super().__init__(latent_dim=latent_dim, regime_name="driven_recurrent_circuit")
        if tau <= 0.0:
            raise ValueError(f"tau must be strictly positive, got {tau}")
        self.input_dim = input_dim
        self.tau = tau

        # Generate orthogonal/scaled recurrent weight matrix
        rng = np.random.default_rng(seed)
        W_raw = rng.normal(0.0, 1.0 / math.sqrt(latent_dim), size=(latent_dim, latent_dim))
        current_sr = np.max(np.abs(np.linalg.eigvals(W_raw)))
        self.W = (spectral_radius / (current_sr + 1e-8)) * W_raw
        self.B = rng.normal(0.0, 1.0, size=(latent_dim, input_dim))

    def vector_field(self, z: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
        """Evaluate (1/tau) * (-z + W * tanh(z) + B * u)."""
        act = np.tanh(z)
        drive = self.W @ act
        if u is not None and len(u) >= self.input_dim:
            drive = drive + self.B @ u[: self.input_dim]
        return (1.0 / self.tau) * (-z + drive)

    def jacobian_at(self, z: np.ndarray) -> np.ndarray:
        """Analytic Jacobian J(z) = (1 / tau) * [ -I + W * diag(1 - tanh^2(z)) ]."""
        d_act = 1.0 - (np.tanh(z) ** 2)
        # W * diag(d_act) multiplies each column j of W by d_act[j]
        W_scaled = self.W * d_act[None, :]
        eye = np.eye(self.latent_dim, dtype=np.float64)
        return (1.0 / self.tau) * (-eye + W_scaled)
