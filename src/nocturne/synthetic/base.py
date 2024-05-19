"""Abstract synthetic circuit base class, numerical integrators, and simulation containers."""

from __future__ import annotations

import abc
import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from nocturne.contracts.neurons import NeuronMetadata, NeuronRole, PopulationMetadata, UnitQuality
from nocturne.contracts.time import BinConfig, TemporalGrid, TimeInterval
from nocturne.contracts.trials import TrialMetadata, TrialType


class LatentTrajectory(BaseModel):
    """Continuous latent dynamical state trajectory and ground-truth derivatives."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    timestamps: np.ndarray = Field(..., description="1D array of timestamps [T]")
    states: np.ndarray = Field(..., description="2D array of latent states [T, latent_dim]")
    derivatives: np.ndarray = Field(..., description="2D array of state derivatives dz/dt [T, latent_dim]")
    inputs: Optional[np.ndarray] = Field(default=None, description="External input drive [T, input_dim]")

    def model_post_init(self, __context: object) -> None:
        T, D = self.states.shape
        if self.timestamps.shape != (T,):
            raise ValueError(f"timestamps shape {self.timestamps.shape} != ({T},)")
        if self.derivatives.shape != (T, D):
            raise ValueError(f"derivatives shape {self.derivatives.shape} != ({T}, {D})")
        if self.inputs is not None and self.inputs.shape[0] != T:
            raise ValueError(f"inputs time dim {self.inputs.shape[0]} != {T}")

    @property
    def num_timepoints(self) -> int:
        return self.states.shape[0]

    @property
    def latent_dim(self) -> int:
        return self.states.shape[1]

    @property
    def input_dim(self) -> int:
        return 0 if self.inputs is None else self.inputs.shape[1]


class CircuitSimulation(BaseModel):
    """Complete simulation bundle containing ground-truth latents, firing rates, and spike counts."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    regime_name: str = Field(..., description="Dynamical regime identifier")
    trial_id: int = Field(..., ge=0, description="Trial number")
    latent: LatentTrajectory = Field(..., description="Privileged ground-truth latent trajectory")
    true_rates_hz: np.ndarray = Field(..., description="True instantaneous firing rates in Hz [T, num_neurons]")
    spike_counts: np.ndarray = Field(..., description="Binned discrete spike counts [T, num_neurons]")
    population: PopulationMetadata = Field(..., description="Synthetic neural population metadata")
    bin_config: BinConfig = Field(..., description="Temporal binning configuration")

    @property
    def num_bins(self) -> int:
        return self.spike_counts.shape[0]

    @property
    def num_neurons(self) -> int:
        return self.population.num_neurons

    @property
    def total_spikes(self) -> int:
        return int(np.sum(self.spike_counts))


class SyntheticCircuit(abc.ABC):
    """Abstract base class for controlled dynamical circuits with known vector fields."""

    def __init__(self, latent_dim: int, regime_name: str):
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {latent_dim}")
        self.latent_dim = latent_dim
        self.regime_name = regime_name

    @abc.abstractmethod
    def vector_field(self, z: np.ndarray, u: Optional[np.ndarray] = None) -> np.ndarray:
        """Evaluate deterministic velocity field dz/dt = f(z, u)."""
        pass

    def integrate_rk4(
        self,
        z0: np.ndarray,
        duration_s: float,
        dt_s: float = 0.001,
        input_fn: Optional[Callable[[float], np.ndarray]] = None,
    ) -> LatentTrajectory:
        """Simulate state trajectory using 4th-order Runge-Kutta numerical integration."""
        if dt_s <= 0.0 or duration_s <= 0.0:
            raise ValueError("dt_s and duration_s must be strictly positive")

        num_steps = int(math.ceil(duration_s / dt_s)) + 1
        timestamps = np.array([i * dt_s for i in range(num_steps)], dtype=np.float64)
        states = np.zeros((num_steps, self.latent_dim), dtype=np.float64)
        derivatives = np.zeros((num_steps, self.latent_dim), dtype=np.float64)

        in_dim = len(input_fn(0.0)) if input_fn is not None else 0
        inputs = np.zeros((num_steps, in_dim), dtype=np.float64) if in_dim > 0 else None

        current_z = np.asarray(z0, dtype=np.float64).copy()
        if current_z.shape != (self.latent_dim,):
            raise ValueError(f"z0 shape {current_z.shape} != ({self.latent_dim},)")

        for step in range(num_steps):
            t = timestamps[step]
            u = input_fn(t) if input_fn is not None else None
            if inputs is not None and u is not None:
                inputs[step] = u

            states[step] = current_z
            dz_dt = self.vector_field(current_z, u)
            derivatives[step] = dz_dt

            if step < num_steps - 1:
                # RK4 intermediate stages
                k1 = dz_dt
                u_mid = input_fn(t + 0.5 * dt_s) if input_fn is not None else None
                k2 = self.vector_field(current_z + 0.5 * dt_s * k1, u_mid)
                k3 = self.vector_field(current_z + 0.5 * dt_s * k2, u_mid)
                u_next = input_fn(t + dt_s) if input_fn is not None else None
                k4 = self.vector_field(current_z + dt_s * k3, u_next)

                current_z = current_z + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        return LatentTrajectory(
            timestamps=timestamps,
            states=states,
            derivatives=derivatives,
            inputs=inputs,
        )
