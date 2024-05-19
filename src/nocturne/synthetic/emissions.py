"""Neural emission models mapping latent states to positive firing rates and spike counts."""

from __future__ import annotations

import math
from typing import Optional, Tuple
import numpy as np

from nocturne.contracts.neurons import NeuronMetadata, NeuronRole, PopulationMetadata, UnitQuality
from nocturne.contracts.time import BinConfig


class SyntheticEmissionModel:
    """Readout model mapping latent vectors z in R^D to firing rates lambda in R^N and spikes."""

    def __init__(
        self,
        latent_dim: int,
        num_neurons: int,
        readout_matrix: Optional[np.ndarray] = None,
        bias_vector: Optional[np.ndarray] = None,
        link_function: str = "softplus",
        dispersion_theta: Optional[float] = None,
        seed: int = 1234,
    ):
        self.latent_dim = latent_dim
        self.num_neurons = num_neurons
        self.link_function = link_function
        self.dispersion_theta = dispersion_theta  # None for Poisson, float > 0 for Negative Binomial

        rng = np.random.default_rng(seed)
        if readout_matrix is not None:
            if readout_matrix.shape != (num_neurons, latent_dim):
                raise ValueError(f"readout_matrix shape must be ({num_neurons}, {latent_dim})")
            self.C = np.asarray(readout_matrix, dtype=np.float64)
        else:
            self.C = rng.normal(0.0, 1.0 / math.sqrt(latent_dim), size=(num_neurons, latent_dim))

        if bias_vector is not None:
            if bias_vector.shape != (num_neurons,):
                raise ValueError(f"bias_vector shape must be ({num_neurons},)")
            self.d = np.asarray(bias_vector, dtype=np.float64)
        else:
            # Baseline log-rates centered around ~15 Hz
            self.d = rng.normal(math.log(15.0), 0.5, size=(num_neurons,))

    def compute_rates_hz(self, states: np.ndarray) -> np.ndarray:
        """Map latent trajectory [T, D] to firing rates [T, N] in Hz."""
        logits = states @ self.C.T + self.d[None, :]  # [T, N]

        if self.link_function == "softplus":
            # Numerically stable softplus: ln(1 + exp(x))
            large = logits > 20.0
            rates = np.empty_like(logits)
            rates[large] = logits[large]
            rates[~large] = np.log1p(np.exp(logits[~large]))
        elif self.link_function == "exp":
            rates = np.exp(np.clip(logits, -10.0, 7.0))
        else:
            raise ValueError(f"Unsupported link_function: {self.link_function}")

        return np.maximum(rates, 1e-4)

    def sample_spikes(
        self,
        rates_hz: np.ndarray,
        bin_width_s: float,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Sample integer spike counts given firing rates and bin width.

        If dispersion_theta is specified, samples from Negative Binomial to simulate overdispersion.
        Otherwise samples from standard Poisson point process.
        """
        if rng is None:
            rng = np.random.default_rng()

        mu = rates_hz * bin_width_s  # Expected counts per bin

        if self.dispersion_theta is None:
            # Poisson counts
            return rng.poisson(mu).astype(np.int64)

        # Negative binomial parameterization: Var = mu + mu^2 / theta
        theta = self.dispersion_theta
        prob = theta / (theta + mu)
        return rng.negative_binomial(theta, prob).astype(np.int64)

    def create_population_metadata(self, session_id: str) -> PopulationMetadata:
        """Construct synthetic PopulationMetadata corresponding to this readout."""
        neurons = []
        num_held_in = int(round(self.num_neurons * 0.7))
        for i in range(self.num_neurons):
            role = NeuronRole.HELD_IN if i < num_held_in else NeuronRole.HELD_OUT
            rate = float(math.exp(self.d[i])) if self.link_function == "exp" else float(math.log1p(math.exp(self.d[i])))
            neurons.append(
                NeuronMetadata(
                    neuron_id=f"synth_{session_id}_{i:03d}",
                    channel_index=i,
                    brain_area="SyntheticCortex",
                    quality=UnitQuality.SUA,
                    role=role,
                    mean_firing_rate_hz=rate,
                )
            )
        return PopulationMetadata(session_id=session_id, neurons=tuple(neurons))
