"""Online recursive point-process Extended Kalman Filter (PPEKF) for neural decoders.

Implements real-time Gaussian-approximation point-process state estimation
based on Eden et al. (2004) and Brown et al. (1998).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch


@dataclass(frozen=True)
class PointProcessKalmanState:
    """Estimated latent state, covariance, and rate output at a single time step."""

    latent_mean: np.ndarray        # [d]
    latent_covariance: np.ndarray  # [d, d]
    predicted_rates_hz: np.ndarray # [N]
    innovation: np.ndarray         # [N] difference between observed and expected counts


class PointProcessEKF:
    """Recursive online point-process Extended Kalman Filter.

    Maintains Gaussian posterior approximation p(x_t | y_{1:t}) ~ N(x_t, P_t)
    updated incrementally upon arrival of discrete spike count vector y_t.
    """

    def __init__(
        self,
        latent_dim: int,
        num_channels: int,
        transition_matrix: Optional[np.ndarray] = None,
        process_noise_cov: Optional[np.ndarray] = None,
        emission_weights: Optional[np.ndarray] = None,
        emission_biases: Optional[np.ndarray] = None,
        bin_width_s: float = 0.02,
    ):
        self.d = latent_dim
        self.N = num_channels
        self.bin_width_s = bin_width_s

        # Latent transition: x_{t} = A x_{t-1} + w_t, w_t ~ N(0, Q)
        if transition_matrix is not None:
            self.A = np.asarray(transition_matrix, dtype=np.float64)
        else:
            self.A = np.eye(self.d, dtype=np.float64) * 0.98

        if process_noise_cov is not None:
            self.Q = np.asarray(process_noise_cov, dtype=np.float64)
        else:
            self.Q = np.eye(self.d, dtype=np.float64) * 0.05

        # Emission mapping: lambda_n(t) = exp(C_n^T x_t + b_n)
        if emission_weights is not None:
            self.C = np.asarray(emission_weights, dtype=np.float64)
        else:
            self.C = np.random.randn(self.N, self.d) * 0.1

        if emission_biases is not None:
            self.b = np.asarray(emission_biases, dtype=np.float64)
        else:
            self.b = np.zeros(self.N, dtype=np.float64)

        # Initial state prior: x_0 ~ N(0, I)
        self.x = np.zeros(self.d, dtype=np.float64)
        self.P = np.eye(self.d, dtype=np.float64)

    def reset(
        self,
        initial_mean: Optional[np.ndarray] = None,
        initial_cov: Optional[np.ndarray] = None,
    ) -> None:
        """Reset internal filter state to prior parameters."""
        self.x = (
            np.asarray(initial_mean, dtype=np.float64).copy()
            if initial_mean is not None
            else np.zeros(self.d, dtype=np.float64)
        )
        self.P = (
            np.asarray(initial_cov, dtype=np.float64).copy()
            if initial_cov is not None
            else np.eye(self.d, dtype=np.float64)
        )

    def step(self, observed_counts: np.ndarray | torch.Tensor) -> PointProcessKalmanState:
        """Execute one recursive causal filter update step given observed spike counts.

        Args:
            observed_counts: 1D array of integer spike counts of shape [N].

        Returns:
            PointProcessKalmanState containing updated posterior mean, covariance, and rates.
        """
        if isinstance(observed_counts, torch.Tensor):
            y = observed_counts.detach().cpu().numpy().astype(np.float64)
        else:
            y = np.asarray(observed_counts, dtype=np.float64)

        if y.ndim != 1 or len(y) != self.N:
            raise ValueError(f"Expected 1D count vector of length {self.N}, got shape {y.shape}")

        dt = self.bin_width_s

        # 1. Prediction step: prior state and covariance
        x_prior = self.A @ self.x
        P_prior = (self.A @ self.P @ self.A.T) + self.Q

        # 2. Prior expected firing rate: lambda_n = exp(C_n^T x_prior + b_n)
        log_rates = (self.C @ x_prior) + self.b
        # Clamping to avoid numerical overflow
        log_rates = np.clip(log_rates, -10.0, 10.0)
        rates_hz = np.exp(log_rates)
        expected_counts = rates_hz * dt

        # Innovation: residual between observed and expected counts
        innov = y - expected_counts

        # 3. Measurement update via Laplace / Newton-Raphson approximation
        # P_post^{-1} = P_prior^{-1} + sum_n C_n C_n^T lambda_n dt
        # Vectorized Hessian: C^T diag(lambda * dt) C
        Hessian_obs = (self.C.T * expected_counts) @ self.C  # [d, d]

        # Information form inversion
        P_prior_inv = np.linalg.inv(P_prior)
        P_post_inv = P_prior_inv + Hessian_obs
        P_post = np.linalg.inv(P_post_inv)

        # Posterior mean: x_post = x_prior + P_post * sum_n C_n [y_n - lambda_n dt]
        # Gradient of log-likelihood: C^T (y - lambda * dt)
        grad_obs = self.C.T @ innov  # [d]
        x_post = x_prior + (P_post @ grad_obs)

        # Update filter state
        self.x = x_post
        self.P = P_post

        # Recompute updated rates at posterior mean
        post_rates_hz = np.exp(np.clip((self.C @ x_post) + self.b, -10.0, 10.0))

        return PointProcessKalmanState(
            latent_mean=x_post.copy(),
            latent_covariance=P_post.copy(),
            predicted_rates_hz=post_rates_hz,
            innovation=innov,
        )

    def filter_sequence(
        self,
        counts_sequence: np.ndarray | torch.Tensor,
    ) -> List[PointProcessKalmanState]:
        """Filter an entire sequence of spike counts causally.

        Args:
            counts_sequence: Tensor or array of shape [T, N].

        Returns:
            List of PointProcessKalmanState for all T time steps.
        """
        if isinstance(counts_sequence, torch.Tensor):
            arr = counts_sequence.detach().cpu().numpy().astype(np.float64)
        else:
            arr = np.asarray(counts_sequence, dtype=np.float64)

        if arr.ndim != 2 or arr.shape[1] != self.N:
            raise ValueError(f"Expected [T, N] sequence with N={self.N}, got {arr.shape}")

        states: List[PointProcessKalmanState] = []
        for t in range(arr.shape[0]):
            state = self.step(arr[t])
            states.append(state)

        return states
