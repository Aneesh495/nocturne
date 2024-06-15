"""Information-theoretic estimators for neural population representations.

Computes mutual information between latent dynamics and continuous behavioral variables
via Kraskov (KSG) k-NN non-parametric estimation and Gaussian analytic bounds.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.spatial import KDTree
from scipy.special import digamma
import torch


@dataclass(frozen=True)
class MutualInformationResult:
    """Quantitative mutual information estimate between latents and behavior."""

    mutual_information_bits: float
    gaussian_linear_bits: float
    non_linear_excess_bits: float
    sample_size: int
    k_neighbors: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert result container to dictionary."""
        return asdict(self)


class MutualInformationEstimator:
    """Estimates mutual information I(Z; X) between neural latents and behavior."""

    def __init__(self, k_neighbors: int = 4, eps: float = 1e-9):
        if k_neighbors < 1:
            raise ValueError(f"k_neighbors must be >= 1, got {k_neighbors}")
        self.k_neighbors = k_neighbors
        self.eps = eps

    def compute_gaussian_mi(
        self,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> float:
        """Compute analytic Gaussian mutual information in bits.

        Formula:
            I_Gauss = 0.5 * log2( det(Sigma_XX) * det(Sigma_YY) / det(Sigma) )
        """
        N = X.shape[0]
        # Concatenate joint matrix
        XY = np.hstack([X, Y])  # [N, dX + dY]

        # Centered covariances with regularization
        cov_X = np.cov(X, rowvar=False) + np.eye(X.shape[1]) * self.eps
        cov_Y = np.cov(Y, rowvar=False) + np.eye(Y.shape[1]) * self.eps
        cov_XY = np.cov(XY, rowvar=False) + np.eye(XY.shape[1]) * self.eps

        # Slogdet for numerical stability
        sign_x, logdet_x = np.linalg.slogdet(cov_X)
        sign_y, logdet_y = np.linalg.slogdet(cov_Y)
        sign_xy, logdet_xy = np.linalg.slogdet(cov_XY)

        if sign_x <= 0 or sign_y <= 0 or sign_xy <= 0:
            return 0.0

        # In natural nats: 0.5 * (logdet_x + logdet_y - logdet_xy)
        nats = 0.5 * (logdet_x + logdet_y - logdet_xy)
        bits = max(0.0, float(nats / math.log(2.0)))
        return bits

    def compute_ksg_mi(
        self,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> float:
        """Compute non-parametric KSG mutual information (Kraskov et al. 2004) in bits.

        Algorithm 1:
            I(X, Y) = psi(k) - (1/N) * sum_i [psi(n_{x,i} + 1) + psi(n_{y,i} + 1)] + psi(N)
        """
        N = X.shape[0]
        k = min(self.k_neighbors, N - 1)
        if k < 1:
            return 0.0

        # Build KD-trees with Chebyshev (L_infinity) norm
        tree_xy = KDTree(np.hstack([X, Y]))
        tree_x = KDTree(X)
        tree_y = KDTree(Y)

        # Find distance to k-th nearest neighbor in joint space
        # query k+1 neighbors because point 0 is the query point itself
        dists, _ = tree_xy.query(np.hstack([X, Y]), k=k + 1, p=np.inf)
        eps_k = dists[:, -1]  # Distance to k-th neighbor (excluding self)

        # Count points strictly within eps_k in marginal spaces (subtract 1 to exclude self)
        # Using query_ball_point with radius eps_k - 1e-15 to match strict inequality
        nx = np.zeros(N, dtype=np.int64)
        ny = np.zeros(N, dtype=np.int64)

        for i in range(N):
            r = eps_k[i]
            # Strict inequality: radius strictly less than r
            r_strict = max(0.0, r - 1e-12)
            nx[i] = len(tree_x.query_ball_point(X[i], r=r_strict, p=np.inf)) - 1
            ny[i] = len(tree_y.query_ball_point(Y[i], r=r_strict, p=np.inf)) - 1

        # Clip counts to non-negative
        nx = np.maximum(0, nx)
        ny = np.maximum(0, ny)

        # KSG Formula in nats
        nats = (
            digamma(k)
            - float(np.mean(digamma(nx + 1) + digamma(ny + 1)))
            + digamma(N)
        )
        bits = max(0.0, float(nats / math.log(2.0)))
        return bits

    def estimate(
        self,
        latents: np.ndarray | torch.Tensor,
        behavior: np.ndarray | torch.Tensor,
    ) -> MutualInformationResult:
        """Estimate mutual information between latents and behavior in bits.

        Args:
            latents: Latent state array [..., d] or [B, T, d].
            behavior: Behavioral variables array [..., K] or [B, T, K].

        Returns:
            MutualInformationResult with KSG bits and linear Gaussian bounds.
        """
        if isinstance(latents, torch.Tensor):
            Z = latents.detach().cpu().numpy().astype(np.float64)
        else:
            Z = np.asarray(latents, dtype=np.float64)

        if isinstance(behavior, torch.Tensor):
            X = behavior.detach().cpu().numpy().astype(np.float64)
        else:
            X = np.asarray(behavior, dtype=np.float64)

        # Flatten trial and time steps to [N_samples, Dim]
        if Z.ndim > 2:
            Z = Z.reshape(-1, Z.shape[-1])
        if X.ndim > 2:
            X = X.reshape(-1, X.shape[-1])

        if Z.shape[0] != X.shape[0]:
            raise ValueError(f"Sample count mismatch: Z has {Z.shape[0]}, X has {X.shape[0]}")

        N = Z.shape[0]
        if N < self.k_neighbors + 2:
            raise ValueError(f"Need at least {self.k_neighbors + 2} samples, got {N}")

        # Add tiny jitter to avoid exact duplicate distance ties
        rng = np.random.RandomState(42)
        Z_jitter = Z + 1e-10 * rng.randn(*Z.shape)
        X_jitter = X + 1e-10 * rng.randn(*X.shape)

        # Subsample if N > 2000 to keep KD-tree query fast
        if N > 2000:
            sub_idx = rng.choice(N, size=2000, replace=False)
            Z_eval = Z_jitter[sub_idx]
            X_eval = X_jitter[sub_idx]
        else:
            Z_eval = Z_jitter
            X_eval = X_jitter

        gauss_bits = self.compute_gaussian_mi(Z_eval, X_eval)
        ksg_bits = self.compute_ksg_mi(Z_eval, X_eval)
        excess = max(0.0, ksg_bits - gauss_bits)

        return MutualInformationResult(
            mutual_information_bits=ksg_bits,
            gaussian_linear_bits=gauss_bits,
            non_linear_excess_bits=excess,
            sample_size=Z_eval.shape[0],
            k_neighbors=self.k_neighbors,
        )
