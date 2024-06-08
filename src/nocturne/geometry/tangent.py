"""Local tangent space alignment and Grassmannian subspace drift analysis.

Estimates intrinsic manifold dimensionality via local PCA participation ratios
and measures differential subspace rotation across neural latent trajectories.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch


@dataclass(frozen=True)
class TangentSpaceResult:
    """Quantitative local tangent space metrics along a latent trajectory."""

    mean_intrinsic_dim: float
    min_intrinsic_dim: float
    max_intrinsic_dim: float
    mean_grassmann_drift: float
    max_grassmann_drift: float
    participation_ratios: List[float]
    grassmann_distances: List[float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert result container to dictionary."""
        return asdict(self)


class TangentSpaceEstimator:
    """Estimates local tangent spaces and manifold dimensionality along latent paths."""

    def __init__(self, window_size: int = 5, k_dim: int = 2, eps: float = 1e-9):
        if window_size < 2:
            raise ValueError(f"window_size must be >= 2, got {window_size}")
        if k_dim < 1:
            raise ValueError(f"k_dim must be >= 1, got {k_dim}")
        self.window_size = window_size
        self.k_dim = k_dim
        self.eps = eps

    def analyze_trajectory(
        self,
        trajectory: np.ndarray | torch.Tensor,
    ) -> TangentSpaceResult:
        """Estimate tangent planes, participation ratios, and Grassmann drift.

        Args:
            trajectory: Latent state trajectory array of shape [T, d].

        Returns:
            TangentSpaceResult with local dimensionality and subspace rotation metrics.
        """
        if isinstance(trajectory, torch.Tensor):
            z = trajectory.detach().cpu().numpy().astype(np.float64)
        else:
            z = np.asarray(trajectory, dtype=np.float64)

        if z.ndim == 3:
            if z.shape[0] != 1:
                raise ValueError(f"Expected single trajectory [T, d], got batched {z.shape}")
            z = z[0]

        T, d = z.shape
        w = self.window_size
        k = min(self.k_dim, d)

        if T < 2 * w + 1:
            raise ValueError(f"Trajectory length T={T} too short for window {w} (need >= {2*w+1})")

        prs: List[float] = []
        bases: List[np.ndarray] = []  # List of [d, k] orthonormal bases

        for i in range(w, T - w):
            local_pts = z[i - w : i + w + 1]  # [2w+1, d]
            centered = local_pts - np.mean(local_pts, axis=0, keepdims=True)

            # Covariance matrix: [d, d]
            cov = (centered.T @ centered) / (len(local_pts) - 1.0)
            eigvals, eigvecs = np.linalg.eigh(cov)

            # Sort descending
            idx = np.argsort(eigvals)[::-1]
            eigvals = np.maximum(0.0, eigvals[idx])
            eigvecs = eigvecs[:, idx]

            # Participation ratio: (sum s_j)^2 / sum(s_j^2)
            sum_s = np.sum(eigvals)
            sum_s2 = np.sum(eigvals**2)
            pr = float((sum_s**2) / (sum_s2 + self.eps)) if sum_s2 > 0 else 1.0
            prs.append(pr)

            # Top-k tangent basis
            top_basis = eigvecs[:, :k]  # [d, k]
            bases.append(top_basis)

        # Compute Grassmannian distance between consecutive tangent spaces
        # d_G(U, V) = sqrt(sum arccos(sigma_m)^2) where sigma_m are singular values of U^T V
        grassmann_dists: List[float] = []
        for j in range(len(bases) - 1):
            U = bases[j]
            V = bases[j + 1]
            M = U.T @ V  # [k, k]
            # Singular values represent cosines of principal angles
            sv = np.linalg.svd(M, compute_uv=False)
            sv = np.clip(sv, 0.0, 1.0)
            principal_angles = np.arccos(sv)
            dist = float(np.sqrt(np.sum(principal_angles**2)))
            grassmann_dists.append(dist)

        mean_pr = float(np.mean(prs))
        min_pr = float(np.min(prs))
        max_pr = float(np.max(prs))
        mean_drift = float(np.mean(grassmann_dists)) if grassmann_dists else 0.0
        max_drift = float(np.max(grassmann_dists)) if grassmann_dists else 0.0

        return TangentSpaceResult(
            mean_intrinsic_dim=mean_pr,
            min_intrinsic_dim=min_pr,
            max_intrinsic_dim=max_pr,
            mean_grassmann_drift=mean_drift,
            max_grassmann_drift=max_drift,
            participation_ratios=prs,
            grassmann_distances=grassmann_dists,
        )
