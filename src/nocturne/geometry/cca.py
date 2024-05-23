"""Canonical Correlation Analysis (CCA) between neural latent representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class CCAResult:
    """Outcome of Canonical Correlation Analysis between two representations."""

    canonical_correlations: np.ndarray  # [K] canonical correlation coefficients in [0, 1]
    mean_correlation: float             # Mean of top canonical correlations
    Wx: np.ndarray                      # [D1, K] projection directions for X
    Wy: np.ndarray                      # [D2, K] projection directions for Y


class CanonicalCorrelationAnalysis:
    """Finds linear combinations of X and Y that maximize mutual correlation.

    Solved via regularized whitening and Singular Value Decomposition (SVD):
        K = (Sigma_xx + eps * I)^{-1/2} * Sigma_xy * (Sigma_yy + eps * I)^{-1/2}
        K = U * diag(rho) * V^T
    """

    def __init__(self, n_components: Optional[int] = None, reg: float = 1e-4):
        self.n_components = n_components
        self.reg = reg

    def fit(self, X: np.ndarray, Y: np.ndarray) -> CCAResult:
        """Fit regularized CCA on observations X and Y.

        Args:
            X: Array of shape [N_samples, D1].
            Y: Array of shape [N_samples, D2].

        Returns:
            CCAResult containing canonical correlations and projection weights.
        """
        X_flat = X.reshape(-1, X.shape[-1]).astype(np.float64)
        Y_flat = Y.reshape(-1, Y.shape[-1]).astype(np.float64)

        if X_flat.shape[0] != Y_flat.shape[0]:
            raise ValueError(
                f"Sample count mismatch: X has {X_flat.shape[0]}, Y has {Y_flat.shape[0]}"
            )

        N, D1 = X_flat.shape
        _, D2 = Y_flat.shape
        K = min(D1, D2) if self.n_components is None else min(self.n_components, D1, D2)

        # Center matrices
        X_c = X_flat - X_flat.mean(axis=0)
        Y_c = Y_flat - Y_flat.mean(axis=0)

        # Sample covariances
        Cxx = (X_c.T @ X_c) / (N - 1) + self.reg * np.eye(D1)
        Cyy = (Y_c.T @ Y_c) / (N - 1) + self.reg * np.eye(D2)
        Cxy = (X_c.T @ Y_c) / (N - 1)

        # Compute inverse square roots via eigendecomposition
        def inv_sqrt_matrix(M: np.ndarray) -> np.ndarray:
            eigvals, eigvecs = np.linalg.eigh(M)
            eigvals = np.clip(eigvals, a_min=1e-12, a_max=None)
            return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        Cxx_inv_sqrt = inv_sqrt_matrix(Cxx)
        Cyy_inv_sqrt = inv_sqrt_matrix(Cyy)

        # Whitened cross-covariance
        M = Cxx_inv_sqrt @ Cxy @ Cyy_inv_sqrt
        U, s, Vt = np.linalg.svd(M)

        corrs = np.clip(s[:K], 0.0, 1.0)
        Wx = Cxx_inv_sqrt @ U[:, :K]
        Wy = Cyy_inv_sqrt @ Vt.T[:, :K]

        mean_corr = float(np.mean(corrs)) if len(corrs) > 0 else 0.0

        return CCAResult(
            canonical_correlations=corrs,
            mean_correlation=mean_corr,
            Wx=Wx,
            Wy=Wy,
        )
