"""Gauge alignment, orthogonal Procrustes analysis, and affine subspace transformations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import torch


@dataclass(frozen=True)
class ProcrustesAlignmentResult:
    """Result of orthogonal Procrustes alignment."""

    R: np.ndarray             # [D, D] orthogonal rotation/reflection matrix
    scale: float              # Optimal isotropic scaling factor
    translation: np.ndarray   # [D] translation vector
    disparity: float          # Normalized Frobenius residual ||Z1_aligned - Z2||_F / ||Z2||_F


class OrthogonalProcrustesAligner:
    """Finds optimal orthogonal transformation between latent trajectories Z1 and Z2.

    Objective:
        min_{R, s, t} || s * (Z1 - mean(Z1)) * R + t - Z2 ||_F^2
        subject to R^T * R = I_D

    Solved via Singular Value Decomposition (SVD) of cross-covariance matrix M = Z1_c^T * Z2_c.
    Strictly follows train-only fitting: transformation is computed on training trials,
    then applied out-of-sample to held-out evaluation trials without refitting.
    """

    def __init__(self, allow_reflection: bool = True):
        self.allow_reflection = allow_reflection
        self.R_: Optional[np.ndarray] = None
        self.scale_: float = 1.0
        self.mu1_: Optional[np.ndarray] = None
        self.mu2_: Optional[np.ndarray] = None

    def fit(self, Z1: np.ndarray, Z2: np.ndarray) -> OrthogonalProcrustesAligner:
        """Fit optimal orthogonal alignment from Z1 onto target Z2.

        Args:
            Z1: Source trajectory array of shape [T_total, D] or [B, T, D].
            Z2: Target trajectory array of identical shape.

        Returns:
            Fitted aligner instance.
        """
        Z1_flat = Z1.reshape(-1, Z1.shape[-1])
        Z2_flat = Z2.reshape(-1, Z2.shape[-1])

        if Z1_flat.shape != Z2_flat.shape:
            raise ValueError(f"Shape mismatch: Z1 {Z1_flat.shape} != Z2 {Z2_flat.shape}")

        N, D = Z1_flat.shape
        self.mu1_ = Z1_flat.mean(axis=0)
        self.mu2_ = Z2_flat.mean(axis=0)

        Z1_c = Z1_flat - self.mu1_
        Z2_c = Z2_flat - self.mu2_

        norm1 = np.linalg.norm(Z1_c)
        norm2 = np.linalg.norm(Z2_c)

        if norm1 < 1e-12 or norm2 < 1e-12:
            self.R_ = np.eye(D)
            self.scale_ = 1.0
            return self

        # Cross-covariance matrix M = Z1_c^T * Z2_c
        M = Z1_c.T @ Z2_c
        U, S, Vt = np.linalg.svd(M)
        R = U @ Vt

        # Handle reflection check if requested
        if not self.allow_reflection and np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = U @ Vt

        self.R_ = R
        # Optimal scale s = tr(Z1_c R Z2_c^T) / ||Z1_c||^2 = sum(S) / ||Z1_c||^2
        self.scale_ = float(np.sum(S) / (norm1 ** 2))
        return self

    def transform(self, Z: np.ndarray) -> np.ndarray:
        """Apply fitted alignment transformation to out-of-sample trajectory Z.

        Args:
            Z: Trajectory array of shape [..., D].

        Returns:
            Aligned trajectory array.
        """
        if self.R_ is None or self.mu1_ is None or self.mu2_ is None:
            raise RuntimeError("Aligner must be fitted before calling transform")

        orig_shape = Z.shape
        Z_flat = Z.reshape(-1, orig_shape[-1])
        Z_c = Z_flat - self.mu1_
        Z_aligned = self.scale_ * (Z_c @ self.R_) + self.mu2_
        return Z_aligned.reshape(orig_shape)

    def evaluate(self, Z1: np.ndarray, Z2: np.ndarray) -> ProcrustesAlignmentResult:
        """Fit on provided trajectories and return structured Procrustes alignment metrics.

        Args:
            Z1: Source trajectory array [..., D].
            Z2: Target trajectory array [..., D].

        Returns:
            ProcrustesAlignmentResult with R, scale, translation, and normalized disparity.
        """
        self.fit(Z1, Z2)
        Z1_aligned = self.transform(Z1)
        residual = np.linalg.norm(Z1_aligned - Z2)
        target_norm = np.linalg.norm(Z2)
        disparity = float(residual / max(1e-12, target_norm))

        return ProcrustesAlignmentResult(
            R=self.R_.copy(),
            scale=self.scale_,
            translation=self.mu2_ - self.scale_ * (self.mu1_ @ self.R_),
            disparity=disparity,
        )
