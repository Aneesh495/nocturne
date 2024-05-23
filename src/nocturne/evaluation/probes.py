"""Frozen train-only behavioral decoding probes mapping latents to kinematics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import torch


@dataclass(frozen=True)
class ProbeEvaluationResult:
    """Out-of-sample behavioral decoding evaluation summary."""

    macro_r2: float
    per_dim_r2: List[float]
    macro_pearson_r: float
    per_dim_pearson_r: List[float]
    mse: float


class BehavioralDecodingProbe:
    """Linear ridge regression decoder mapping latent representations to behavior.

    Strict Train-Only Contract:
        The probe parameters W are fit exclusively on training latents and behavior:
            W* = (Z_train^T * Z_train + lambda * I)^{-1} * Z_train^T * V_train
        W* is frozen and applied strictly out-of-sample to held-out test partitions.
    """

    def __init__(self, ridge_alpha: float = 1e-2):
        self.ridge_alpha = ridge_alpha
        self.W_: Optional[np.ndarray] = None
        self.b_: Optional[np.ndarray] = None

    def fit(self, latents_train: np.ndarray, behavior_train: np.ndarray) -> BehavioralDecodingProbe:
        """Fit linear probe strictly on training partition.

        Args:
            latents_train: Array of shape [B_train, T, D] or [N_train_samples, D].
            behavior_train: Array of shape [B_train, T, K] or [N_train_samples, K].

        Returns:
            Fitted probe instance.
        """
        Z = latents_train.reshape(-1, latents_train.shape[-1]).astype(np.float64)
        V = behavior_train.reshape(-1, behavior_train.shape[-1]).astype(np.float64)

        if Z.shape[0] != V.shape[0]:
            raise ValueError(
                f"Sample count mismatch: latents {Z.shape[0]} != behavior {V.shape[0]}"
            )

        # Center data
        mu_Z = Z.mean(axis=0)
        mu_V = V.mean(axis=0)
        Z_c = Z - mu_Z
        V_c = V - mu_V

        D = Z.shape[1]
        # Regularized normal equations
        reg_eye = self.ridge_alpha * np.eye(D)
        W = np.linalg.solve(Z_c.T @ Z_c + reg_eye, Z_c.T @ V_c)
        b = mu_V - mu_Z @ W

        self.W_ = W
        self.b_ = b
        return self

    def predict(self, latents: np.ndarray) -> np.ndarray:
        """Predict behavior from out-of-sample latents using frozen weights."""
        if self.W_ is None or self.b_ is None:
            raise RuntimeError("Probe must be fitted before predict")

        orig_shape = latents.shape
        Z = latents.reshape(-1, orig_shape[-1]).astype(np.float64)
        V_pred = Z @ self.W_ + self.b_
        return V_pred.reshape(*orig_shape[:-1], self.W_.shape[1])

    def evaluate(self, latents_test: np.ndarray, behavior_test: np.ndarray) -> ProbeEvaluationResult:
        """Evaluate out-of-sample decoding accuracy on held-out test partition.

        Args:
            latents_test: Array of shape [B_test, T, D] or [N_test, D].
            behavior_test: Array of shape [B_test, T, K] or [N_test, K].

        Returns:
            ProbeEvaluationResult with per-dimension and macro R2 and Pearson r.
        """
        V_true = behavior_test.reshape(-1, behavior_test.shape[-1]).astype(np.float64)
        V_pred = self.predict(latents_test).reshape(-1, V_true.shape[-1])

        K = V_true.shape[1]
        per_dim_r2 = []
        per_dim_r = []

        for k in range(K):
            y = V_true[:, k]
            y_hat = V_pred[:, k]

            # R^2 = 1 - SS_res / SS_tot
            ss_tot = np.sum((y - y.mean()) ** 2)
            ss_res = np.sum((y - y_hat) ** 2)
            r2 = float(1.0 - (ss_res / max(1e-12, ss_tot)))
            per_dim_r2.append(r2)

            # Pearson correlation r
            if np.std(y) > 1e-12 and np.std(y_hat) > 1e-12:
                corr = float(np.corrcoef(y, y_hat)[0, 1])
            else:
                corr = 0.0
            per_dim_r.append(corr)

        macro_r2 = float(np.mean(per_dim_r2))
        macro_r = float(np.mean(per_dim_r))
        mse = float(np.mean((V_true - V_pred) ** 2))

        return ProbeEvaluationResult(
            macro_r2=macro_r2,
            per_dim_r2=per_dim_r2,
            macro_pearson_r=macro_r,
            per_dim_pearson_r=per_dim_r,
            mse=mse,
        )
