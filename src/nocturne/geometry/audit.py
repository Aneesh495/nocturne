"""Prediction-vs-structure divergence audit and cross-model geometry evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn

from nocturne.geometry.alignment import OrthogonalProcrustesAligner, ProcrustesAlignmentResult
from nocturne.geometry.cca import CanonicalCorrelationAnalysis, CCAResult
from nocturne.geometry.fixed_points import FixedPointFinder, FixedPoint


@dataclass(frozen=True)
class GeometricDivergenceReport:
    """Quantitative comparison of representation geometry across models or ground truth."""

    num_trials_evaluated: int
    train_procrustes_disparity: float
    test_procrustes_disparity: float
    mean_cca_correlation: float
    canonical_correlations: List[float]
    ground_truth_fixed_points: int
    inferred_fixed_points: int
    has_geometric_fidelity: bool
    summary: Dict[str, Any]


class PredictionStructureDivergenceAudit:
    """Evaluates whether high predictive accuracy corresponds to true latent geometry."""

    def __init__(self, disparity_threshold: float = 0.5):
        self.disparity_threshold = disparity_threshold
        self.aligner = OrthogonalProcrustesAligner(allow_reflection=True)
        self.cca = CanonicalCorrelationAnalysis(reg=1e-4)
        self.fp_finder = FixedPointFinder()

    def audit_dynamics(
        self,
        true_latents_train: np.ndarray,
        inferred_latents_train: np.ndarray,
        true_latents_test: np.ndarray,
        inferred_latents_test: np.ndarray,
    ) -> GeometricDivergenceReport:
        """Run gauge alignment and geometry comparison across train and test partitions.

        Args:
            true_latents_train: Array [B_train, T, D] ground truth latents.
            inferred_latents_train: Array [B_train, T, D] inferred latents.
            true_latents_test: Array [B_test, T, D] held-out test latents (>= 200 trials for BA09).
            inferred_latents_test: Array [B_test, T, D] held-out test inferred latents.

        Returns:
            GeometricDivergenceReport with train/test disparities and CCA metrics.
        """
        N_test = true_latents_test.shape[0]

        # 1. Fit Procrustes strictly on training trials
        train_res = self.aligner.evaluate(inferred_latents_train, true_latents_train)

        # 2. Apply fitted alignment out-of-sample to held-out test trials
        inferred_test_aligned = self.aligner.transform(inferred_latents_test)
        test_residual = np.linalg.norm(inferred_test_aligned - true_latents_test)
        test_target_norm = np.linalg.norm(true_latents_test)
        test_disparity = float(test_residual / max(1e-12, test_target_norm))

        # 3. Canonical Correlation Analysis on test representations
        cca_res = self.cca.fit(inferred_latents_test, true_latents_test)

        has_fidelity = (
            test_disparity <= self.disparity_threshold
            and cca_res.mean_correlation >= 0.70
        )

        return GeometricDivergenceReport(
            num_trials_evaluated=N_test,
            train_procrustes_disparity=train_res.disparity,
            test_procrustes_disparity=test_disparity,
            mean_cca_correlation=cca_res.mean_correlation,
            canonical_correlations=[float(c) for c in cca_res.canonical_correlations],
            ground_truth_fixed_points=0,
            inferred_fixed_points=0,
            has_geometric_fidelity=has_fidelity,
            summary={
                "rotation_matrix": train_res.R.tolist(),
                "scale": float(train_res.scale),
                "is_aligned": has_fidelity,
            },
        )
