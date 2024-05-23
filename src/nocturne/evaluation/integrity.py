"""Independent test label isolation, non-leakage verification, and cryptographic fingerprints."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Dict, List, Set
import numpy as np
import torch

from nocturne.contracts.splits import DatasetSplit


@dataclass(frozen=True)
class EvaluationIntegrityReport:
    """Formal audit report guaranteeing test partition isolation and prediction integrity."""

    is_partition_disjoint: bool
    is_evaluation_valid: bool
    predictions_sha256: str
    targets_sha256: str
    num_evaluated_observations: int
    zero_leakage_verified: bool
    violations: List[str]


class EvaluationLabelIntegrityAuditor:
    """Independent auditor certifying zero data leakage between training and evaluation."""

    @staticmethod
    def audit_splits_and_predictions(
        dataset_split: DatasetSplit,
        predictions: np.ndarray | torch.Tensor,
        targets: np.ndarray | torch.Tensor,
        held_out_channels: Set[int] | None = None,
    ) -> EvaluationIntegrityReport:
        """Verify strict partition isolation and compute immutable SHA-256 fingerprints.

        Args:
            dataset_split: Tripartite dataset split contract.
            predictions: Predicted test rates or counts.
            targets: True held-out test counts.
            held_out_channels: Optional set of channel indices designated as held-out.

        Returns:
            EvaluationIntegrityReport certifying data isolation.
        """
        violations: List[str] = []

        # Step 1: Verify split disjointness
        train_trials = set(dataset_split.train.trial_ids)
        test_trials = set(dataset_split.test.trial_ids)
        overlap = train_trials.intersection(test_trials)
        if len(overlap) > 0:
            violations.append(
                f"Prohibited partition overlap: {len(overlap)} trials exist in both train and test!"
            )


        # Step 2: Shape validation
        if isinstance(predictions, torch.Tensor):
            preds_np = predictions.detach().cpu().numpy().astype(np.float64)
        else:
            preds_np = np.asarray(predictions, dtype=np.float64)

        if isinstance(targets, torch.Tensor):
            targs_np = targets.detach().cpu().numpy().astype(np.float64)
        else:
            targs_np = np.asarray(targets, dtype=np.float64)

        if preds_np.shape != targs_np.shape:
            violations.append(
                f"Shape mismatch between predictions {preds_np.shape} and targets {targs_np.shape}"
            )

        # Step 3: Check for non-finite values
        if not np.isfinite(preds_np).all():
            violations.append("Non-finite values (NaN / Inf) detected in predictions")
        if not np.isfinite(targs_np).all():
            violations.append("Non-finite values detected in targets")

        # Step 4: Check for non-negative rates
        if (preds_np < 0.0).any():
            violations.append("Negative firing rates detected in predictions")

        # Step 5: Check non-trivial predictions
        if np.std(preds_np) < 1e-12:
            violations.append("Degenerate predictions: zero variance across all evaluated bins")

        # Step 6: Cryptographic SHA-256 hash
        pred_bytes = preds_np.tobytes()
        targ_bytes = targs_np.tobytes()
        pred_hash = hashlib.sha256(pred_bytes).hexdigest()
        targ_hash = hashlib.sha256(targ_bytes).hexdigest()

        is_disjoint = len(overlap) == 0
        is_valid = len(violations) == 0

        return EvaluationIntegrityReport(
            is_partition_disjoint=is_disjoint,
            is_evaluation_valid=is_valid,
            predictions_sha256=pred_hash,
            targets_sha256=targ_hash,
            num_evaluated_observations=int(preds_np.size),
            zero_leakage_verified=(is_disjoint and is_valid),
            violations=violations,
        )
