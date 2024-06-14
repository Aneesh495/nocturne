"""Out-of-distribution condition generalization and Leave-One-Condition-Out (LOCO) evaluation.

Tests whether learned neural representations generalize to unfamiliar reach directions,
novel movement targets, or unobserved perturbation conditions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from nocturne.evaluation.probes import BehavioralDecodingProbe


@dataclass(frozen=True)
class ConditionMetricRecord:
    """Quantitative performance on an individual held-out behavioral condition."""

    condition_label: str
    num_trials: int
    probe_r2: float
    poisson_deviance: float
    bits_per_spike: Optional[float]


@dataclass(frozen=True)
class CrossConditionResult:
    """Summary report of Leave-One-Condition-Out generalization evaluation."""

    num_conditions: int
    in_distribution_mean_r2: float
    out_of_distribution_mean_r2: float
    generalization_gap: float
    is_robust_generalizer: bool
    condition_records: List[ConditionMetricRecord]
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert result container to nested dictionary."""
        return {
            "num_conditions": self.num_conditions,
            "in_distribution_mean_r2": self.in_distribution_mean_r2,
            "out_of_distribution_mean_r2": self.out_of_distribution_mean_r2,
            "generalization_gap": self.generalization_gap,
            "is_robust_generalizer": self.is_robust_generalizer,
            "condition_records": [asdict(r) for r in self.condition_records],
            "summary": self.summary,
        }


class CrossConditionEvaluator:
    """Evaluates representation robustness under Leave-One-Condition-Out (LOCO) splits."""

    def __init__(self, ridge_alpha: float = 1.0, gap_threshold: float = 0.35):
        self.ridge_alpha = ridge_alpha
        self.gap_threshold = gap_threshold

    def evaluate_loco(
        self,
        latents_by_condition: Dict[str, np.ndarray | torch.Tensor],
        behavior_by_condition: Dict[str, np.ndarray | torch.Tensor],
    ) -> CrossConditionResult:
        """Execute Leave-One-Condition-Out cross-validation.

        Args:
            latents_by_condition: Mapping from condition label to latent tensor [B_c, T, d].
            behavior_by_condition: Mapping from condition label to kinematics [B_c, T, K].

        Returns:
            CrossConditionResult with condition records and generalization gap.
        """
        conditions = sorted(latents_by_condition.keys())
        if len(conditions) < 2:
            raise ValueError(f"Need at least 2 distinct conditions for LOCO, got {len(conditions)}")

        # Convert all to numpy arrays
        lat_dict: Dict[str, np.ndarray] = {}
        beh_dict: Dict[str, np.ndarray] = {}
        for c in conditions:
            v_lat = latents_by_condition[c]
            v_beh = behavior_by_condition[c]
            lat_dict[c] = (
                v_lat.detach().cpu().numpy().astype(np.float64)
                if isinstance(v_lat, torch.Tensor)
                else np.asarray(v_lat, dtype=np.float64)
            )
            beh_dict[c] = (
                v_beh.detach().cpu().numpy().astype(np.float64)
                if isinstance(v_beh, torch.Tensor)
                else np.asarray(v_beh, dtype=np.float64)
            )

        # 1. In-distribution baseline: train on 50% of each condition, test on remaining 50%
        train_id_lat: List[np.ndarray] = []
        train_id_beh: List[np.ndarray] = []
        test_id_lat: List[np.ndarray] = []
        test_id_beh: List[np.ndarray] = []

        for c in conditions:
            B_c = lat_dict[c].shape[0]
            split_idx = max(1, B_c // 2)
            train_id_lat.append(lat_dict[c][:split_idx])
            train_id_beh.append(beh_dict[c][:split_idx])
            test_id_lat.append(lat_dict[c][split_idx:])
            test_id_beh.append(beh_dict[c][split_idx:])

        probe_id = BehavioralDecodingProbe(ridge_alpha=self.ridge_alpha)
        all_train_id_lat = np.concatenate(train_id_lat, axis=0)
        all_train_id_beh = np.concatenate(train_id_beh, axis=0)
        all_test_id_lat = np.concatenate(test_id_lat, axis=0)
        all_test_id_beh = np.concatenate(test_id_beh, axis=0)

        probe_id.fit(all_train_id_lat, all_train_id_beh)
        id_eval = probe_id.evaluate(all_test_id_lat, all_test_id_beh)
        id_mean_r2 = float(id_eval.macro_r2)

        # 2. Leave-One-Condition-Out (OOD generalization)
        records: List[ConditionMetricRecord] = []
        ood_r2_scores: List[float] = []

        for c_held in conditions:
            # Training: all conditions except c_held
            train_lat = [lat_dict[c] for c in conditions if c != c_held]
            train_beh = [beh_dict[c] for c in conditions if c != c_held]
            X_train = np.concatenate(train_lat, axis=0)
            Y_train = np.concatenate(train_beh, axis=0)

            # Test: strictly held-out condition c_held
            X_test = lat_dict[c_held]
            Y_test = beh_dict[c_held]

            probe_loco = BehavioralDecodingProbe(ridge_alpha=self.ridge_alpha)
            probe_loco.fit(X_train, Y_train)
            loco_eval = probe_loco.evaluate(X_test, Y_test)
            r2_val = float(loco_eval.macro_r2)
            ood_r2_scores.append(r2_val)

            records.append(
                ConditionMetricRecord(
                    condition_label=c_held,
                    num_trials=X_test.shape[0],
                    probe_r2=r2_val,
                    poisson_deviance=0.0,
                    bits_per_spike=None,
                )
            )

        ood_mean_r2 = float(np.mean(ood_r2_scores))
        gap = id_mean_r2 - ood_mean_r2
        is_robust = gap < self.gap_threshold

        summary = {
            "num_conditions": len(conditions),
            "in_distribution_r2": id_mean_r2,
            "out_of_distribution_r2": ood_mean_r2,
            "generalization_gap": gap,
            "is_robust": is_robust,
        }

        return CrossConditionResult(
            num_conditions=len(conditions),
            in_distribution_mean_r2=id_mean_r2,
            out_of_distribution_mean_r2=ood_mean_r2,
            generalization_gap=gap,
            is_robust_generalizer=is_robust,
            condition_records=records,
            summary=summary,
        )
