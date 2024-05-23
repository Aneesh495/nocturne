"""Tests for EvaluationLabelIntegrityAuditor test partition isolation and SHA-256 fingerprints."""

from __future__ import annotations

import numpy as np
import pytest

from nocturne.contracts.splits import DatasetSplit, SplitPartition, SplitType
from nocturne.evaluation.integrity import EvaluationLabelIntegrityAuditor


def test_integrity_auditor_clean_isolation():
    split = DatasetSplit(
        split_name="test_split_001",
        split_type=SplitType.OFFICIAL_NLB,
        session_id="session_001",
        train=SplitPartition(name="train", trial_ids=(0, 1, 2, 3, 4)),
        val=SplitPartition(name="val", trial_ids=(5, 6)),
        test=SplitPartition(name="test", trial_ids=(7, 8, 9)),
    )

    preds = np.random.uniform(5.0, 20.0, size=(3, 10, 4))
    targets = np.random.poisson(0.5, size=(3, 10, 4))

    report = EvaluationLabelIntegrityAuditor.audit_splits_and_predictions(
        dataset_split=split,
        predictions=preds,
        targets=targets,
    )

    assert report.is_partition_disjoint
    assert report.is_evaluation_valid
    assert report.zero_leakage_verified
    assert len(report.predictions_sha256) == 64
    assert len(report.targets_sha256) == 64
    assert report.num_evaluated_observations == 120


def test_integrity_auditor_detects_prohibited_overlap():
    # Attempting to construct overlapping split raises ValueError at contract post-init
    with pytest.raises(ValueError, match="Prohibited overlap"):
        DatasetSplit(
            split_name="corrupt_split",
            split_type=SplitType.OFFICIAL_NLB,
            session_id="session_001",
            train=SplitPartition(name="train", trial_ids=(0, 1, 2, 3)),
            val=SplitPartition(name="val", trial_ids=(4, 5)),
            test=SplitPartition(name="test", trial_ids=(3, 6, 7)),
        )

