"""Dataset split auditing and cross-partition leakage detection."""

from __future__ import annotations

from typing import List, Set
from pydantic import BaseModel, ConfigDict, Field

from nocturne.contracts.neurons import NeuronRole
from nocturne.contracts.splits import DatasetSplit, SplitType
from nocturne.recordings.base import ProcessedRecording


class SplitAuditReport(BaseModel):
    """Integrity audit report confirming zero leakage across data partitions."""

    model_config = ConfigDict(frozen=True)

    dataset_name: str = Field(..., description="Dataset name")
    session_id: str = Field(..., description="Session identifier")
    split_name: str = Field(..., description="Split name audited")
    is_valid: bool = Field(..., description="Whether split passed all integrity checks without leakage")
    num_train_trials: int = Field(..., ge=0)
    num_val_trials: int = Field(..., ge=0)
    num_test_trials: int = Field(..., ge=0)
    leakage_violations: List[str] = Field(default_factory=list, description="Descriptions of any detected leaks")
    chronological_violations: List[str] = Field(default_factory=list, description="Chronological ordering violations")


class SplitAuditor:
    """Audits dataset splits and recordings for zero trial leakage and chronological consistency."""

    @staticmethod
    def audit_recording_split(
        recording: ProcessedRecording,
        split_name: str = "official",
    ) -> SplitAuditReport:
        """Perform exhaustive leakage audit on a recording's split."""
        split = recording.get_split(split_name)
        leakage_violations: List[str] = []
        chronological_violations: List[str] = []

        # 1. Check trial ID disjointness
        train_set = set(split.train.trial_ids)
        val_set = set(split.val.trial_ids)
        test_set = set(split.test.trial_ids)

        tv = train_set.intersection(val_set)
        if tv:
            leakage_violations.append(f"Train/Val overlap on trial IDs: {sorted(list(tv))}")

        tt = train_set.intersection(test_set)
        if tt:
            leakage_violations.append(f"Train/Test overlap on trial IDs: {sorted(list(tt))}")

        vt = val_set.intersection(test_set)
        if vt:
            leakage_violations.append(f"Val/Test overlap on trial IDs: {sorted(list(vt))}")

        # 2. Check chronological consistency if split is CHRONOLOGICAL
        if split.split_type == SplitType.CHRONOLOGICAL and recording.trials is not None:
            trials = recording.trials
            train_stops = [trials.get_by_id(tid).interval.stop for tid in split.train.trial_ids if tid in [t.trial_id for t in trials.trials]]
            val_starts = [trials.get_by_id(tid).interval.start for tid in split.val.trial_ids if tid in [t.trial_id for t in trials.trials]]
            val_stops = [trials.get_by_id(tid).interval.stop for tid in split.val.trial_ids if tid in [t.trial_id for t in trials.trials]]
            test_starts = [trials.get_by_id(tid).interval.start for tid in split.test.trial_ids if tid in [t.trial_id for t in trials.trials]]

            if train_stops and val_starts:
                max_train = max(train_stops)
                min_val = min(val_starts)
                if max_train > min_val:
                    chronological_violations.append(f"Train trial stop ({max_train:.3f}s) occurs after Val trial start ({min_val:.3f}s)")

            if val_stops and test_starts:
                max_val = max(val_stops)
                min_test = min(test_starts)
                if max_val > min_test:
                    chronological_violations.append(f"Val trial stop ({max_val:.3f}s) occurs after Test trial start ({min_test:.3f}s)")

        # 3. Check held-out unit purity
        pop = recording.metadata.population
        held_out = set(pop.held_out_indices)
        held_in = set(pop.held_in_indices)
        overlap_units = held_out.intersection(held_in)
        if overlap_units:
            leakage_violations.append(f"Channel indices assigned as both held-in and held-out: {sorted(list(overlap_units))}")

        is_valid = len(leakage_violations) == 0 and len(chronological_violations) == 0

        return SplitAuditReport(
            dataset_name=recording.metadata.provenance.dataset_name,
            session_id=recording.session_id,
            split_name=split_name,
            is_valid=is_valid,
            num_train_trials=split.train.num_trials,
            num_val_trials=split.val.num_trials,
            num_test_trials=split.test.num_trials,
            leakage_violations=leakage_violations,
            chronological_violations=chronological_violations,
        )
