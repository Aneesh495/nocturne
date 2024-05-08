"""Dataset split contracts and zero-overlap partition auditing."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional, Sequence, Set, Tuple
from pydantic import BaseModel, ConfigDict, Field

from nocturne.contracts.time import TimeInterval


class SplitType(str, Enum):
    """Partitioning protocol for cross-validation and benchmarking."""

    OFFICIAL_NLB = "official_nlb"
    CHRONOLOGICAL = "chronological"
    GROUPED_CONDITION = "grouped_condition"


class SplitPartition(BaseModel):
    """A single partition (e.g. train, val, test) within a dataset split."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Partition name ('train', 'val', or 'test')")
    trial_ids: Tuple[int, ...] = Field(default_factory=tuple, description="Assigned trial IDs")
    intervals: Tuple[TimeInterval, ...] = Field(
        default_factory=tuple, description="Continuous intervals for continuous recordings"
    )

    @property
    def num_trials(self) -> int:
        return len(self.trial_ids)

    def contains_trial(self, trial_id: int) -> bool:
        return trial_id in self.trial_ids


class DatasetSplit(BaseModel):
    """Complete tripartite partition (train, val, test) with strict zero-leakage guarantees."""

    model_config = ConfigDict(frozen=True)

    split_name: str = Field(..., description="Identifier for this split configuration")
    split_type: SplitType = Field(..., description="Partitioning protocol")
    session_id: str = Field(..., description="Parent recording session ID")
    train: SplitPartition = Field(..., description="Training partition")
    val: SplitPartition = Field(..., description="Validation / model selection partition")
    test: SplitPartition = Field(..., description="Held-out evaluation partition")

    def model_post_init(self, __context: object) -> None:
        self.verify_zero_overlap()

    def verify_zero_overlap(self) -> None:
        """Enforce strict disjointness between train, validation, and test partitions."""
        train_set: Set[int] = set(self.train.trial_ids)
        val_set: Set[int] = set(self.val.trial_ids)
        test_set: Set[int] = set(self.test.trial_ids)

        overlap_tv = train_set.intersection(val_set)
        if overlap_tv:
            raise ValueError(f"Prohibited overlap between train and val: {sorted(list(overlap_tv))}")

        overlap_tt = train_set.intersection(test_set)
        if overlap_tt:
            raise ValueError(f"Prohibited overlap between train and test: {sorted(list(overlap_tt))}")

        overlap_vt = val_set.intersection(test_set)
        if overlap_vt:
            raise ValueError(f"Prohibited overlap between val and test: {sorted(list(overlap_vt))}")

        # Check interval overlaps if continuous intervals are defined
        for i_tr in self.train.intervals:
            for i_va in self.val.intervals:
                if i_tr.overlaps(i_va):
                    raise ValueError(f"Prohibited interval overlap between train {i_tr} and val {i_va}")
            for i_te in self.test.intervals:
                if i_tr.overlaps(i_te):
                    raise ValueError(f"Prohibited interval overlap between train {i_tr} and test {i_te}")
        for i_va in self.val.intervals:
            for i_te in self.test.intervals:
                if i_va.overlaps(i_te):
                    raise ValueError(f"Prohibited interval overlap between val {i_va} and test {i_te}")

    def get_partition_for_trial(self, trial_id: int) -> str | None:
        """Identify which partition contains the specified trial ID, or None if unassigned."""
        if self.train.contains_trial(trial_id):
            return "train"
        if self.val.contains_trial(trial_id):
            return "val"
        if self.test.contains_trial(trial_id):
            return "test"
        return None

    @classmethod
    def create_from_trial_ids(
        cls,
        split_name: str,
        split_type: SplitType,
        session_id: str,
        train_ids: Sequence[int],
        val_ids: Sequence[int],
        test_ids: Sequence[int],
    ) -> DatasetSplit:
        """Construct and validate a DatasetSplit from sequence of trial IDs."""
        return cls(
            split_name=split_name,
            split_type=split_type,
            session_id=session_id,
            train=SplitPartition(name="train", trial_ids=tuple(sorted(train_ids))),
            val=SplitPartition(name="val", trial_ids=tuple(sorted(val_ids))),
            test=SplitPartition(name="test", trial_ids=tuple(sorted(test_ids))),
        )
