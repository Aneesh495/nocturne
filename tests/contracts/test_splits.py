"""Unit tests for dataset split contracts and zero-overlap partition auditing."""

import pytest

from nocturne.contracts.splits import DatasetSplit, SplitPartition, SplitType
from nocturne.contracts.time import TimeInterval


def test_dataset_split_valid():
    split = DatasetSplit.create_from_trial_ids(
        split_name="official_test_split",
        split_type=SplitType.OFFICIAL_NLB,
        session_id="session_001",
        train_ids=[0, 1, 2, 3],
        val_ids=[4, 5],
        test_ids=[6, 7, 8],
    )

    assert split.train.num_trials == 4
    assert split.val.num_trials == 2
    assert split.test.num_trials == 3
    assert split.get_partition_for_trial(0) == "train"
    assert split.get_partition_for_trial(5) == "val"
    assert split.get_partition_for_trial(7) == "test"
    assert split.get_partition_for_trial(99) is None


def test_dataset_split_prohibited_trial_overlap():
    # Train and Val overlap
    with pytest.raises(ValueError, match="Prohibited overlap between train and val"):
        DatasetSplit.create_from_trial_ids(
            split_name="bad_split_tv",
            split_type=SplitType.CHRONOLOGICAL,
            session_id="s1",
            train_ids=[0, 1, 2],
            val_ids=[2, 3],
            test_ids=[4, 5],
        )

    # Train and Test overlap
    with pytest.raises(ValueError, match="Prohibited overlap between train and test"):
        DatasetSplit.create_from_trial_ids(
            split_name="bad_split_tt",
            split_type=SplitType.CHRONOLOGICAL,
            session_id="s1",
            train_ids=[0, 1, 2],
            val_ids=[3, 4],
            test_ids=[1, 5],
        )

    # Val and Test overlap
    with pytest.raises(ValueError, match="Prohibited overlap between val and test"):
        DatasetSplit.create_from_trial_ids(
            split_name="bad_split_vt",
            split_type=SplitType.CHRONOLOGICAL,
            session_id="s1",
            train_ids=[0, 1, 2],
            val_ids=[3, 4],
            test_ids=[4, 5],
        )


def test_dataset_split_interval_overlap():
    tr_int = TimeInterval(start=0.0, stop=10.0)
    va_int = TimeInterval(start=9.0, stop=15.0)  # Overlaps [9.0, 10.0)
    te_int = TimeInterval(start=15.0, stop=20.0)

    with pytest.raises(ValueError, match="Prohibited interval overlap"):
        DatasetSplit(
            split_name="interval_overlap_split",
            split_type=SplitType.CHRONOLOGICAL,
            session_id="s1",
            train=SplitPartition(name="train", intervals=(tr_int,)),
            val=SplitPartition(name="val", intervals=(va_int,)),
            test=SplitPartition(name="test", intervals=(te_int,)),
        )
