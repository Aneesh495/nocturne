"""Unit tests for dataset split auditor and cross-partition leakage checker."""

import pytest

from nocturne.contracts.splits import DatasetSplit, SplitPartition, SplitType
from nocturne.recordings.mc_maze import MCMazeAdapter
from nocturne.recordings.split_audit import SplitAuditor


def test_split_auditor_clean_recording():
    adapter = MCMazeAdapter()
    recording = adapter.create_synthetic_maze_fixture(num_trials=20, num_neurons=15)

    report_official = SplitAuditor.audit_recording_split(recording, split_name="official")
    assert report_official.is_valid is True
    assert len(report_official.leakage_violations) == 0

    report_chron = SplitAuditor.audit_recording_split(recording, split_name="chronological")
    assert report_chron.is_valid is True
    assert len(report_chron.chronological_violations) == 0


def test_split_auditor_detects_chronological_violation():
    adapter = MCMazeAdapter()
    recording = adapter.create_synthetic_maze_fixture(num_trials=20, num_neurons=15)

    # Invert chronological split: put later trials in train and earlier in val
    bad_chron = DatasetSplit(
        split_name="bad_chron",
        split_type=SplitType.CHRONOLOGICAL,
        session_id=recording.session_id,
        train=SplitPartition(name="train", trial_ids=(10, 11, 12)),
        val=SplitPartition(name="val", trial_ids=(0, 1, 2)),
        test=SplitPartition(name="test", trial_ids=(18, 19)),
    )
    # Add bad split to recording
    recording.splits["bad_chron"] = bad_chron

    report = SplitAuditor.audit_recording_split(recording, split_name="bad_chron")
    assert report.is_valid is False
    assert len(report.chronological_violations) > 0
    assert "occurs after Val trial start" in report.chronological_violations[0]
