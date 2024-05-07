"""Unit tests for trial, event alignment, and behavioral contracts."""

import pytest
import numpy as np

from nocturne.contracts.time import TimeInterval
from nocturne.contracts.trials import (
    BehavioralTrajectory,
    TrialCollection,
    TrialMetadata,
    TrialType,
)


def test_behavioral_trajectory():
    traj = BehavioralTrajectory(
        variable_names=("x", "y"),
        timestamps=(0.0, 0.01, 0.02),
        values=((1.0, 2.0), (1.1, 2.1), (1.2, 2.2)),
    )
    assert traj.num_timepoints == 3
    assert traj.num_channels == 2
    ts, vals = traj.as_numpy()
    assert isinstance(ts, np.ndarray)
    assert isinstance(vals, np.ndarray)
    assert vals.shape == (3, 2)


def test_behavioral_trajectory_mismatch():
    with pytest.raises(ValueError, match="must match timestamps length"):
        BehavioralTrajectory(
            variable_names=("x", "y"),
            timestamps=(0.0, 0.01),
            values=((1.0, 2.0),),
        )

    with pytest.raises(ValueError, match="must match variable_names length"):
        BehavioralTrajectory(
            variable_names=("x", "y"),
            timestamps=(0.0, 0.01),
            values=((1.0,), (1.1,)),
        )


def test_trial_metadata_align_window():
    trial = TrialMetadata(
        trial_id=1,
        session_id="ses_01",
        interval=TimeInterval(start=10.0, stop=15.0),
        events={"go_cue": 11.2, "move_onset": 11.5, "reward": 14.0},
        trial_type=TrialType.DELAYED_REACH,
        condition_id=3,
    )
    assert trial.trial_id == 1
    assert trial.get_event_time("move_onset") == 11.5

    win = trial.align_window("move_onset", pre_s=0.2, post_s=0.5)
    assert win.start == pytest.approx(11.3)
    assert win.stop == pytest.approx(12.0)

    with pytest.raises(KeyError):
        trial.get_event_time("non_existent_event")

    with pytest.raises(ValueError):
        trial.align_window("move_onset", pre_s=-0.1, post_s=0.5)


def test_trial_collection():
    t0 = TrialMetadata(trial_id=0, session_id="s1", interval=TimeInterval(start=0.0, stop=5.0), condition_id=1, is_valid=True)
    t1 = TrialMetadata(trial_id=1, session_id="s1", interval=TimeInterval(start=6.0, stop=10.0), condition_id=2, is_valid=False)
    t2 = TrialMetadata(trial_id=2, session_id="s1", interval=TimeInterval(start=11.0, stop=15.0), condition_id=1, is_valid=True)

    col = TrialCollection(session_id="s1", trials=(t0, t1, t2))
    assert col.num_trials == 3
    assert len(col.valid_trials) == 2
    assert col.get_by_id(1).is_valid is False
    assert len(col.filter_by_condition(1)) == 2
    assert len(col.filter_by_condition(2)) == 1

    with pytest.raises(ValueError, match="Duplicate trial_id"):
        TrialCollection(session_id="s1", trials=(t0, t0))
