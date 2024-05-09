"""Unit tests for recording gap detection and interval window cutting."""

import pytest

from nocturne.binning.intervals import GapDetector, IntervalCutter, RecordingGap
from nocturne.contracts.time import TimeInterval


def test_gap_detector_identifies_breaks():
    i1 = TimeInterval(start=0.0, stop=10.0)
    i2 = TimeInterval(start=12.5, stop=20.0)  # Gap of 2.5s
    i3 = TimeInterval(start=20.2, stop=30.0)  # Gap of 0.2s

    detector = GapDetector(min_gap_duration_s=1.0)
    gaps = detector.detect_gaps_between_intervals([i1, i2, i3])

    assert len(gaps) == 1
    assert gaps[0].duration_s == pytest.approx(2.5)
    assert gaps[0].interval.start == pytest.approx(10.0)
    assert gaps[0].interval.stop == pytest.approx(12.5)


def test_gap_detector_no_gaps():
    i1 = TimeInterval(start=0.0, stop=5.0)
    i2 = TimeInterval(start=5.0, stop=10.0)

    detector = GapDetector(min_gap_duration_s=0.5)
    gaps = detector.detect_gaps_between_intervals([i1, i2])
    assert len(gaps) == 0


def test_interval_cutter_non_overlapping():
    cutter = IntervalCutter(window_duration_s=2.0, step_duration_s=2.0)
    intervals = [
        TimeInterval(start=0.0, stop=5.0),   # Can produce [0, 2), [2, 4)
        TimeInterval(start=10.0, stop=13.0), # Can produce [10, 12)
    ]

    windows = cutter.cut_valid_windows(intervals)
    assert len(windows) == 3

    assert windows[0].start == pytest.approx(0.0)
    assert windows[0].stop == pytest.approx(2.0)
    assert windows[1].start == pytest.approx(2.0)
    assert windows[1].stop == pytest.approx(4.0)
    assert windows[2].start == pytest.approx(10.0)
    assert windows[2].stop == pytest.approx(12.0)


def test_interval_cutter_sliding():
    cutter = IntervalCutter(window_duration_s=2.0, step_duration_s=1.0)
    intervals = [TimeInterval(start=0.0, stop=4.0)]

    windows = cutter.cut_valid_windows(intervals)
    assert len(windows) == 3
    assert windows[0].start == pytest.approx(0.0)
    assert windows[0].stop == pytest.approx(2.0)
    assert windows[1].start == pytest.approx(1.0)
    assert windows[1].stop == pytest.approx(3.0)
    assert windows[2].start == pytest.approx(2.0)
    assert windows[2].stop == pytest.approx(4.0)


def test_interval_cutter_discards_short():
    cutter = IntervalCutter(window_duration_s=5.0)
    intervals = [
        TimeInterval(start=0.0, stop=3.0),   # 3s < 5s -> discarded
        TimeInterval(start=10.0, stop=16.0), # 6s -> produces [10, 15)
    ]
    windows = cutter.cut_valid_windows(intervals)
    assert len(windows) == 1
    assert windows[0].start == pytest.approx(10.0)
    assert windows[0].stop == pytest.approx(15.0)
