"""Tests for the unified Nocturne CLI dispatcher and argument routing."""

import pytest
from nocturne.cli import build_parser, main


def test_cli_help_and_version(capsys):
    parser = build_parser()
    assert parser.prog == "nocturne"

    # Help output
    exit_code = main([])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Nocturne:" in captured.out
    assert "doctor" in captured.out
    assert "synthetic" in captured.out
    assert "verify" in captured.out
    assert "acceptance" in captured.out
    assert "benchmark" in captured.out
    assert "stream" in captured.out
    assert "raster" in captured.out
    assert "phase_portrait" in captured.out


def test_cli_doctor_dispatch(capsys):
    exit_code = main(["doctor"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "=== Nocturne System Diagnostics ===" in captured.out


def test_cli_raster_dispatch(capsys):
    exit_code = main(["raster", "--dataset", "mc_maze", "--format", "ascii", "--max-bins", "20", "--max-neurons", "4"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "=== Nocturne Spike Raster" in captured.out


def test_cli_unknown_command():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["unknown_command_xyz"])
