"""Tests for synthetic circuit simulation CLI subcommand."""

import argparse
from nocturne.cli_subcommands.cmd_synthetic import (
    run_synthetic_command,
    add_synthetic_arguments,
)


def test_add_synthetic_arguments():
    parser = argparse.ArgumentParser()
    add_synthetic_arguments(parser)
    args = parser.parse_args(["--regime", "linear_oscillator", "--trials", "5", "--neurons", "10"])
    assert args.regime == "linear_oscillator"
    assert args.trials == 5
    assert args.neurons == 10


def test_run_synthetic_command_single_regime(capsys):
    exit_code = run_synthetic_command(
        regime="linear_oscillator",
        num_trials=4,
        num_neurons=8,
        bin_width_ms=20,
        duration_s=0.5,
        verbose=True,
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "linear_oscillator" in captured.out
    assert "spikes" in captured.out


def test_run_synthetic_command_all_regimes():
    exit_code = run_synthetic_command(
        regime="all",
        num_trials=2,
        num_neurons=6,
        bin_width_ms=20,
        duration_s=0.5,
        verbose=False,
    )
    assert exit_code == 0


def test_run_synthetic_command_invalid_regime():
    exit_code = run_synthetic_command(
        regime="unknown_circuit_xyz",
        num_trials=2,
        verbose=False,
    )
    assert exit_code == 1
