"""Tests for adversarial attractor falsification CLI subcommand."""

import argparse
from pathlib import Path
from nocturne.cli_subcommands.cmd_falsify import (
    run_falsify_command,
    add_falsify_arguments,
)


def test_add_falsify_arguments():
    parser = argparse.ArgumentParser()
    add_falsify_arguments(parser)
    args = parser.parse_args(["--seeds", "10", "--shuffles", "20", "-o", "report.json"])
    assert args.seeds == 10
    assert args.shuffles == 20
    assert args.output == "report.json"


def test_run_falsify_command(tmp_path, capsys):
    out_file = tmp_path / "falsify_report.json"
    exit_code = run_falsify_command(
        num_seeds=8,
        surrogate_shuffles=15,
        output_path=str(out_file),
        verbose=True,
    )

    assert exit_code == 0
    assert out_file.exists()

    captured = capsys.readouterr()
    assert "Nocturne Adversarial Attractor Falsification Suite" in captured.out
    assert "ALL NULL CONTROLS CONFIRMED" in captured.out
