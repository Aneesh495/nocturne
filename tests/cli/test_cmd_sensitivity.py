"""Tests for sensitivity sweep CLI subcommand."""

import argparse
from pathlib import Path
from nocturne.cli_subcommands.cmd_sensitivity import (
    run_sensitivity_command,
    add_sensitivity_arguments,
)


def test_add_sensitivity_arguments():
    parser = argparse.ArgumentParser()
    add_sensitivity_arguments(parser)
    args = parser.parse_args(["--parameter", "latent_dimension", "-o", "sens.json"])
    assert args.parameter == "latent_dimension"
    assert args.output == "sens.json"


def test_run_sensitivity_command(tmp_path, capsys):
    out_file = tmp_path / "sens_report.json"
    exit_code = run_sensitivity_command(
        parameter="latent_dimension",
        output_path=str(out_file),
        verbose=True,
    )

    assert exit_code == 0
    assert out_file.exists()

    captured = capsys.readouterr()
    assert "Hyperparameter Sensitivity: `latent_dimension`" in captured.out
    assert "Optimal Dimensionality Saturation Knee:" in captured.out
