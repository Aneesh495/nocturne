"""Tests for phase portrait CLI subcommand."""

import argparse
from pathlib import Path
from nocturne.cli_subcommands.cmd_phase_portrait import (
    run_phase_portrait_command,
    add_phase_portrait_arguments,
)


def test_add_phase_portrait_arguments():
    parser = argparse.ArgumentParser()
    add_phase_portrait_arguments(parser)
    args = parser.parse_args(["--regime", "bistable_memory", "--format", "svg", "--num-trials", "3"])
    assert args.regime == "bistable_memory"
    assert args.format == "svg"
    assert args.num_trials == 3


def test_run_phase_portrait_command_ascii_and_svg(tmp_path, capsys):
    out_svg = tmp_path / "phase.svg"
    exit_code = run_phase_portrait_command(
        regime="nonlinear_oscillator",
        format_type="both",
        num_trials=2,
        duration_s=0.5,
        output_path=str(out_svg),
        verbose=True,
    )

    assert exit_code == 0
    assert out_svg.exists()
    content = out_svg.read_text(encoding="utf-8")
    assert "<svg" in content
    assert "</svg>" in content

    captured = capsys.readouterr()
    assert "Latent Phase Portrait: nonlinear_oscillator" in captured.out
