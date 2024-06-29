"""Tests for neural spike raster CLI subcommand."""

import argparse
from pathlib import Path
from nocturne.cli_subcommands.cmd_raster import (
    run_raster_command,
    add_raster_arguments,
)


def test_add_raster_arguments():
    parser = argparse.ArgumentParser()
    add_raster_arguments(parser)
    args = parser.parse_args(["--dataset", "mc_maze", "--format", "svg", "--max-bins", "30"])
    assert args.dataset == "mc_maze"
    assert args.format == "svg"
    assert args.max_bins == 30


def test_run_raster_command_ascii_and_svg(tmp_path, capsys):
    out_svg = tmp_path / "raster.svg"
    exit_code = run_raster_command(
        dataset="mc_maze",
        format_type="both",
        max_bins=25,
        max_neurons=8,
        output_path=str(out_svg),
        verbose=True,
    )

    assert exit_code == 0
    assert out_svg.exists()
    content = out_svg.read_text(encoding="utf-8")
    assert "<svg" in content
    assert "</svg>" in content

    captured = capsys.readouterr()
    assert "Nocturne Spike Raster" in captured.out
    assert "N00 |" in captured.out
