"""Tests for spectrogram visualization CLI subcommand."""

import argparse
from pathlib import Path
from nocturne.cli_subcommands.cmd_spectrogram import (
    run_spectrogram_command,
    add_spectrogram_arguments,
)


def test_add_spectrogram_arguments():
    parser = argparse.ArgumentParser()
    add_spectrogram_arguments(parser)
    args = parser.parse_args(["--dataset", "mc_maze", "--max-bins", "30", "--num-freq-bins", "12"])
    assert args.dataset == "mc_maze"
    assert args.max_bins == 30
    assert args.num_freq_bins == 12


def test_run_spectrogram_command(tmp_path, capsys):
    out_svg = tmp_path / "spectrogram.svg"
    exit_code = run_spectrogram_command(
        dataset="mc_maze",
        max_bins=20,
        num_freq_bins=8,
        output_path=str(out_svg),
        verbose=True,
    )

    assert exit_code == 0
    assert out_svg.exists()
    content = out_svg.read_text(encoding="utf-8")
    assert "<svg" in content
    assert "</svg>" in content

    captured = capsys.readouterr()
    assert "Neural Population Spectrogram: mc_maze" in captured.out
    assert "Peak Power:" in captured.out
