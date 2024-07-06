"""Tests for 3D latent manifold CLI subcommand."""

import argparse
from pathlib import Path
from nocturne.cli_subcommands.cmd_manifold import (
    run_manifold_command,
    add_manifold_arguments,
)


def test_add_manifold_arguments():
    parser = argparse.ArgumentParser()
    add_manifold_arguments(parser)
    args = parser.parse_args(["--regime", "driven_recurrent", "--elevation", "45", "--azimuth", "60"])
    assert args.regime == "driven_recurrent"
    assert args.elevation == 45.0
    assert args.azimuth == 60.0


def test_run_manifold_command(tmp_path, capsys):
    out_svg = tmp_path / "manifold.svg"
    exit_code = run_manifold_command(
        regime="driven_recurrent",
        elevation_deg=35.0,
        azimuth_deg=45.0,
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
    assert "3D Latent Manifold Visualizer: driven_recurrent" in captured.out
    assert "Elevation: 35.0 deg" in captured.out
