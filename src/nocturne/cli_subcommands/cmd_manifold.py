"""3D latent manifold orbit and mesh visualization CLI subcommand."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional
import numpy as np

from nocturne.synthetic.generator import SyntheticDatasetGenerator
from nocturne.viewer.manifold_mesh import ManifoldMeshVisualizer


def run_manifold_command(
    regime: str = "driven_recurrent",
    elevation_deg: float = 30.0,
    azimuth_deg: float = 45.0,
    duration_s: float = 1.5,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> int:
    """Render 3D isometric latent manifold trajectory into vector SVG."""
    generator = SyntheticDatasetGenerator(seed=42)

    sims, _ = generator.generate_regime_corpus(
        regime_name=regime if regime in generator.SUPPORTED_REGIMES else "driven_recurrent",
        num_trials=1,
        trial_duration_s=duration_s,
    )
    latent_states = sims[0].latent.states  # [T, D]
    T, D = latent_states.shape

    if D >= 3:
        traj_3d = latent_states[:, :3]
    else:
        # Pad with helical temporal coordinate if 2D
        t_coord = np.linspace(-1.0, 1.0, T)[:, np.newaxis]
        traj_3d = np.concatenate([latent_states, t_coord], axis=1)

    viz = ManifoldMeshVisualizer(elevation_deg=elevation_deg, azimuth_deg=azimuth_deg)
    svg_xml = viz.render_trajectory_svg(
        trajectory_3d=traj_3d,
        title=f"3D Latent Manifold Dynamics: {regime}",
    )

    if verbose:
        print(f"=== 3D Latent Manifold Visualizer: {regime} ===")
        print(f"Elevation: {elevation_deg:.1f} deg | Azimuth: {azimuth_deg:.1f} deg | Timepoints: {T}")
        print(f"Latent Extents: X=[{traj_3d[:,0].min():.2f}, {traj_3d[:,0].max():.2f}], Y=[{traj_3d[:,1].min():.2f}, {traj_3d[:,1].max():.2f}], Z=[{traj_3d[:,2].min():.2f}, {traj_3d[:,2].max():.2f}]")

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(svg_xml)
        if verbose:
            print(f"\nVector SVG 3D manifold saved to: {p}")

    return 0


def add_manifold_arguments(parser: argparse.ArgumentParser) -> None:
    """Register CLI arguments for the manifold subcommand."""
    parser.add_argument(
        "--regime",
        type=str,
        default="driven_recurrent",
        choices=["driven_recurrent", "linear_oscillator", "nonlinear_oscillator", "bistable_memory"],
        help="Circuit regime from which to project 3D manifold",
    )
    parser.add_argument(
        "--elevation",
        type=float,
        default=30.0,
        help="Elevation viewing angle in degrees",
    )
    parser.add_argument(
        "--azimuth",
        type=float,
        default=45.0,
        help="Azimuth viewing angle in degrees",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.5,
        help="Trajectory duration in seconds",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional path to write SVG output",
    )
