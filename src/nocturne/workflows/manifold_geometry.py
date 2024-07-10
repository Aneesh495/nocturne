"""End-to-end scientific workflow: Differential manifold geometry and Riemannian trajectory analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import torch

from nocturne.geometry.alignment import OrthogonalProcrustesAligner
from nocturne.geometry.curvature import CurvatureAnalyzer, TrajectoryCurvatureResult
from nocturne.geometry.tangent import TangentSpaceEstimator, TangentSpaceResult
from nocturne.synthetic.generator import SyntheticDatasetGenerator
from nocturne.viewer.manifold_mesh import ManifoldMeshVisualizer


@dataclass(frozen=True)
class ManifoldGeometryWorkflowResult:
    """Quantitative outcome of neural manifold differential geometry and topology analysis."""

    regime_name: str
    num_conditions: int
    latent_dim: int
    mean_path_length: float
    mean_speed: float
    mean_curvature: float
    max_curvature: float
    mean_intrinsic_dim: float
    mean_grassmann_drift: float
    mean_procrustes_disparity: float
    condition_curvatures: Dict[str, float]
    condition_intrinsic_dims: Dict[str, float]
    svg_manifold_mesh: str


def run_manifold_geometry_workflow(
    regime_name: str = "nonlinear_oscillator",
    num_conditions: int = 4,
    num_neurons: int = 16,
    trial_duration_s: float = 1.0,
    bin_width_ms: int = 20,
    elevation_deg: float = 30.0,
    azimuth_deg: float = 45.0,
    seed: int = 42,
) -> ManifoldGeometryWorkflowResult:
    """Execute end-to-end Riemannian manifold geometry and Grassmannian drift workflow.

    Args:
        regime_name: Synthetic circuit dynamical regime identifier.
        num_conditions: Number of distinct experimental conditions or trajectories to audit.
        num_neurons: Neural population count.
        trial_duration_s: Trial trajectory duration in seconds.
        bin_width_ms: Temporal discretization bin width in milliseconds.
        elevation_deg: Isometric 3D camera elevation angle.
        azimuth_deg: Isometric 3D camera azimuth angle.
        seed: Random seed for synthetic manifold generation.

    Returns:
        ManifoldGeometryWorkflowResult containing differential geometry metrics and vector SVG.
    """
    bin_width_s = bin_width_ms / 1000.0
    generator = SyntheticDatasetGenerator(seed=seed)

    # 1. Generate multi-trial/multi-condition trajectories
    sims, _ = generator.generate_regime_corpus(
        regime_name=regime_name,
        num_trials=num_conditions,
        num_neurons=num_neurons,
        bin_width_s=bin_width_s,
        trial_duration_s=trial_duration_s,
    )

    latent_trajectories: List[np.ndarray] = [s.latent.states for s in sims]
    latent_dim = latent_trajectories[0].shape[-1]

    # 2. Compute curvature and smoothness profiles per condition
    curv_analyzer = CurvatureAnalyzer()
    curv_results: List[TrajectoryCurvatureResult] = []
    condition_curvatures: Dict[str, float] = {}

    for idx, traj in enumerate(latent_trajectories):
        cond_label = f"cond_{idx + 1}"
        c_res = curv_analyzer.compute_trajectory_curvature(traj, dt=bin_width_s)
        curv_results.append(c_res)
        condition_curvatures[cond_label] = float(c_res.mean_curvature)

    # 3. Compute local tangent space participation and Grassmannian drift
    k_dim = min(2, latent_dim)
    tangent_estimator = TangentSpaceEstimator(window_size=5, k_dim=k_dim)
    tangent_results: List[TangentSpaceResult] = []
    condition_intrinsic_dims: Dict[str, float] = {}

    for idx, traj in enumerate(latent_trajectories):
        cond_label = f"cond_{idx + 1}"
        t_res = tangent_estimator.analyze_trajectory(traj)
        tangent_results.append(t_res)
        condition_intrinsic_dims[cond_label] = float(t_res.mean_intrinsic_dim)

    # 4. Compute pairwise Procrustes gauge disparities across conditions
    procrustes_aligner = OrthogonalProcrustesAligner()
    disparities: List[float] = []
    for i in range(len(latent_trajectories)):
        for j in range(i + 1, len(latent_trajectories)):
            t_i = latent_trajectories[i]
            t_j = latent_trajectories[j]
            p_res = procrustes_aligner.evaluate(t_i, t_j)
            disparities.append(float(p_res.disparity))

    mean_procrustes = float(np.mean(disparities)) if disparities else 0.0

    # 5. Synthesize 3D manifold projection and render vector SVG
    mesh_viewer = ManifoldMeshVisualizer(
        elevation_deg=elevation_deg,
        azimuth_deg=azimuth_deg,
    )

    # Pad or project to 3D for visualization
    primary_traj = latent_trajectories[0]
    if latent_dim >= 3:
        traj_3d = primary_traj[:, :3]
    elif latent_dim == 2:
        # Extrude into 3D using velocity norm or zeros
        z_extra = np.linspace(0, 1.0, primary_traj.shape[0])
        traj_3d = np.column_stack([primary_traj, z_extra])
    else:
        traj_3d = np.column_stack([primary_traj, np.zeros((primary_traj.shape[0], 2))])

    svg_mesh = mesh_viewer.render_trajectory_svg(
        trajectory_3d=traj_3d,
        title=f"Manifold Geometry: {regime_name} (Elevation {elevation_deg} deg, Azimuth {azimuth_deg} deg)",
    )

    return ManifoldGeometryWorkflowResult(
        regime_name=regime_name,
        num_conditions=num_conditions,
        latent_dim=latent_dim,
        mean_path_length=float(np.mean([r.path_length for r in curv_results])),
        mean_speed=float(np.mean([r.mean_speed for r in curv_results])),
        mean_curvature=float(np.mean([r.mean_curvature for r in curv_results])),
        max_curvature=float(np.max([r.max_curvature for r in curv_results])),
        mean_intrinsic_dim=float(np.mean([r.mean_intrinsic_dim for r in tangent_results])),
        mean_grassmann_drift=float(np.mean([r.mean_grassmann_drift for r in tangent_results])),
        mean_procrustes_disparity=mean_procrustes,
        condition_curvatures=condition_curvatures,
        condition_intrinsic_dims=condition_intrinsic_dims,
        svg_manifold_mesh=svg_mesh,
    )
