"""Tests for 3D latent manifold mesh and orbit vector SVG visualizer."""

import re
import pytest
import numpy as np
import torch
from nocturne.viewer.manifold_mesh import ManifoldMeshVisualizer


def test_manifold_mesh_projection():
    viz = ManifoldMeshVisualizer(elevation_deg=30.0, azimuth_deg=45.0)
    origin = np.array([0.0, 0.0, 0.0])
    x, y, z = viz.project_point(origin)
    assert abs(x) < 1e-6
    assert abs(y) < 1e-6
    assert abs(z) < 1e-6

    # Point along Z axis
    z_pt = np.array([0.0, 0.0, 1.0])
    x_z, y_z, z_depth = viz.project_point(z_pt)
    assert isinstance(x_z, float)
    assert isinstance(y_z, float)


def test_manifold_mesh_render_svg_validity():
    viz = ManifoldMeshVisualizer()
    t = np.linspace(0, 4 * np.pi, 100)
    # 3D spiral trajectory
    traj = np.stack([np.cos(t), np.sin(t), t / 10.0], axis=1)

    svg = viz.render_trajectory_svg(traj, title="Spiral Orbit")
    assert svg.startswith("<svg")
    assert svg.endswith("</svg>")
    assert "Spiral Orbit" in svg
    assert '<path d="M' in svg
    assert '<circle cx="' in svg
    assert "t=0" in svg
    assert "t=T" in svg


def test_manifold_mesh_pytorch_tensor_input():
    viz = ManifoldMeshVisualizer()
    traj_tensor = torch.randn(50, 3)
    svg = viz.render_trajectory_svg(traj_tensor)
    assert "<svg" in svg
    assert "</svg>" in svg


def test_manifold_mesh_invalid_shape_raises():
    viz = ManifoldMeshVisualizer()
    with pytest.raises(ValueError, match="Expected 3D trajectory"):
        viz.render_trajectory_svg(np.zeros((10, 2)))

    with pytest.raises(ValueError, match="Trajectory must have at least 2 points"):
        viz.render_trajectory_svg(np.zeros((1, 3)))
