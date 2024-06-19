"""3D isometric latent manifold orbit and surface mesh vector SVG visualizer.

Projects 3D neural population trajectories and manifold surfaces onto 2D viewports
using orthographic isometric transformations, depth sorting, and vector SVG paths.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union
import numpy as np
import torch


class ManifoldMeshVisualizer:
    """Renders 3D latent state trajectories and manifold grids into standalone vector SVG."""

    def __init__(self, elevation_deg: float = 30.0, azimuth_deg: float = 45.0):
        self.elevation_deg = elevation_deg
        self.azimuth_deg = azimuth_deg
        self._set_projection_matrix(elevation_deg, azimuth_deg)

    def _set_projection_matrix(self, elevation_deg: float, azimuth_deg: float) -> None:
        theta = math.radians(azimuth_deg)
        phi = math.radians(elevation_deg)
        # Rotation around Z (azimuth) followed by rotation around X (elevation)
        self.R_z = np.array([
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta),  math.cos(theta), 0.0],
            [0.0,              0.0,             1.0],
        ])
        self.R_x = np.array([
            [1.0, 0.0,             0.0],
            [0.0, math.cos(phi), -math.sin(phi)],
            [0.0, math.sin(phi),  math.cos(phi)],
        ])
        self.R = self.R_x @ self.R_z

    def project_point(self, point_3d: np.ndarray) -> Tuple[float, float, float]:
        """Project 3D coordinate [x, y, z] to 2D viewport (x_proj, y_proj) and depth z_depth."""
        p_rot = self.R @ point_3d
        # x_screen = p_rot[0], y_screen = -p_rot[1] (invert Y for screen space), depth = p_rot[2]
        return float(p_rot[0]), float(-p_rot[1]), float(p_rot[2])

    def render_trajectory_svg(
        self,
        trajectory_3d: np.ndarray | torch.Tensor,
        title: str = "3D Latent Manifold Dynamics",
        width: int = 700,
        height: int = 600,
        stroke_color: str = "#00f2fe",
    ) -> str:
        """Render 3D latent trajectory into a vector SVG document.

        Args:
            trajectory_3d: Array or Tensor of shape [T, 3] representing coordinates.
            title: SVG chart title.
            width: Viewport width in pixels.
            height: Viewport height in pixels.
            stroke_color: Base hex color for the trajectory line.

        Returns:
            Valid standalone SVG XML string.
        """
        if isinstance(trajectory_3d, torch.Tensor):
            pts = trajectory_3d.detach().cpu().numpy().astype(np.float64)
        else:
            pts = np.asarray(trajectory_3d, dtype=np.float64)

        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"Expected 3D trajectory [T, 3], got shape {pts.shape}")

        T = pts.shape[0]
        if T < 2:
            raise ValueError(f"Trajectory must have at least 2 points, got {T}")

        # Center and normalize coordinates to fit viewport
        centroid = np.mean(pts, axis=0)
        centered = pts - centroid
        max_dist = float(np.max(np.linalg.norm(centered, axis=1)))
        scale_factor = (min(width, height) * 0.35) / max(1e-6, max_dist)

        center_x = width / 2.0
        center_y = height / 2.0

        # Project 3D points
        proj_pts: List[Tuple[float, float, float]] = []
        for i in range(T):
            x_2d, y_2d, z_d = self.project_point(centered[i] * scale_factor)
            proj_pts.append((center_x + x_2d, center_y + y_2d, z_d))

        # Project 3D coordinate axes
        axis_len = scale_factor * max_dist * 0.8
        axes_3d = [
            (np.array([axis_len, 0.0, 0.0]), "X", "#ff4b4b"),
            (np.array([0.0, axis_len, 0.0]), "Y", "#00d26a"),
            (np.array([0.0, 0.0, axis_len]), "Z", "#3b82f6"),
        ]

        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="background-color: #0b0f19; font-family: monospace;">',
            f'<text x="{center_x}" y="35" fill="#f8fafc" font-size="15" font-weight="bold" text-anchor="middle">{title}</text>',
        ]

        # Draw origin coordinate axes
        for axis_vec, label, color in axes_3d:
            ax_x, ax_y, _ = self.project_point(axis_vec)
            end_x = center_x + ax_x
            end_y = center_y + ax_y
            svg_parts.append(
                f'<line x1="{center_x:.1f}" y1="{center_y:.1f}" x2="{end_x:.1f}" y2="{end_y:.1f}" stroke="{color}" stroke-width="2" opacity="0.7" />'
            )
            svg_parts.append(
                f'<text x="{end_x + 5:.1f}" y="{end_y + 4:.1f}" fill="{color}" font-size="11" font-weight="bold">{label}</text>'
            )

        # Draw trajectory path with SVG polyline / segments
        d_path = [f"M {proj_pts[0][0]:.1f},{proj_pts[0][1]:.1f}"]
        for p in proj_pts[1:]:
            d_path.append(f"L {p[0]:.1f},{p[1]:.1f}")

        path_str = " ".join(d_path)
        svg_parts.append(
            f'<path d="{path_str}" fill="none" stroke="{stroke_color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.9" />'
        )

        # Mark start point (green circle) and end point (red circle)
        start_p = proj_pts[0]
        end_p = proj_pts[-1]
        svg_parts.append(
            f'<circle cx="{start_p[0]:.1f}" cy="{start_p[1]:.1f}" r="5" fill="#00d26a" stroke="#ffffff" stroke-width="1.5" />'
        )
        svg_parts.append(
            f'<circle cx="{end_p[0]:.1f}" cy="{end_p[1]:.1f}" r="5" fill="#ff4b4b" stroke="#ffffff" stroke-width="1.5" />'
        )
        svg_parts.append(
            f'<text x="{start_p[0] + 8:.1f}" y="{start_p[1] + 4:.1f}" fill="#00d26a" font-size="10">t=0</text>'
        )
        svg_parts.append(
            f'<text x="{end_p[0] + 8:.1f}" y="{end_p[1] + 4:.1f}" fill="#ff4b4b" font-size="10">t=T</text>'
        )

        svg_parts.append("</svg>")
        return "\n".join(svg_parts)
