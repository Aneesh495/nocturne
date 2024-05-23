"""Phase portrait rendering for continuous latent state trajectories and vector fields."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple
import numpy as np


class PhasePortraitVisualizer:
    """Renders vector SVG and ASCII phase portraits of continuous latent dynamics."""

    @staticmethod
    def render_ascii(
        trajectories: np.ndarray,
        fixed_points: Optional[np.ndarray] = None,
        grid_size: int = 25,
    ) -> str:
        """Render a 2D ASCII phase portrait of latent trajectories.

        Args:
            trajectories: Array of latent trajectories [T, D] or [K, T, D] where D >= 2.
            fixed_points: Optional array of fixed points [M, D].
            grid_size: Resolution of the ASCII grid.

        Returns:
            ASCII string rendering of phase plane.
        """
        if trajectories.ndim == 2:
            trajectories = trajectories[np.newaxis, ...]  # [1, T, D]
        K, T, D = trajectories.shape
        if D < 2:
            raise ValueError(f"Phase portrait requires at least 2 latent dimensions, got {D}")

        z1 = trajectories[..., 0].flatten()
        z2 = trajectories[..., 1].flatten()

        min_x, max_x = float(np.min(z1)), float(np.max(z1))
        min_y, max_y = float(np.min(z2)), float(np.max(z2))

        # Add 10% margin
        span_x = max(1e-5, max_x - min_x)
        span_y = max(1e-5, max_y - min_y)
        min_x -= 0.1 * span_x
        max_x += 0.1 * span_x
        min_y -= 0.1 * span_y
        max_y += 0.1 * span_y

        canvas = [[" " for _ in range(grid_size)] for _ in range(grid_size)]

        # Plot trajectories
        for k in range(K):
            for t in range(T):
                x = trajectories[k, t, 0]
                y = trajectories[k, t, 1]
                col = int((x - min_x) / (max_x - min_x) * (grid_size - 1))
                row = int((max_y - y) / (max_y - min_y) * (grid_size - 1))
                if 0 <= col < grid_size and 0 <= row < grid_size:
                    if t == 0:
                        canvas[row][col] = "o"  # start
                    elif canvas[row][col] == " ":
                        canvas[row][col] = "."

        # Plot fixed points
        if fixed_points is not None:
            for m in range(len(fixed_points)):
                fx, fy = fixed_points[m, 0], fixed_points[m, 1]
                col = int((fx - min_x) / (max_x - min_x) * (grid_size - 1))
                row = int((max_y - fy) / (max_y - min_y) * (grid_size - 1))
                if 0 <= col < grid_size and 0 <= row < grid_size:
                    canvas[row][col] = "*"  # fixed point

        lines = [f"=== Latent Phase Plane (z1: [{min_x:.2f}, {max_x:.2f}], z2: [{min_y:.2f}, {max_y:.2f}]) ==="]
        lines.append("+" + "-" * grid_size + "+")
        for row in canvas:
            lines.append("|" + "".join(row) + "|")
        lines.append("+" + "-" * grid_size + "+")
        lines.append("Legend: 'o' = initial state, '.' = trajectory, '*' = fixed point")
        return "\n".join(lines)

    @staticmethod
    def render_svg(
        trajectories: np.ndarray,
        flow_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        fixed_points: Optional[np.ndarray] = None,
        stability_labels: Optional[List[str]] = None,
        width: int = 700,
        height: int = 700,
        title: str = "Latent Phase Portrait & Vector Field",
    ) -> str:
        """Generate high-resolution standalone vector SVG phase portrait.

        Args:
            trajectories: Latent paths [T, D] or [K, T, D] where D >= 2.
            flow_fn: Optional vector field evaluation function f(z) returning dz/dt.
            fixed_points: Optional fixed points [M, D].
            stability_labels: Optional labels for fixed points ('stable', 'saddle', 'unstable').
            width: SVG width in pixels.
            height: SVG height in pixels.
            title: Figure title.

        Returns:
            Complete SVG XML document as string.
        """
        if trajectories.ndim == 2:
            trajectories = trajectories[np.newaxis, ...]
        K, T, D = trajectories.shape
        if D < 2:
            raise ValueError(f"Phase portrait requires at least 2 latent dimensions, got {D}")

        margin = 60
        plot_w = width - 2 * margin
        plot_h = height - 2 * margin

        all_z1 = trajectories[..., 0].flatten()
        all_z2 = trajectories[..., 1].flatten()
        if fixed_points is not None and len(fixed_points) > 0:
            all_z1 = np.concatenate([all_z1, fixed_points[:, 0]])
            all_z2 = np.concatenate([all_z2, fixed_points[:, 1]])

        min_x, max_x = float(np.min(all_z1)), float(np.max(all_z1))
        min_y, max_y = float(np.min(all_z2)), float(np.max(all_z2))
        span_x = max(1e-4, max_x - min_x)
        span_y = max(1e-4, max_y - min_y)
        min_x -= 0.15 * span_x
        max_x += 0.15 * span_x
        min_y -= 0.15 * span_y
        max_y += 0.15 * span_y

        def to_svg_x(x: float) -> float:
            return margin + ((x - min_x) / (max_x - min_x)) * plot_w

        def to_svg_y(y: float) -> float:
            return margin + ((max_y - y) / (max_y - min_y)) * plot_h

        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
            f'<rect width="{width}" height="{height}" fill="#0F172A"/>',  # Slate dark background
            f'<text x="{margin}" y="36" fill="#F8FAFC" font-family="monospace" font-size="16" font-weight="bold">{title}</text>',
            f'<rect x="{margin}" y="{margin}" width="{plot_w}" height="{plot_h}" fill="#1E293B" stroke="#334155" stroke-width="1.5"/>',
        ]

        # Draw grid axes
        zero_x = to_svg_x(0.0)
        zero_y = to_svg_y(0.0)
        if margin <= zero_x <= margin + plot_w:
            svg_parts.append(
                f'<line x1="{zero_x:.1f}" y1="{margin}" x2="{zero_x:.1f}" y2="{margin + plot_h}" stroke="#475569" stroke-dasharray="3,3" stroke-width="1"/>'
            )
        if margin <= zero_y <= margin + plot_h:
            svg_parts.append(
                f'<line x1="{margin}" y1="{zero_y:.1f}" x2="{margin + plot_w}" y2="{zero_y:.1f}" stroke="#475569" stroke-dasharray="3,3" stroke-width="1"/>'
            )

        # Draw vector field if flow_fn is provided
        if flow_fn is not None:
            n_grid = 16
            gx = np.linspace(min_x, max_x, n_grid)
            gy = np.linspace(min_y, max_y, n_grid)
            grid_pts = []
            for y_val in gy:
                for x_val in gx:
                    pt = np.zeros(D)
                    pt[0] = x_val
                    pt[1] = y_val
                    grid_pts.append(pt)
            grid_arr = np.array(grid_pts)
            flow_vecs = flow_fn(grid_arr)  # [n_grid*n_grid, D]

            max_norm = float(np.max(np.linalg.norm(flow_vecs[:, :2], axis=1))) + 1e-8
            arrow_scale = (plot_w / n_grid) * 0.75

            for idx, pt in enumerate(grid_arr):
                v = flow_vecs[idx]
                v_norm = float(np.linalg.norm(v[:2]))
                if v_norm < 1e-6:
                    continue
                dx = (v[0] / max_norm) * arrow_scale
                dy = (v[1] / max_norm) * arrow_scale

                x1_s = to_svg_x(pt[0])
                y1_s = to_svg_y(pt[1])
                x2_s = x1_s + dx
                y2_s = y1_s - dy  # SVG inverted y

                mag_ratio = min(1.0, v_norm / max_norm)
                opacity = 0.25 + 0.45 * mag_ratio
                svg_parts.append(
                    f'<line x1="{x1_s:.1f}" y1="{y1_s:.1f}" x2="{x2_s:.1f}" y2="{y2_s:.1f}" stroke="#94A3B8" stroke-width="1.2" stroke-opacity="{opacity:.2f}"/>'
                )

        # Draw trajectory paths
        colors = ["#38BDF8", "#34D399", "#F472B6", "#FBBF24", "#A78BFA"]
        for k in range(K):
            col = colors[k % len(colors)]
            pts_svg = []
            for t in range(T):
                xs = to_svg_x(trajectories[k, t, 0])
                ys = to_svg_y(trajectories[k, t, 1])
                pts_svg.append(f"{xs:.1f},{ys:.1f}")

            polyline_str = " ".join(pts_svg)
            svg_parts.append(
                f'<polyline points="{polyline_str}" fill="none" stroke="{col}" stroke-width="2.2" stroke-opacity="0.9"/>'
            )

            # Mark start point
            x0 = to_svg_x(trajectories[k, 0, 0])
            y0 = to_svg_y(trajectories[k, 0, 1])
            svg_parts.append(
                f'<circle cx="{x0:.1f}" cy="{y0:.1f}" r="4" fill="{col}" stroke="#0F172A" stroke-width="1.5"/>'
            )

        # Draw fixed points
        if fixed_points is not None:
            for m in range(len(fixed_points)):
                fx = to_svg_x(fixed_points[m, 0])
                fy = to_svg_y(fixed_points[m, 1])
                label = stability_labels[m] if (stability_labels and m < len(stability_labels)) else "point"

                # Color by stability
                if "stable" in label.lower() and "un" not in label.lower():
                    pt_col = "#10B981"  # Emerald
                elif "saddle" in label.lower():
                    pt_col = "#F59E0B"  # Amber
                else:
                    pt_col = "#EF4444"  # Red

                svg_parts.append(
                    f'<polygon points="{fx:.1f},{fy-7:.1f} {fx+6:.1f},{fy+5:.1f} {fx-6:.1f},{fy+5:.1f}" fill="{pt_col}" stroke="#FFFFFF" stroke-width="1.5"/>'
                )
                svg_parts.append(
                    f'<text x="{fx + 8:.1f}" y="{fy - 8:.1f}" fill="{pt_col}" font-family="monospace" font-size="11" font-weight="bold">{label}</text>'
                )

        # Axis annotations
        svg_parts.append(
            f'<text x="{margin + plot_w / 2}" y="{height - 18}" fill="#94A3B8" font-family="monospace" font-size="12" text-anchor="middle">Latent Coordinate z_1</text>'
        )
        svg_parts.append(
            f'<text x="20" y="{margin + plot_h / 2}" fill="#94A3B8" font-family="monospace" font-size="12" text-anchor="middle" transform="rotate(-90 20 {margin + plot_h / 2})">Latent Coordinate z_2</text>'
        )

        svg_parts.append("</svg>")
        return "\n".join(svg_parts)

    @classmethod
    def save_svg(
        cls,
        output_path: str | Path,
        trajectories: np.ndarray,
        flow_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        fixed_points: Optional[np.ndarray] = None,
        stability_labels: Optional[List[str]] = None,
        title: str = "Latent Phase Portrait & Vector Field",
    ) -> Path:
        """Render and save phase portrait SVG to file.

        Args:
            output_path: File destination path.
            trajectories: Latent coordinates.
            flow_fn: Vector field function.
            fixed_points: Fixed point locations.
            stability_labels: Stability classification strings.
            title: Title.

        Returns:
            Resolved Path.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        svg_str = cls.render_svg(
            trajectories=trajectories,
            flow_fn=flow_fn,
            fixed_points=fixed_points,
            stability_labels=stability_labels,
            title=title,
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg_str)
        return path
