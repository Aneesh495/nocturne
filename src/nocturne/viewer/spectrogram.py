"""Vector SVG time-frequency spectrogram and population wave visualizer.

Generates standalone, resolution-independent vector SVG spectrogram charts
with viridis/inferno color palettes, frequency tick labels, and power intensity scales.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union
import numpy as np
import torch


def _viridis_colormap(val: float) -> str:
    """Map normalized value in [0, 1] to viridis-inspired RGB hex string."""
    v = max(0.0, min(1.0, float(val)))
    # 4-point piecewise linear approximation of viridis colormap
    colors = [
        (68, 1, 84),     # 0.0: deep purple
        (59, 82, 139),   # 0.33: blue
        (33, 145, 140),  # 0.67: teal
        (253, 231, 37),  # 1.0: yellow
    ]
    if v <= 0.33:
        t = v / 0.33
        c0, c1 = colors[0], colors[1]
    elif v <= 0.67:
        t = (v - 0.33) / 0.34
        c0, c1 = colors[1], colors[2]
    else:
        t = (v - 0.67) / 0.33
        c0, c1 = colors[2], colors[3]

    r = int(c0[0] + t * (c1[0] - c0[0]))
    g = int(c0[1] + t * (c1[1] - c0[1]))
    b = int(c0[2] + t * (c1[2] - c0[2]))
    return f"#{r:02x}{g:02x}{b:02x}"


class SpectrogramVisualizer:
    """Renders time-frequency power spectrogram matrices into standalone vector SVG."""

    def __init__(self, bin_width_s: float = 0.02):
        self.bin_width_s = bin_width_s

    def render_svg(
        self,
        power_matrix: np.ndarray | torch.Tensor,
        freq_grid_hz: Optional[List[float]] = None,
        title: str = "Neural Population Time-Frequency Spectrogram",
        width: int = 800,
        height: int = 400,
    ) -> str:
        """Render a 2D time-frequency power matrix to an SVG XML document.

        Args:
            power_matrix: 2D array of shape [F, T] (frequency rows, time columns).
            freq_grid_hz: Optional list of frequency values corresponding to rows.
            title: Title text for the spectrogram plot.
            width: Total SVG viewport width in pixels.
            height: Total SVG viewport height in pixels.

        Returns:
            Valid standalone SVG string.
        """
        if isinstance(power_matrix, torch.Tensor):
            mat = power_matrix.detach().cpu().numpy().astype(np.float64)
        else:
            mat = np.asarray(power_matrix, dtype=np.float64)

        if mat.ndim != 2:
            raise ValueError(f"Expected 2D matrix [F, T], got shape {mat.shape}")

        F_bins, T_bins = mat.shape
        if F_bins < 2 or T_bins < 2:
            raise ValueError(f"Matrix must have at least 2 rows and columns, got {mat.shape}")

        margin_left = 70
        margin_right = 80
        margin_top = 50
        margin_bottom = 50

        plot_w = width - margin_left - margin_right
        plot_h = height - margin_top - margin_bottom

        # Log power for dynamic range visualization: 10 * log10(P + eps)
        log_p = 10.0 * np.log10(np.maximum(mat, 1e-12))
        min_val = float(np.min(log_p))
        max_val = float(np.max(log_p))
        val_range = max(1e-6, max_val - min_val)

        dx = plot_w / float(T_bins)
        dy = plot_h / float(F_bins)

        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="background-color: #0f111a; font-family: monospace;">',
            f'<text x="{width / 2}" y="28" fill="#e0e6ed" font-size="14" font-weight="bold" text-anchor="middle">{title}</text>',
        ]

        # Draw spectrogram heatmap cells
        # Note: frequency increases upwards, so row 0 (lowest freq) is at bottom of plot
        for f_idx in range(F_bins):
            # Row index inverted for standard Cartesian frequency orientation
            y_pos = margin_top + plot_h - (f_idx + 1) * dy
            for t_idx in range(T_bins):
                x_pos = margin_left + t_idx * dx
                norm_val = (log_p[f_idx, t_idx] - min_val) / val_range
                color = _viridis_colormap(norm_val)
                svg_parts.append(
                    f'<rect x="{x_pos:.1f}" y="{y_pos:.1f}" width="{dx + 0.5:.1f}" height="{dy + 0.5:.1f}" fill="{color}" stroke="none" />'
                )

        # Plot bounding box
        svg_parts.append(
            f'<rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#4a5568" stroke-width="1.5" />'
        )

        # X-axis ticks (Time in seconds)
        num_x_ticks = 5
        for i in range(num_x_ticks):
            frac = i / float(num_x_ticks - 1)
            t_val = frac * (T_bins * self.bin_width_s)
            x_pos = margin_left + frac * plot_w
            svg_parts.append(f'<line x1="{x_pos:.1f}" y1="{margin_top + plot_h}" x2="{x_pos:.1f}" y2="{margin_top + plot_h + 6}" stroke="#718096" stroke-width="1" />')
            svg_parts.append(f'<text x="{x_pos:.1f}" y="{margin_top + plot_h + 20}" fill="#a0aec0" font-size="10" text-anchor="middle">{t_val:.2f}s</text>')

        # Y-axis ticks (Frequency in Hz)
        num_y_ticks = 5
        max_f = freq_grid_hz[-1] if freq_grid_hz else (F_bins * 2.0)
        min_f = freq_grid_hz[0] if freq_grid_hz else 0.0

        for j in range(num_y_ticks):
            frac = j / float(num_y_ticks - 1)
            f_val = min_f + frac * (max_f - min_f)
            y_pos = margin_top + plot_h - frac * plot_h
            svg_parts.append(f'<line x1="{margin_left - 6}" y1="{y_pos:.1f}" x2="{margin_left}" y2="{y_pos:.1f}" stroke="#718096" stroke-width="1" />')
            svg_parts.append(f'<text x="{margin_left - 10}" y="{y_pos + 4:.1f}" fill="#a0aec0" font-size="10" text-anchor="end">{f_val:.0f}Hz</text>')

        # Colorbar
        cb_x = width - margin_right + 20
        cb_w = 15
        cb_h = plot_h
        for step in range(50):
            step_frac = step / 50.0
            cb_y = margin_top + plot_h - (step + 1) * (cb_h / 50.0)
            cb_col = _viridis_colormap(step_frac)
            svg_parts.append(f'<rect x="{cb_x}" y="{cb_y:.1f}" width="{cb_w}" height="{(cb_h / 50.0) + 0.5:.1f}" fill="{cb_col}" stroke="none" />')

        svg_parts.append(f'<rect x="{cb_x}" y="{margin_top}" width="{cb_w}" height="{cb_h}" fill="none" stroke="#4a5568" stroke-width="1" />')
        svg_parts.append(f'<text x="{cb_x + cb_w + 5}" y="{margin_top + 10}" fill="#a0aec0" font-size="9">{max_val:.0f}dB</text>')
        svg_parts.append(f'<text x="{cb_x + cb_w + 5}" y="{margin_top + plot_h}" fill="#a0aec0" font-size="9">{min_val:.0f}dB</text>')

        svg_parts.append("</svg>")
        return "\n".join(svg_parts)
