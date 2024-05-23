"""Spike raster rendering and predicted firing rate trajectory overlays."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import numpy as np


class RasterVisualizer:
    """Renders vector SVG and ASCII spike rasters with predicted rate overlays."""

    @staticmethod
    def render_ascii(
        counts: np.ndarray,
        neuron_indices: Optional[List[int]] = None,
        max_time_bins: int = 60,
    ) -> str:
        """Render lightweight ASCII raster display for CLI terminals.

        Args:
            counts: Spike count array [T, N].
            neuron_indices: Neurons to display.
            max_time_bins: Temporal bins to render.

        Returns:
            Multi-line string representation.
        """
        T, N = counts.shape
        T_show = min(T, max_time_bins)
        neurons = neuron_indices if neuron_indices is not None else list(range(min(N, 16)))

        lines = [f"=== Nocturne Spike Raster (T={T_show} bins, N={len(neurons)} neurons) ==="]
        for n in neurons:
            row_str = f"N{n:02d} |"
            for t in range(T_show):
                c = int(counts[t, n])
                if c == 0:
                    row_str += " "
                elif c == 1:
                    row_str += "|"
                elif c == 2:
                    row_str += ":"
                else:
                    row_str += "#"
            lines.append(row_str)
        lines.append("-" * (T_show + 7))
        return "\n".join(lines)

    @staticmethod
    def render_svg(
        counts: np.ndarray,
        rates_hz: Optional[np.ndarray] = None,
        neuron_indices: Optional[List[int]] = None,
        bin_width_s: float = 0.02,
        width: int = 800,
        height: int = 400,
        title: str = "Neural Population Activity & Inferred Rates",
    ) -> str:
        """Generate publication-ready standalone vector SVG graphic.

        Args:
            counts: Spike count array [T, N].
            rates_hz: Optional predicted rates [T, N] in Hz.
            neuron_indices: Selected neurons to display.
            bin_width_s: Bin duration in seconds.
            width: SVG width in pixels.
            height: SVG height in pixels.
            title: Figure title.

        Returns:
            Complete SVG XML document as string.
        """
        T, N = counts.shape
        neurons = neuron_indices if neuron_indices is not None else list(range(min(N, 20)))
        num_n = len(neurons)

        margin_left = 60
        margin_right = 40
        margin_top = 50
        margin_bottom = 40

        plot_w = width - margin_left - margin_right
        plot_h = height - margin_top - margin_bottom

        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
            f'<rect width="{width}" height="{height}" fill="#111827"/>',  # Dark theme background
            f'<text x="{margin_left}" y="32" fill="#F9FAFB" font-family="monospace" font-size="16" font-weight="bold">{title}</text>',
            f'<rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" fill="#1F2937" stroke="#374151"/>',
        ]

        # Draw grid and spikes
        row_height = plot_h / max(1, num_n)

        for row_idx, n in enumerate(neurons):
            y_baseline = margin_top + (row_idx + 0.5) * row_height
            # Neuron label
            svg_parts.append(
                f'<text x="{margin_left - 8}" y="{y_baseline + 4}" fill="#9CA3AF" font-family="monospace" font-size="10" text-anchor="end">N{n:02d}</text>'
            )

            # Draw spikes as vertical tick marks
            for t in range(T):
                c = int(counts[t, n])
                if c > 0:
                    x_t = margin_left + (t / max(1, T - 1)) * plot_w
                    y_top = y_baseline - row_height * 0.4
                    y_bot = y_baseline + row_height * 0.4
                    opacity = min(1.0, 0.4 + 0.3 * c)
                    svg_parts.append(
                        f'<line x1="{x_t:.1f}" y1="{y_top:.1f}" x2="{x_t:.1f}" y2="{y_bot:.1f}" stroke="#60A5FA" stroke-width="1.5" stroke-opacity="{opacity}"/>'
                    )

            # Draw rate overlay curve if provided
            if rates_hz is not None:
                max_rate = max(1.0, float(np.max(rates_hz[:, n])))
                points = []
                for t in range(T):
                    x_t = margin_left + (t / max(1, T - 1)) * plot_w
                    norm_rate = float(rates_hz[t, n]) / max_rate
                    y_r = y_baseline + row_height * 0.4 - norm_rate * (row_height * 0.8)
                    points.append(f"{x_t:.1f},{y_r:.1f}")

                polyline_pts = " ".join(points)
                svg_parts.append(
                    f'<polyline points="{polyline_pts}" fill="none" stroke="#F59E0B" stroke-width="1.5" stroke-opacity="0.85"/>'
                )

        # Time axis label
        duration_s = T * bin_width_s
        svg_parts.append(
            f'<text x="{margin_left + plot_w / 2}" y="{height - 12}" fill="#9CA3AF" font-family="monospace" font-size="12" text-anchor="middle">Time (0.0s to {duration_s:.2f}s)</text>'
        )

        svg_parts.append("</svg>")
        return "\n".join(svg_parts)

    @classmethod
    def save_svg(
        cls,
        output_path: str | Path,
        counts: np.ndarray,
        rates_hz: Optional[np.ndarray] = None,
        neuron_indices: Optional[List[int]] = None,
        bin_width_s: float = 0.02,
        title: str = "Neural Population Activity & Inferred Rates",
    ) -> Path:
        """Render and save SVG graphic directly to file.

        Args:
            output_path: Destination path.
            counts: Count array.
            rates_hz: Predicted rates.
            neuron_indices: Channels.
            bin_width_s: Bin duration.
            title: Title.

        Returns:
            Resolved Path of saved file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        svg_content = cls.render_svg(
            counts=counts,
            rates_hz=rates_hz,
            neuron_indices=neuron_indices,
            bin_width_s=bin_width_s,
            title=title,
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        return path
