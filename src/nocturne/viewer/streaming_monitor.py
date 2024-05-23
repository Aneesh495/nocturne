"""Streaming performance monitor and latency histogram visualizer."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


class StreamingMonitorVisualizer:
    """Renders real-time latency histograms, buffer fill levels, and throughput metrics."""

    @staticmethod
    def render_ascii(
        latencies_ms: List[float] | np.ndarray,
        buffer_sizes: Optional[List[int] | np.ndarray] = None,
        deadline_ms: float = 20.0,
    ) -> str:
        """Render lightweight ASCII diagnostic summary for terminal monitoring.

        Args:
            latencies_ms: List or array of per-chunk step latencies in milliseconds.
            buffer_sizes: Optional history of buffer lengths.
            deadline_ms: Real-time deadline threshold in milliseconds.

        Returns:
            Formatted ASCII text summary.
        """
        arr = np.asarray(latencies_ms, dtype=float)
        if len(arr) == 0:
            return "StreamingMonitor: No latency data recorded."

        p50 = float(np.percentile(arr, 50))
        p95 = float(np.percentile(arr, 95))
        p99 = float(np.percentile(arr, 99))
        max_lat = float(np.max(arr))
        mean_lat = float(np.mean(arr))
        violations = int(np.sum(arr > deadline_ms))

        lines = [
            "=== Nocturne Online Streaming Latency Monitor ===",
            f"Total Chunks Processed : {len(arr)}",
            f"Mean Latency           : {mean_lat:.2f} ms",
            f"p50 Latency            : {p50:.2f} ms",
            f"p95 Latency            : {p95:.2f} ms",
            f"p99 Latency            : {p99:.2f} ms",
            f"Max Latency            : {max_lat:.2f} ms",
            f"Deadline ({deadline_ms:.1f}ms) : {violations} violations ({violations / len(arr) * 100:.1f}%)",
        ]

        if buffer_sizes is not None and len(buffer_sizes) > 0:
            b_arr = np.asarray(buffer_sizes, dtype=int)
            lines.append(f"Buffer Occupancy (min/mean/max): {int(np.min(b_arr))}/{np.mean(b_arr):.1f}/{int(np.max(b_arr))}")

        # ASCII mini-histogram
        bins = np.linspace(0.0, max(deadline_ms * 1.5, max_lat), 10)
        hist, _ = np.histogram(arr, bins=bins)
        lines.append("Latency Distribution:")
        max_count = max(1, int(np.max(hist)))
        for i in range(len(hist)):
            bar_len = int((hist[i] / max_count) * 25)
            bar = "#" * bar_len
            low, high = bins[i], bins[i + 1]
            lines.append(f"  [{low:5.1f} - {high:5.1f} ms]: {bar:<25} ({hist[i]})")

        return "\n".join(lines)

    @staticmethod
    def render_svg(
        latencies_ms: List[float] | np.ndarray,
        buffer_sizes: Optional[List[int] | np.ndarray] = None,
        deadline_ms: float = 20.0,
        width: int = 800,
        height: int = 420,
        title: str = "Online Streaming Latency & Buffer Monitor",
    ) -> str:
        """Generate vector SVG performance dashboard.

        Args:
            latencies_ms: Step latencies in milliseconds.
            buffer_sizes: Optional buffer occupancy sequence.
            deadline_ms: Real-time budget limit in milliseconds.
            width: SVG width.
            height: SVG height.
            title: Dashboard title.

        Returns:
            Complete SVG XML string.
        """
        arr = np.asarray(latencies_ms, dtype=float)
        if len(arr) == 0:
            arr = np.array([0.0])

        p50 = float(np.percentile(arr, 50))
        p95 = float(np.percentile(arr, 95))
        p99 = float(np.percentile(arr, 99))
        max_lat = max(float(np.max(arr)), deadline_ms * 1.2)

        margin_left = 60
        margin_right = 30
        margin_top = 50
        panel_gap = 40
        panel_w = (width - margin_left - margin_right - panel_gap) / 2
        panel_h = height - margin_top - 50

        svg_parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
            f'<rect width="{width}" height="{height}" fill="#090D16"/>',
            f'<text x="{margin_left}" y="32" fill="#F3F4F6" font-family="monospace" font-size="16" font-weight="bold">{title}</text>',
        ]

        # Panel 1: Latency timeline
        p1_x = margin_left
        p1_y = margin_top
        svg_parts.extend([
            f'<rect x="{p1_x}" y="{p1_y}" width="{panel_w}" height="{panel_h}" fill="#111827" stroke="#374151"/>',
            f'<text x="{p1_x + 10}" y="{p1_y + 20}" fill="#9CA3AF" font-family="monospace" font-size="12">Step Latency Timeline (ms)</text>',
        ])

        # Deadline line
        deadline_y = p1_y + panel_h - (deadline_ms / max_lat) * panel_h
        svg_parts.append(
            f'<line x1="{p1_x}" y1="{deadline_y:.1f}" x2="{p1_x + panel_w}" y2="{deadline_y:.1f}" stroke="#EF4444" stroke-dasharray="4,4" stroke-width="1.5"/>'
        )
        svg_parts.append(
            f'<text x="{p1_x + panel_w - 5}" y="{deadline_y - 4:.1f}" fill="#EF4444" font-family="monospace" font-size="10" text-anchor="end">Deadline {deadline_ms:.0f}ms</text>'
        )

        # Plot latency points
        N_pts = len(arr)
        if N_pts > 1:
            pts = []
            for i, lat in enumerate(arr):
                x = p1_x + (i / (N_pts - 1)) * panel_w
                y = p1_y + panel_h - (min(lat, max_lat) / max_lat) * panel_h
                pts.append(f"{x:.1f},{y:.1f}")
            svg_parts.append(
                f'<polyline points="{" ".join(pts)}" fill="none" stroke="#38BDF8" stroke-width="1.5"/>'
            )

        # Panel 2: Latency Histogram & Percentiles
        p2_x = p1_x + panel_w + panel_gap
        p2_y = margin_top
        svg_parts.extend([
            f'<rect x="{p2_x}" y="{p2_y}" width="{panel_w}" height="{panel_h}" fill="#111827" stroke="#374151"/>',
            f'<text x="{p2_x + 10}" y="{p2_y + 20}" fill="#9CA3AF" font-family="monospace" font-size="12">Latency Distribution & Percentiles</text>',
        ])

        # Compute histogram bars
        num_bins = 12
        bins = np.linspace(0.0, max_lat, num_bins + 1)
        hist, _ = np.histogram(arr, bins=bins)
        max_h = max(1, int(np.max(hist)))
        bar_w = panel_w / num_bins

        for b_idx in range(num_bins):
            b_cnt = hist[b_idx]
            bar_h = (b_cnt / max_h) * (panel_h - 40)
            bx = p2_x + b_idx * bar_w
            by = p2_y + panel_h - bar_h
            col = "#34D399" if bins[b_idx + 1] <= deadline_ms else "#F87171"
            svg_parts.append(
                f'<rect x="{bx + 1:.1f}" y="{by:.1f}" width="{bar_w - 2:.1f}" height="{bar_h:.1f}" fill="{col}" fill-opacity="0.8"/>'
            )

        # Percentile metrics summary table
        metrics_y = p2_y + 40
        metrics_info = [
            ("p50 Median", f"{p50:.2f} ms", "#E5E7EB"),
            ("p95 Bound", f"{p95:.2f} ms", "#FBBF24"),
            ("p99 Bound", f"{p99:.2f} ms", "#F87171"),
        ]
        for idx, (label, val, c_txt) in enumerate(metrics_info):
            row_y = metrics_y + idx * 18
            svg_parts.append(
                f'<text x="{p2_x + 15}" y="{row_y}" fill="#9CA3AF" font-family="monospace" font-size="11">{label}:</text>'
            )
            svg_parts.append(
                f'<text x="{p2_x + 130}" y="{row_y}" fill="{c_txt}" font-family="monospace" font-size="11" font-weight="bold">{val}</text>'
            )

        svg_parts.append("</svg>")
        return "\n".join(svg_parts)

    @classmethod
    def save_svg(
        cls,
        output_path: str | Path,
        latencies_ms: List[float] | np.ndarray,
        buffer_sizes: Optional[List[int] | np.ndarray] = None,
        deadline_ms: float = 20.0,
        title: str = "Online Streaming Latency & Buffer Monitor",
    ) -> Path:
        """Render and save streaming monitor SVG to disk."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        svg_content = cls.render_svg(
            latencies_ms=latencies_ms,
            buffer_sizes=buffer_sizes,
            deadline_ms=deadline_ms,
            title=title,
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        return path
