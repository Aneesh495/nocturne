"""Tests for scientific visualizers, SVG rendering, and HTML dashboards."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from nocturne.viewer.raster import RasterVisualizer
from nocturne.viewer.phase_portrait import PhasePortraitVisualizer
from nocturne.viewer.streaming_monitor import StreamingMonitorVisualizer
from nocturne.viewer.dashboard import DashboardVisualizer


def test_raster_visualizer_ascii_and_svg(tmp_path: Path):
    T, N = 50, 10
    np.random.seed(42)
    counts = np.random.poisson(lam=0.5, size=(T, N)).astype(float)
    rates_hz = np.ones((T, N)) * 15.0

    # ASCII render
    ascii_str = RasterVisualizer.render_ascii(counts, max_time_bins=20)
    assert "Nocturne Spike Raster" in ascii_str
    assert "N00 |" in ascii_str

    # SVG render
    svg_str = RasterVisualizer.render_svg(counts, rates_hz=rates_hz)
    assert "<svg" in svg_str
    assert "</svg>" in svg_str
    assert "Neural Population Activity" in svg_str

    # Save SVG
    out_file = tmp_path / "test_raster.svg"
    saved_path = RasterVisualizer.save_svg(out_file, counts, rates_hz=rates_hz)
    assert saved_path.exists()
    assert saved_path.stat().st_size > 0


def test_phase_portrait_visualizer(tmp_path: Path):
    T, D = 100, 2
    t = np.linspace(0, 4 * np.pi, T)
    trajectory = np.stack([np.cos(t) * np.exp(-0.05 * t), np.sin(t) * np.exp(-0.05 * t)], axis=-1)
    fixed_pts = np.array([[0.0, 0.0]])
    labels = ["stable spiral"]

    def linear_flow(z: np.ndarray) -> np.ndarray:
        A = np.array([[-0.1, -1.0], [1.0, -0.1]])
        return z @ A.T

    # ASCII
    ascii_out = PhasePortraitVisualizer.render_ascii(trajectory, fixed_points=fixed_pts)
    assert "Latent Phase Plane" in ascii_out

    # SVG
    svg_out = PhasePortraitVisualizer.render_svg(
        trajectories=trajectory,
        flow_fn=linear_flow,
        fixed_points=fixed_pts,
        stability_labels=labels,
    )
    assert "<svg" in svg_out
    assert "Latent Coordinate z_1" in svg_out
    assert "stable spiral" in svg_out

    # Dimension check error
    with pytest.raises(ValueError):
        PhasePortraitVisualizer.render_svg(np.ones((10, 1)))

    # Save
    out_svg = tmp_path / "portrait.svg"
    saved = PhasePortraitVisualizer.save_svg(out_svg, trajectory, fixed_points=fixed_pts)
    assert saved.exists()


def test_streaming_monitor_visualizer(tmp_path: Path):
    latencies = [1.2, 2.5, 3.1, 4.0, 15.2, 22.1, 3.4, 2.9]
    buffer_sizes = [50, 50, 50, 50, 50, 50, 50, 50]

    # ASCII
    ascii_res = StreamingMonitorVisualizer.render_ascii(latencies, buffer_sizes)
    assert "p50 Latency" in ascii_res
    assert "Latency Distribution" in ascii_res

    # Empty case
    assert "No latency data" in StreamingMonitorVisualizer.render_ascii([])

    # SVG
    svg_res = StreamingMonitorVisualizer.render_svg(latencies)
    assert "<svg" in svg_res
    assert "Deadline" in svg_res

    # Save
    out_svg = tmp_path / "stream_monitor.svg"
    saved = StreamingMonitorVisualizer.save_svg(out_svg, latencies)
    assert saved.exists()


def test_dashboard_visualizer(tmp_path: Path):
    metadata = {
        "Model": "VLDM",
        "Dataset": "mc_maze",
        "Seed": 42,
    }
    metrics = {
        "bps_held_in": 0.285,
        "pseudo_r2": 0.412,
    }
    counts = np.zeros((30, 8))
    rates = np.ones((30, 8)) * 10.0
    trajectories = np.random.randn(30, 2)
    latencies = [2.1, 3.2, 4.0]

    html_str = DashboardVisualizer.render_html(
        experiment_name="VLDM Benchmark Test",
        metadata=metadata,
        metrics=metrics,
        counts=counts,
        rates_hz=rates,
        trajectories=trajectories,
        latencies_ms=latencies,
    )
    assert "<!DOCTYPE html>" in html_str
    assert "Nocturne Scientific Report: VLDM Benchmark Test" in html_str
    assert "0.2850" in html_str
    assert "<svg" in html_str

    out_html = tmp_path / "report.html"
    saved = DashboardVisualizer.save_html(
        out_html,
        experiment_name="Test Report",
        metadata=metadata,
        metrics=metrics,
    )
    assert saved.exists()
    assert saved.stat().st_size > 0
