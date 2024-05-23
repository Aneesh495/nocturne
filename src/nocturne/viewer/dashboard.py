"""Scientific experiment dashboard and multi-panel HTML evaluation report generator."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from nocturne.viewer.raster import RasterVisualizer
from nocturne.viewer.phase_portrait import PhasePortraitVisualizer
from nocturne.viewer.streaming_monitor import StreamingMonitorVisualizer


class DashboardVisualizer:
    """Generates comprehensive scientific HTML report dashboards with embedded vector figures."""

    @staticmethod
    def render_html(
        experiment_name: str,
        metadata: Dict[str, Any],
        metrics: Dict[str, float],
        counts: Optional[np.ndarray] = None,
        rates_hz: Optional[np.ndarray] = None,
        trajectories: Optional[np.ndarray] = None,
        fixed_points: Optional[np.ndarray] = None,
        latencies_ms: Optional[List[float] | np.ndarray] = None,
    ) -> str:
        """Render complete standalone scientific HTML report.

        Args:
            experiment_name: Title of experimental report.
            metadata: Experiment provenance (model, dataset, split, seed, git hash).
            metrics: Computed evaluation metrics (BPS, deviance, R2, etc.).
            counts: Optional spike counts [T, N].
            rates_hz: Optional inferred rates [T, N].
            trajectories: Optional latent trajectories [K, T, D] or [T, D].
            fixed_points: Optional fixed points [M, D].
            latencies_ms: Optional streaming step latencies.

        Returns:
            Complete HTML5 document string.
        """
        # Generate SVG components
        raster_svg = ""
        if counts is not None:
            raster_svg = RasterVisualizer.render_svg(
                counts=counts,
                rates_hz=rates_hz,
                width=850,
                height=350,
                title="Spike Raster & Inferred Rate Overlay",
            )

        portrait_svg = ""
        if trajectories is not None and trajectories.shape[-1] >= 2:
            portrait_svg = PhasePortraitVisualizer.render_svg(
                trajectories=trajectories,
                fixed_points=fixed_points,
                width=700,
                height=450,
                title="Latent Phase Plane & Dynamical Trajectories",
            )

        monitor_svg = ""
        if latencies_ms is not None and len(latencies_ms) > 0:
            monitor_svg = StreamingMonitorVisualizer.render_svg(
                latencies_ms=latencies_ms,
                width=850,
                height=300,
                title="Causal Streaming Latency Profile",
            )

        # Build metrics rows
        metric_rows = []
        for k, v in metrics.items():
            fmt_v = f"{v:.4f}" if isinstance(v, float) else str(v)
            metric_rows.append(f"<tr><td>{k}</td><td class='val'>{fmt_v}</td></tr>")
        metric_table_body = "\n".join(metric_rows)

        # Build metadata list
        meta_items = []
        for k, v in metadata.items():
            meta_items.append(f"<li><strong>{k}:</strong> {v}</li>")
        meta_list_body = "\n".join(meta_items)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Nocturne: {experiment_name}</title>
  <style>
    body {{
      background-color: #0B0F19;
      color: #E2E8F0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", monospace;
      margin: 0;
      padding: 30px;
    }}
    .container {{
      max-width: 1100px;
      margin: 0 auto;
    }}
    header {{
      border-bottom: 1px solid #1E293B;
      padding-bottom: 20px;
      margin-bottom: 30px;
    }}
    h1 {{
      font-size: 26px;
      color: #F8FAFC;
      margin: 0 0 8px 0;
    }}
    .subtitle {{
      color: #94A3B8;
      font-size: 14px;
    }}
    .grid-2 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin-bottom: 30px;
    }}
    .card {{
      background: #111827;
      border: 1px solid #1F2937;
      border-radius: 8px;
      padding: 20px;
    }}
    h2 {{
      font-size: 18px;
      color: #38BDF8;
      margin-top: 0;
      margin-bottom: 15px;
      border-bottom: 1px solid #374151;
      padding-bottom: 8px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      padding: 8px 12px;
      text-align: left;
      border-bottom: 1px solid #1F2937;
    }}
    td.val {{
      text-align: right;
      font-family: monospace;
      font-weight: bold;
      color: #34D399;
    }}
    ul.meta {{
      list-style: none;
      padding: 0;
      margin: 0;
      font-size: 13px;
      line-height: 1.8;
    }}
    .figure-card {{
      background: #111827;
      border: 1px solid #1F2937;
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 25px;
      text-align: center;
    }}
    .figure-card svg {{
      max-width: 100%;
      height: auto;
      border-radius: 4px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Nocturne Scientific Report: {experiment_name}</h1>
      <div class="subtitle">Neural Dynamics Laboratory Evaluation & Integrity Audit</div>
    </header>

    <div class="grid-2">
      <div class="card">
        <h2>Provenance & Configuration</h2>
        <ul class="meta">
          {meta_list_body}
        </ul>
      </div>

      <div class="card">
        <h2>Primary Evaluation Metrics</h2>
        <table>
          <thead>
            <tr><th>Metric</th><th style="text-align:right">Value</th></tr>
          </thead>
          <tbody>
            {metric_table_body}
          </tbody>
        </table>
      </div>
    </div>

    {f'<div class="figure-card"><h2>Population Activity & Decoded Rates</h2>{raster_svg}</div>' if raster_svg else ''}
    {f'<div class="figure-card"><h2>Continuous Latent State Geometry</h2>{portrait_svg}</div>' if portrait_svg else ''}
    {f'<div class="figure-card"><h2>Online Causal Stream Latency</h2>{monitor_svg}</div>' if monitor_svg else ''}

  </div>
</body>
</html>
"""
        return html

    @classmethod
    def save_html(
        cls,
        output_path: str | Path,
        experiment_name: str,
        metadata: Dict[str, Any],
        metrics: Dict[str, float],
        counts: Optional[np.ndarray] = None,
        rates_hz: Optional[np.ndarray] = None,
        trajectories: Optional[np.ndarray] = None,
        fixed_points: Optional[np.ndarray] = None,
        latencies_ms: Optional[List[float] | np.ndarray] = None,
    ) -> Path:
        """Render and write HTML report to file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html_str = cls.render_html(
            experiment_name=experiment_name,
            metadata=metadata,
            metrics=metrics,
            counts=counts,
            rates_hz=rates_hz,
            trajectories=trajectories,
            fixed_points=fixed_points,
            latencies_ms=latencies_ms,
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(html_str)
        return path
