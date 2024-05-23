# Visualizer and Scientific Artifact Guide

This document describes the rendering engines, vector SVG outputs, and scientific artifact registry within `nocturne.viewer` and `nocturne.artifacts`.

---

## 1. Architectural Principles

All visualizers in Nocturne operate without mandatory third-party GUI or heavy browser rendering dependencies. They emit clean, standalone vector SVG (Scalable Vector Graphics) and self-contained HTML5 reports. This ensures complete determinism, exact headless server compatibility, and crisp publication-quality scaling.

The visualizer suite comprises four distinct components:
1. `RasterVisualizer`: Spike trains across population channels with overlaid model firing rate predictions $\hat{\lambda}_n(t)$.
2. `PhasePortraitVisualizer`: Continuous 2D/3D latent state trajectories $z(t)$, autonomous flow fields $f_\theta(z)$, and classified dynamical fixed points.
3. `StreamingMonitorVisualizer`: Causal online latency distributions, percentile tracking (p50, p95, p99), and real-time deadline monitoring.
4. `DashboardVisualizer`: Unified multi-panel HTML scientific report combining experimental provenance, quantitative metrics, and embedded SVG figures.

---

## 2. Raster Visualizer (`RasterVisualizer`)

The raster visualizer renders spike counts $y \in \mathbb{N}^{T \times N}$ and optional inferred continuous rates $\hat{\lambda} \in \mathbb{R}_{\ge 0}^{T \times N}$.

### ASCII Terminal Mode
For immediate terminal inspection during training or remote SSH sessions:
```python
from nocturne.viewer import RasterVisualizer

ascii_art = RasterVisualizer.render_ascii(
    counts=counts_matrix,
    max_time_bins=60,
)
print(ascii_art)
```

### Publication SVG Mode
For vector graphics export:
```python
svg_path = RasterVisualizer.save_svg(
    output_path="artifacts/raster_trial_0.svg",
    counts=counts_matrix,
    rates_hz=rates_matrix,
    bin_width_s=0.02,
    title="MC_Maze Trial 0 Spike Activity and VLDM Rates",
)
```

---

## 3. Phase Portrait Visualizer (`PhasePortraitVisualizer`)

The phase portrait engine visualizes continuous latent trajectories in state space.

### Flow Field Sampling
Given an autonomous transition function $f_\theta(z)$, the visualizer evaluates flow vectors on a regular 2D grid covering the trajectory envelope:
$$
\mathbf{v}(z_1, z_2) = f_\theta\begin{pmatrix} z_1 \\ z_2 \\ 0 \\ \vdots \end{pmatrix}_{1:2}
$$
Vector magnitudes are normalized and scaled to prevent visual clutter, with directional opacity reflecting local velocity magnitude.

### Fixed Point Overlays
Identified fixed points $z^*$ are rendered with markers corresponding to their Jacobian stability classification:
- Stable attractors (all $\operatorname{Re}(\lambda_i) < 0$): Emerald markers.
- Saddles (mixed sign eigenvalues): Amber markers.
- Unstable sources (all $\operatorname{Re}(\lambda_i) > 0$): Red markers.

---

## 4. Streaming Monitor (`StreamingMonitorVisualizer`)

For online closed-loop neural decoding, maintaining latency below the bin boundary $\Delta t$ is critical. The streaming monitor tracks:
- Per-chunk step latency sequence $\tau_k$.
- Percentiles $p_{50}, p_{95}, p_{99}$ and maximum latency.
- Real-time deadline threshold violations ($k$ where $\tau_k > \Delta t$).
- Ring buffer occupancy history.

---

## 5. Artifact Registry (`ArtifactRegistry`)

Experimental outputs are registered in a central registry (`artifact_index.json`):
```json
{
  "artifact_id": "vldm_mc_maze_seed42_eval",
  "artifact_type": "eval_bundle",
  "dataset_name": "mc_maze",
  "split_name": "val",
  "model_name": "vldm",
  "seed": 42,
  "metrics": {
    "bps_held_in": 0.321,
    "deviance": 142.5
  },
  "file_path": "reports/mc_maze_vldm.html",
  "sha256_hash": "..."
}
```
All entries are validated against the Pydantic schema in `nocturne.artifacts.schema.ArtifactMetadata`.
