# Automated Research Laboratory Workflows

Nocturne packages reproducible, end-to-end scientific workflows that automate multi-step numerical pipelines, model training loops, differential geometric audits, and streaming benchmarks.

Every workflow is importable directly from `nocturne.workflows` and returns a strongly-typed frozen dataclass encapsulating numerical metrics, diagnostics, and vector visualizations.

---

## 1. Synthetic Circuit Simulation & Geometry Pipeline

### Entrypoint
`nocturne.workflows.run_synthetic_pipeline`

### Scientific Objective
Simulates controlled nonlinear neural circuits, fits a low-rank Poisson factor model, optimizes vector field kinetic energy to locate equilibria and slow points, calculates Frenet-Serret trajectory curvature, and renders a continuous phase portrait.

### Key Parameters
- `regime_name`: Circuit regime (`linear_oscillator`, `nonlinear_oscillator`, `bistable_memory`, `driven_recurrent`). Default: `nonlinear_oscillator`.
- `num_trials`: Number of independent realizations. Default: `5`.
- `num_neurons`: Population channel dimension $N$. Default: `16`.
- `duration_s`: Trial length in seconds. Default: `1.0`.
- `bin_width_ms`: Sampling bin width. Default: `20`.
- `seed`: Random generator seed. Default: `42`.

### Returned Data Structure
`SyntheticPipelineResult`:
- `total_spikes`: Total discrete spike count.
- `identified_fixed_points`: Count of localized fixed points with $\|f(z)\| \le 10^{-2}$.
- `mean_trajectory_curvature`: Mean extrinsic path curvature $\kappa(t)$.
- `reconstruction_r2`: Deviance pseudo-$R^2$ of low-rank factors.
- `svg_phase_portrait`: Vector SVG string rendering continuous vector flow and orbits.

---

## 2. Online Causal Streaming & Closed-Loop Stimulation

### Entrypoint
`nocturne.workflows.run_streaming_simulation`

### Scientific Objective
Evaluates real-time low-latency causal streaming inference over bounded sliding history windows, monitors sub-5ms latency budgets, and drives closed-loop neurostimulation feedback with electrical artifact blanking and refractory lockout interlocks.

### Key Parameters
- `num_neurons`: Number of monitored neural channels. Default: `16`.
- `latent_dim`: Latent state bottleneck dimensionality. Default: `4`.
- `total_bins`: Total discrete time bins processed. Default: `100`.
- `chunk_size_bins`: Bins arriving per hardware tick (e.g. 1 bin = 50 Hz streaming loop). Default: `2`.
- `rate_threshold_hz`: Firing rate threshold triggering stimulation. Default: `35.0`.
- `min_isi_s`: Minimum inter-stimulus interval lockout. Default: `0.05`.
- `blanking_duration_s`: Sensor blanking artifact duration. Default: `0.02`.

### Returned Data Structure
`StreamingSimulationResult`:
- `mean_latency_ms`: Mean step processing wall time.
- `p95_latency_ms`: 95th percentile latency.
- `max_latency_ms`: Peak latency jitter.
- `deadline_violations`: Chunks exceeding processing budget.
- `num_stim_events`: Count of dispatched closed-loop stimulation pulses.
- `num_blanked_bins`: Bins zeroed out during active electrical artifact blanking.
- `reconstruction_rmse`: Root-mean-squared error against true firing rates.
- `ascii_summary`: Lightweight ASCII terminal latency monitor table.
- `svg_monitor`: Vector SVG performance dashboard.
- `stim_events`: Complete event records with timestamps, trigger types, and energy units.

---

## 3. Differential Manifold Geometry & Grassmannian Drift

### Entrypoint
`nocturne.workflows.run_manifold_geometry_workflow`

### Scientific Objective
Audits the intrinsic differential geometry and topological stability of neural population state spaces across experimental conditions, computing Frenet-Serret curvature, local tangent space participation ratios, Grassmannian subspace rotation drift, and orthogonal Procrustes gauge disparities.

### Key Parameters
- `regime_name`: Circuit regime. Default: `nonlinear_oscillator`.
- `num_conditions`: Number of conditions or trajectory paths evaluated. Default: `4`.
- `num_neurons`: Population channel dimension. Default: `16`.
- `elevation_deg`: 3D isometric camera elevation angle. Default: `30.0`.
- `azimuth_deg`: 3D isometric camera azimuth angle. Default: `45.0`.

### Returned Data Structure
`ManifoldGeometryWorkflowResult`:
- `mean_path_length`: Integrated arc length along trajectories.
- `mean_speed`: Mean latent velocity magnitude $\|\dot{z}(t)\|$.
- `mean_curvature`: Differential Frenet-Serret curvature $\kappa(t)$.
- `mean_intrinsic_dim`: Mean local participation ratio dimensionality.
- `mean_grassmann_drift`: Subspace rotation distance between adjacent tangent planes.
- `mean_procrustes_disparity`: Gauge-invariant Procrustes disparity between conditions.
- `svg_manifold_mesh`: 3D isometric vector SVG manifold orbit.

---

## 4. Cross-Condition Generalization Campaign (LOCO)

### Entrypoint
`nocturne.workflows.run_generalization_campaign`

### Scientific Objective
Performs rigorous Leave-One-Condition-Out (LOCO) cross-validation to assess out-of-distribution robustness of latent representations when decoding kinematics or behavior across unfamiliar movement conditions.

### Key Parameters
- `regime_name`: Circuit regime. Default: `driven_recurrent`.
- `num_conditions`: Total conditions. Default: `4`.
- `trials_per_condition`: Trials per condition. Default: `6`.
- `gap_threshold`: Maximum allowable generalization gap before failing robustness. Default: `0.35`.

### Returned Data Structure
`GeneralizationCampaignResult`:
- `in_distribution_mean_r2`: Decoding $R^2$ when condition is seen during training.
- `out_of_distribution_mean_r2`: Held-out decoding $R^2$ on unobserved conditions.
- `generalization_gap`: In-distribution minus out-of-distribution performance delta.
- `is_robust_generalizer`: Boolean flag confirming gap is within budget.
- `condition_scores`: Per-condition held-out $R^2$ scores.
- `summary_markdown`: Markdown table formatting cross-condition results.

---

## 5. Teacher-Student Knowledge Distillation

### Entrypoint
`nocturne.workflows.run_distillation_workflow`

### Scientific Objective
Compresses high-capacity offline bidirectional models into low-latency causal streaming models by minimizing rate I-divergence (generalized Poisson relative entropy) alongside latent subspace alignment.

### Key Parameters
- `teacher_d_model`: Offline teacher hidden dimension. Default: `32`.
- `student_d_model`: Streaming student hidden dimension. Default: `16`.
- `alpha`: Weight for rate distillation loss in $[0, 1]$. Default: `0.6`.
- `beta`: Weight for latent subspace alignment. Default: `0.2`.
- `num_steps`: Optimization steps. Default: `15`.

### Returned Data Structure
`DistillationWorkflowResult`:
- `parameter_compression_ratio`: Ratio of teacher to student parameters.
- `teacher_student_rate_correlation`: Pearson correlation between student and teacher rate fields.
- `teacher_latency_ms`: Mean forward pass wall time of teacher.
- `student_latency_ms`: Mean forward pass wall time of student.
- `latency_speedup_factor`: Relative inference speedup.
- `loss_history`: Step-by-step distillation training trajectory.

---

## 6. Deterministic Bitwise Reproducibility Audit

### Entrypoint
`nocturne.workflows.run_reproducibility_audit`

### Scientific Objective
Verifies numerical determinism, bitwise parameter replication across identical random seeds, SHA-256 state hash equality, and non-trivial cross-seed divergence.

### Key Parameters
- `seed`: Canonical seed tested. Default: `42`.
- `num_steps`: Optimization steps. Default: `5`.

### Returned Data Structure
`ReproducibilityAuditResult`:
- `is_bitwise_identical`: True if weights and losses match within $10^{-7}$.
- `sha256_run1`: SHA-256 checksum of model parameters in Run 1.
- `sha256_run2`: SHA-256 checksum of model parameters in Run 2.
- `cross_seed_divergence`: Mean weight norm divergence when seed is varied.
- `audit_passed`: Boolean confirmation of deterministic reproducibility.
- `summary_report`: ASCII verification report.
