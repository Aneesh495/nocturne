# Changelog

All notable changes to Nocturne are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-10-10

### Added

#### Core Contracts & Data Architecture
- Typed Pydantic and dataclass contracts for temporal intervals, spatial electrode channels, unit qualities, trial structures, and splits.
- Inhomogeneous Poisson point-process GLM likelihood with Softplus rate activation.
- Offline and local dataset adapters for MC_Maze, MC_RTT, and Area2_Bump benchmark fixtures post-EvalAI closure.

#### Generative Models & Physics-Informed Dynamics
- Symplectic Störmer-Verlet integrator preserving Hamiltonian energy and canonical 2-form invariants.
- Continuous-time Variational Latent Dynamics Model (VLDM) parameterized by Neural ODE flows with adjoint sensitivity rollouts.
- Spatiotemporal decomposed axial attention model with relative rotary position embeddings (RoPE) and sparse spatial electrode connectivity.
- Classical and probabilistic baselines: Low-Rank Poisson Factor Model, Mean Rate Predictor, and PLDS with Laplace Gaussian approximation.

#### Differential Geometry & Topology
- Numerical equilibrium and slow point discovery via kinetic energy gradient descent.
- Exact Jacobian autograd linearization, eigenspectrum diagnostics, and topological stability classification (sinks, saddles, spirals).
- Extrinsic Frenet-Serret trajectory curvature, torsion, and integrated path length calculator.
- Local tangent space PCA participation ratio estimator and Grassmannian subspace rotation drift analyzer.
- Orthogonal Procrustes gauge alignment with reflection handling and train-only fitting.

#### Online Real-Time Causal Streaming & Closed-Loop Neuromodulation
- Bounded sliding ring buffer maintaining O(max_window_bins) resident memory footprint for arbitrary session lengths.
- Online causal streaming inference coordinator with sub-5ms step latency profiling.
- Closed-loop responsive neurostimulation engine with electrical artifact blanking, minimum ISI lockouts, and session energy budgets.
- Fault-tolerance fuzzing suite passing 150 degraded and corrupted hardware stream scenarios.

#### Scientific Benchmarking, Falsification & Ablations
- Five-family ablation study matrix (objective, masking, capacity, causality, dynamics drive).
- Adversarial attractor falsifier utilizing phase-randomized Fourier surrogate data and Ornstein-Uhlenbeck stochastic diffusion controls.
- Hyperparameter sensitivity analyzer with discrete secant distance elbow knee detection.
- Comprehensive system verification suite catching negative controls NC01 through NC12.

#### Automated Workflows & Standalone Examples
- Six automated scientific workflows: synthetic pipeline, streaming simulation, manifold geometry, cross-condition generalization (LOCO), knowledge distillation, and reproducibility audit.
- Eight standalone, executable Python example scripts in `examples/` with dedicated automated regression test suite.

#### Command-Line Interface & Tooling
- Unified `nocturne` CLI dispatcher supporting 14 subcommands: `doctor`, `synthetic`, `verify`, `acceptance`, `benchmark`, `ablate`, `stream`, `raster`, `phase_portrait`, `matrix`, `falsify`, `sensitivity`, `spectrogram`, `manifold`, and `workflow`.
- Standalone vector SVG and terminal ASCII visualizers for spike rasters, continuous phase portraits, power spectrograms, 3D isometric manifolds, and streaming latency monitors.
- Cryptographic acceptance bundle generator emitting signed SHA-256 manifests across capability registers B01-B32 and gates BA01-BA12.
