# Nocturne: Probabilistic Learning of Neural Population Dynamics

Nocturne is an open research laboratory investigating the structure of neural population activity through probabilistic point-process models, variational latent dynamics, and masked spatiotemporal attention.

The central scientific question addressed by Nocturne is whether neural models that achieve high predictive accuracy on held-out spike observations reliably recover the underlying dynamical structure of the neural circuit, or whether predictive fidelity and dynamical recovery diverge under partial observations, unknown inputs, and causal observation restrictions.

## Core Questions

1. **Prediction versus Dynamical Recovery**: Does optimizing for held-out point-process log-likelihood guarantee identification of true dynamical attractors, fixed points, and rotational modes, or does likelihood saturate before dynamical geometry is resolved?
2. **Observation Masks and Identifiability**: How do random neuron dropout, co-smoothing masks, and causal time-filtering affect latent coordinate recovery across known dynamical regimes?
3. **Inferred Drive versus Autonomous Flow**: In nonlinear latent models with inferred external inputs, does external drive over-explain trial-to-trial variability at the expense of learning genuine autonomous vector fields?
4. **Causal Online Inference**: Can spatiotemporal attention architectures and variational filters stream neural predictions in real time while maintaining strict invariance to future observations?

## Architecture Overview

Nocturne provides an end-to-end pipeline spanning continuous spike recordings to gauge-invariant latent geometry:

```
[Raw Spike Trains / NWB Trials]
              │
              ▼
   [Contracts & Valid Bins] ────► [Observation Masks (Observed / Held-out / Causal)]
              │
              ├─────────────────────────────────────────────────┐
              ▼                                                 ▼
[Variational Latent Dynamics (VLDM)]         [Axial Spatiotemporal Attention (STA)]
  • Bidirectional observation encoder          • Factorized time and population attention
  • Nonlinear neural transition prior          • Positive-rate Poisson likelihood readout
  • Inferred vs autonomous drive switch        • Strict causal prefix-masking mode
              │                                                 │
              └───────────────────────┬─────────────────────────┘
                                      ▼
                      [Offline Co-Smoothing & Online Causal Streaming]
                                      │
                                      ▼
                      [Evaluation & Geometric Diagnostics]
                        • Independent bits-per-spike scoring
                        • Fixed-point finding and local Jacobians
                        • Gauge-aware Procrustes and subspace alignment
                        • Behavioral kinematic decoding probes
```

## Dataset Scope

Nocturne evaluates models across both licensed experimental recordings and controlled synthetic circuits:

- **Public Recordings (Neural Latents Benchmark / DANDI)**:
  - `MC_Maze`: Primary motor and premotor cortex during delayed reach execution with obstacles.
  - `MC_RTT`: Motor cortex during continuous random target tracking.
  - `Area2_Bump`: Somatosensory area 2 activity during mechanical limb perturbations.
- **Controlled Synthetic Circuits**:
  - Damped linear oscillation with known complex conjugate eigenvalues.
  - Van der Pol and Stuart-Landau limit cycle oscillators.
  - Bistable double-well memory attractors.
  - Driven nonlinear recurrent networks with task-specific input pulses.

## Getting Started

### Installation

Nocturne requires Python 3.12 and PyTorch. Using `uv` or standard pip:

```bash
# Set up virtual environment and install dependencies
make bootstrap

# Verify toolchains and device capabilities
make doctor

# Install Nocturne package in editable mode
make build

# Run unit tests and mathematical checks
make test
```

### Reproducing Baseline Models and Training

```bash
# Acquire and validate dataset manifests
make data

# Train variational latent dynamics and axial attention models
make train

# Evaluate held-out spike likelihood and bits-per-spike
make evaluate

# Run the interactive scientific viewer
make demo
```

## Repository Structure

```
nocturne/
├── src/nocturne/
│   ├── contracts/    # Data schemas, trial boundaries, and mask definitions
│   ├── recordings/   # NWB/DANDI loaders, provenance, and split management
│   ├── binning/      # Temporal binning, count checks, and reference likelihoods
│   ├── synthetic/    # Ground-truth dynamical systems and spike generators
│   ├── models/       # Baselines, Variational Latent Dynamics, Axial Attention
│   ├── inference/    # Co-smoothing, causal filtering, and prefix invariance
│   ├── training/     # Multi-task training engine and atomic checkpointing
│   ├── streaming/    # Causal online streaming engine and memory bounds
│   ├── geometry/     # Gauge alignment, fixed points, and Jacobian diagnostics
│   ├── evaluation/   # Independent scoring, calibration, and behavioral probes
│   ├── artifacts/    # Serialized experiment artifacts and metadata bundles
│   ├── viewer/       # Interactive visualizer for rasters, rates, and latents
│   └── cli.py        # Command line interface
├── tests/            # Test suite spanning mathematical and semantic checks
├── docs/             # Architecture, derivations, ADRs, and experiment plans
└── Makefile          # Unified development, verification, and benchmark targets
```

## License

MIT License. See `LICENSE` for details.
