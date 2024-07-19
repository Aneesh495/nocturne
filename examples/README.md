# Nocturne Laboratory Examples

This directory contains standalone, self-contained Python examples demonstrating the core mathematical algorithms, generative models, inference routines, and scientific audits in Nocturne.

Every example is designed to execute directly from the repository root using standard Python with zero external service dependencies.

---

## Example Directory Index

### 1. Controlled Synthetic Circuit Simulation
- **File**: `examples/quickstart_synthetic.py`
- **Description**: Simulates 4 canonical neural dynamical regimes: linear damped oscillator (spiral sink), nonlinear limit cycle oscillator (Hopf bifurcation), bistable memory (double-well attractor), and driven recurrent circuit. Generates spike counts and displays terminal ASCII rasters.
- **Run**:
  ```bash
  python examples/quickstart_synthetic.py
  ```

### 2. Continuous-Time Variational Latent Dynamics (VLDM)
- **File**: `examples/train_vldm_continuous.py`
- **Description**: Configures and trains a continuous-time Variational Latent Dynamics Model with Neural ODE flow prior. Maximizes Poisson evidence lower bound (ELBO) over continuous trajectory rollouts.
- **Run**:
  ```bash
  python examples/train_vldm_continuous.py
  ```

### 3. Spatiotemporal Axial Attention Co-Smoothing
- **File**: `examples/axial_attention_co_smoothing.py`
- **Description**: Employs decomposed spatiotemporal axial attention blocks to reconstruct unobserved held-out neural channels from observed population spikes (co-smoothing protocol).
- **Run**:
  ```bash
  python examples/axial_attention_co_smoothing.py
  ```

### 4. 50 Hz Closed-Loop Real-Time Streaming
- **File**: `examples/realtime_streaming_demo.py`
- **Description**: Simulates 50 Hz (20 ms tick) online causal streaming inference with responsive closed-loop neurostimulation feedback, latency budgeting, and sensory blanking interlocks.
- **Run**:
  ```bash
  python examples/realtime_streaming_demo.py
  ```

### 5. Dynamical Fixed Point Discovery & Jacobian Topology
- **File**: `examples/fixed_point_topology.py`
- **Description**: Discovers vector field equilibria via kinetic energy gradient descent, computes exact Jacobian matrices, and classifies topological stability (stable sinks, saddles, unstable spirals).
- **Run**:
  ```bash
  python examples/fixed_point_topology.py
  ```

### 6. Five-Family Ablation Study Matrix
- **File**: `examples/full_ablation_sweep.py`
- **Description**: Executes the standardized 5-family ablation matrix: loss objective (Poisson vs MSE vs Gaussian), masking (co-smoothing vs unmasked vs dropout), capacity scaling, temporal causality, and dynamics drive.
- **Run**:
  ```bash
  python examples/full_ablation_sweep.py
  ```

### 7. Adversarial Attractor Falsification
- **File**: `examples/adversarial_falsification.py`
- **Description**: Subjects geometric claims to rigorous null-hypothesis falsification tests using phase-randomized Fourier surrogate trajectories and continuous Ornstein-Uhlenbeck stochastic diffusion controls.
- **Run**:
  ```bash
  python examples/adversarial_falsification.py
  ```

### 8. Latent Dimensionality Saturation Knee Sweep
- **File**: `examples/hyperparameter_knee_sweep.py`
- **Description**: Sweeps latent state dimensionality across multiple orders of magnitude, analyzes performance curves, and detects the geometric elbow knee point.
- **Run**:
  ```bash
  python examples/hyperparameter_knee_sweep.py
  ```
