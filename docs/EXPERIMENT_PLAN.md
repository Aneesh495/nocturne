# Nocturne Experiment Plan and Benchmark Execution Strategy

This document specifies the experimental roadmap, parameter grids, compute budgets, and evaluation milestones for the Nocturne probabilistic neural dynamics laboratory.

---

## 1. Experimental Overview and Objectives

The primary scientific objectives of the Nocturne benchmark suite are:
1. **Validation of Point-Process Likelihood Advantage**: Demonstrate that continuous-time neural ODEs and axial attention models trained with Poisson point-process loss consistently outperform Gaussian/MSE baselines on discrete spike data.
2. **Evaluation of Axial Spatiotemporal Attention**: Quantify bits-per-spike (BPS) gains and computational scaling of factorized $\mathcal{O}(BNT^2 + BTN^2)$ axial attention relative to classic static and linear state-space models.
3. **Assessment of Latent Dynamical Topology**: Confirm that learned variational latent spaces preserve true phase portrait geometry, fixed points, and limit cycle attractors in synthetic ground truth systems.
4. **Online Streaming Verification**: Establish that causal attention and bounded ring buffer states achieve equivalent filtering accuracy to offline causal batch passes under hard real-time latency budgets.

---

## 2. Benchmark Dataset Matrix

| Dataset Identifier | Neural Channels ($N$) | Bin Width ($\Delta t$) | Task Paradigm | Primary Evaluation Split |
| :--- | :---: | :---: | :--- | :--- |
| **MC_Maze** (NLB 000128) | 182 | 20 ms | Delayed reach center-out maze | Chronological test split ($B_{\text{test}} = 250$) |
| **MC_RTT** (NLB 000129) | 130 | 20 ms | Continuous random target pursuit | Chronological continuous block ($T_{\text{test}} = 600$) |
| **Area2_Bump** (NLB 000127) | 65 | 20 ms | Passive mechanical limb perturbation | Perturbed vs unperturbed held-out trials |
| **Linear Oscillator** | 16 | 20 ms | Synthetic damped harmonic spiral | 200 held-out phase trajectories |
| **Nonlinear Oscillator** | 16 | 20 ms | Synthetic cubic limit cycle attractor | 200 held-out phase trajectories |
| **Bistable Memory** | 16 | 20 ms | Double-well potential switching | 200 held-out phase trajectories |
| **Driven Recurrent** | 32 | 20 ms | Continuous chaotic drive circuit | 200 held-out phase trajectories |

---

## 3. Evaluated Model Configurations

### Baseline 1: Marginal Mean Rate
- Architecture: Empirical channel-wise average spike rate: $\lambda_n = \frac{1}{B T \Delta t}\sum_{b, t} y_{b, t, n}$.
- Purpose: Serves as the universal null model ($\mathcal{L}_0$) for bits-per-spike normalization.

### Baseline 2: Low-Rank Poisson Factor Model
- Latent Dimension: $d \in \{4, 8, 16\}$.
- Rate Decoder: $\lambda_t = \text{softplus}(W z_t + b)$.
- Training: Static low-rank coordinate estimation via AdamW.

### Baseline 3: Poisson Linear Dynamical System (PLDS)
- Latent Dimension: $d \in \{4, 8\}$.
- Dynamics: $z_{t+1} = A z_t + w_t, \quad w_t \sim \mathcal{N}(0, Q)$.
- Inference: Laplace approximation and causal Kalman smoother.

### Model 1: Variational Latent Dynamics Model (VLDM)
- Latent Dimension: $d = 8$.
- Observation Encoder: Bidirectional GRU (hidden size 64).
- Latent Dynamics: Parameterized neural ODE $dz/dt = f_\theta(z) + B u_t$ with 2-layer MLP vector field.
- Integrator: Fourth-order Runge-Kutta (RK4) with step size equal to bin width.
- Readout: Positive rate decoder with softplus activation.

### Model 2: Axial Spatiotemporal Masked Attention (STA)
- Embedding Dimension: $d_{\text{model}} = 64$.
- Attention Heads: 4 heads per axial dimension.
- Depth: 3 stacked axial blocks (temporal MHA + spatial MHA).
- Feedforward Expansion: 128 units with GELU activation.
- Masking Modes: Causal lower-triangular (filtering/forecasting) vs bidirectional (co-smoothing).

---

## 4. Compute Resource Budget and Execution Timeline

All benchmarks are engineered to execute on Apple Silicon (M-series MPS) and standard Linux x86_64 CPU/GPU environments:
- **Synthetic Circuit Generation**: $\sim 2.5$ seconds per 1,000 trials.
- **Single Benchmark Run**: $\sim 5$ to 15 seconds per model on 20 steps (fast verification mode); $\sim 2$ minutes on full 100-epoch convergence.
- **Full 3-Seed Benchmark Matrix**: 5 models $\times$ 4 datasets $\times$ 3 seeds $= 60$ runs ($\approx 12$ minutes in automated CI).
- **Ablation Matrix**: 5 families $\times$ 3 variants $\times$ 3 seeds $= 45$ runs ($\approx 10$ minutes).
- **Verification and Acceptance Suite**: 12 negative controls $+$ 150 streaming checks ($\approx 25$ seconds).
