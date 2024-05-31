# Benchmark Matrix and Empirical Evaluation Protocol

This document specifies the empirical evaluation methodology, multi-seed comparison protocol, ablation matrices, and robustness sweeps implemented in the Nocturne research laboratory.

---

## 1. Multi-Seed Benchmark Suite

To ensure statistical reliability, all candidate architectures are evaluated across three pinned random seeds ($S \in \{42, 43, 44\}$).

### Target Datasets
1. **MC_Maze (NLB / DANDI 000128)**: Primary motor and premotor cortex multi-electrode array recordings during delayed reach maze task.
2. **MC_RTT (NLB / DANDI 000129)**: Primary motor cortex recording during continuous random target pursuit.
3. **Area2_Bump (NLB / DANDI 000127)**: Somatosensory cortex area 2 recording during passive mechanical limb perturbations.
4. **Synthetic Oscillators & Circuits**: Controlled dynamical ground truth systems (linear oscillator, nonlinear cubic oscillator, bistable memory, driven recurrent circuit).

### Evaluated Model Families
- **MeanRateBaseline**: Stationary marginal Poisson firing rate per channel. Serves as the null model for Bits-Per-Spike (BPS) calculations.
- **LowRankPoissonFactorModel**: Static low-rank matrix factorization with softplus rate mapping.
- **PoissonLinearDynamicalSystem (PLDS)**: Linear latent autoregressive dynamics with Laplace approximation inference.
- **VariationalLatentDynamicsModel (VLDM)**: Continuous-time variational neural ODE latent dynamics with GRU observation encoder.
- **SpatiotemporalAttentionModel (STA)**: Factorized axial spatiotemporal transformer with masked causal/noncausal cross-attention.

---

## 2. Quantitative Evaluation Metrics

All metrics are calculated independently by `nocturne.evaluation` routines:

### Bits Per Spike (BPS)
Measures log-likelihood improvement per emitted spike over the marginal mean rate baseline:
$$\text{BPS} = \frac{\ln \mathcal{L}(\hat{\lambda}) - \ln \mathcal{L}(\lambda_0)}{N_{\text{spikes}} \cdot \ln 2}$$
Strict zero-spike contract: If $N_{\text{spikes}} = 0$ in the evaluated evaluation slice, BPS returns `None` rather than dividing by zero.

### Poisson Deviance
Twice the difference in log-likelihood between the saturated model and the evaluated model:
$$D = 2 \sum_{b, t, n} \left[ y \ln \frac{y}{\hat{\lambda} \Delta t} - (y - \hat{\lambda} \Delta t) \right]$$
Deviance pseudo-$R^2$ is computed as:
$$R_{\text{pseudo}}^2 = 1 - \frac{D(\text{model})}{D(\text{null})}$$

### Kinematic Decoding Probe $R^2$
A linear ridge regression probe ($L_2$ penalty $\alpha = 1.0$) is trained strictly on training latents to predict continuous hand velocity or joint kinematics:
$$R^2 = 1 - \frac{\sum_t (v_t - \hat{v}_t)^2}{\sum_t (v_t - \bar{v})^2}$$
Evaluated macro-averaged across behavioral degrees of freedom on the held-out test split.

---

## 3. Five-Family Ablation Matrix (Gate BA10)

The ablation suite decomposes architectural contributions into five controlled experimental families:

| Family | Variants Evaluated | Theoretical Hypothesis Tested |
| :--- | :--- | :--- |
| **1. Objective** | Poisson NLL, Gaussian NLL, Mean Squared Error (MSE) | Poisson point-process loss correctly models counting shot noise, outperforming continuous quadratic losses. |
| **2. Masking Protocol** | Co-smoothing (held-out neurons), Random token dropout, Unmasked | Spatiotemporal masking forces global population coordination rather than trivial autoregression. |
| **3. Depth Capacity** | Depth = 1 layer, Depth = 2 layers, Depth = 4 layers | Scaling depth improves expressive capacity until saturation or overfitting on short trial sequences. |
| **4. Temporal Causality** | Causal (lower-triangular mask), Bidirectional (full visibility) | Bidirectional attention improves offline smoothing, while causal attention enables online streaming. |
| **5. Latent Dynamics** | Autonomous flow ($dz/dt = f_\theta(z)$), Inferred external drive ($dz/dt = f_\theta(z) + B u_t$) | External drive prevents autonomous dynamics collapse during perturbation epochs. |

---

## 4. Robustness Campaigns (Gate BA08)

Robustness is assessed across two critical dimensions of experimental degradation:

### Forecasting Horizon Degradation
Autoregressive rollouts are evaluated at five distinct future horizons:
$$H \in \{1, 2, 5, 10, 20\} \text{ bins ahead}$$
For 20 ms bins, this corresponds to future horizons of 20 ms, 40 ms, 100 ms, 200 ms, and 400 ms.
- Monotonicity check: deviance should increase smoothly with horizon length without sudden numerical instabilities.

### Missing-Neuron Dropout Gracefulness
Models are evaluated under five channel dropout fractions:
$$p \in \{0.0, 0.1, 0.25, 0.5, 0.75\}$$
- Gracefulness check: at $p = 0.50$ (50% missing channels), performance degrades smoothly; relative loss increase remains bounded ($< 10\times$ baseline) with zero NaNs or negative rates.

---

## 5. Statistical Hypothesis Testing

For comparing model $A$ against model $B$ across matched seeds $S$, Nocturne executes a paired Student $t$-test:
$$t = \frac{\bar{d}}{s_d / \sqrt{n}}$$
where $d_s = M_A(s) - M_B(s)$. A result is designated statistically significant when $p < 0.05$.

---

## 6. Artifact Verification and Export

The `MatrixAggregator` exports:
1. `benchmark_matrix.md`: Formatted Markdown report with mean and std.
2. `benchmark_table.tex`: Publication-ready LaTeX `tabular` using `booktabs`.
3. `benchmark_runs.csv`: Full raw execution records.
Each generated artifact is hashed with SHA-256 and registered in the `ArtifactRegistry`.
