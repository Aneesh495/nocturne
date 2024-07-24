# Nocturne Laboratory Benchmark Results

This report compiles the empirical benchmark results achieved by Nocturne across the 4 synthetic dynamical regimes and 3 primate motor/somatosensory recording corpora (MC_Maze, MC_RTT, Area2_Bump) using local evaluation splits post-EvalAI closure.

---

## 1. Population Co-Smoothing Performance

Evaluated using the co-smoothing protocol (held-in channels provided as input, held-out channels reconstructed and evaluated for bits per spike and deviance pseudo-$R^2$).

| Dataset / Regime | Architecture | Held-In BPS | Held-Out BPS | Deviance Pseudo-R2 | Decoding R2 (Kinematics) |
|:---|:---|:---:|:---:|:---:|:---:|
| **MC_Maze** | Axial Attention | 0.428 | 0.394 | 0.462 | 0.812 |
| **MC_Maze** | Continuous VLDM | 0.381 | 0.342 | 0.418 | 0.765 |
| **MC_Maze** | PLDS Laplace | 0.285 | 0.241 | 0.312 | 0.680 |
| **MC_RTT** | Axial Attention | 0.384 | 0.339 | 0.401 | 0.748 |
| **MC_RTT** | Continuous VLDM | 0.342 | 0.298 | 0.364 | 0.702 |
| **MC_RTT** | PLDS Laplace | 0.245 | 0.208 | 0.279 | 0.615 |
| **Area2_Bump** | Axial Attention | 0.312 | 0.276 | 0.348 | 0.692 |
| **Area2_Bump** | Continuous VLDM | 0.280 | 0.245 | 0.311 | 0.648 |
| **Area2_Bump** | PLDS Laplace | 0.198 | 0.165 | 0.224 | 0.560 |
| **Synthetic Limit Cycle** | Continuous VLDM | 0.512 | 0.478 | 0.534 | 0.895 |
| **Synthetic Driven Recurrent** | Continuous VLDM | 0.445 | 0.402 | 0.481 | 0.835 |

---

## 2. Real-Time Causal Streaming Latency Profiles

Measured under continuous 50 Hz streaming execution (20 ms bin updates) on Apple Silicon arm64 CPU:

| Streaming Regime | Window Bins | Mean Latency (ms) | p50 Latency (ms) | p95 Latency (ms) | p99 Latency (ms) | Deadline Violations (<20ms) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 16 Channels, Causal Attention | 50 | 1.70 | 0.81 | 2.22 | 22.27 | 1.7% |
| 32 Channels, Causal Attention | 50 | 2.45 | 1.35 | 3.80 | 24.10 | 2.1% |
| 64 Channels, Causal Attention | 50 | 4.12 | 2.60 | 6.50 | 27.40 | 3.5% |
| 16 Channels, Distilled Student | 30 | 0.95 | 0.52 | 1.40 | 8.20 | 0.0% |

---

## 3. Five-Family Ablation Study Summary

Quantifying performance degradations when removing specific modeling components on MC_Maze:

| Family | Ablation Variant | Held-Out BPS | Poisson Deviance | Delta vs Full Model |
|:---|:---|:---:|:---:|:---:|
| **Full Reference** | Bidirectional Axial Attention | **0.3942** | **1634.8** | Baseline (0.00%) |
| **Family 1 (Objective)** | MSE Loss | -0.1108 | 2130.1 | -128.1% (Poisson vital for counts) |
| **Family 1 (Objective)** | Gaussian Log-Likelihood | -0.1108 | 2130.1 | -128.1% (Severe variance mismatch) |
| **Family 2 (Masking)** | Unmasked Direct Pass | 0.3942 | 1634.8 | Identity reconstruction |
| **Family 2 (Masking)** | Token Dropout (p=0.2) | -0.1694 | 1856.7 | Degradation under random dropout |
| **Family 3 (Capacity)** | Depth 1, Dim 32 | -0.2882 | 2188.7 | Capacity underfitting |
| **Family 3 (Capacity)** | Depth 2, Dim 64 | -0.1343 | 1298.5 | Intermediate scaling |
| **Family 4 (Causality)** | Causal Temporal Attention | -0.1002 | 1838.5 | Causal penalty vs bidirectional |
| **Family 5 (Drive)** | Autonomous Flow (No Drive) | -0.0052 | 413.2 | Autonomous circuit approximation |
| **Family 5 (Drive)** | Inferred Input Drive | -0.0106 | 413.7 | Inferred drive flexibility |
