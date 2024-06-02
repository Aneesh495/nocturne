# Nocturne System Architecture and Theoretical Blueprint

This document details the architectural design, mathematical foundations, dataflow pipelines, and state machines governing the Nocturne probabilistic neural dynamics laboratory.

---

## 1. End-to-End Laboratory Dataflow

The laboratory processes continuous electrophysiological timestamps and kinematics into discrete tensors, trains probabilistic models, executes causal streaming inference, and evaluates out-of-sample dynamical geometry without data leakage.

```mermaid
flowchart TD
    RawSpikes["Raw Spikes & Kinematics (NWB / DANDI / Synth)"] --> ContractValidation["Contracts & Split Auditing (contracts/ & recordings/)"]
    ContractValidation --> BinningAcc["SpikeAccumulator & TemporalGrid (binning/accumulator.py)"]
    BinningAcc --> BatchCollator["NeuralBatchCollator & Masks (training/collator.py)"]
    BatchCollator --> ModelFamily{"Model Architecture Selection"}
    ModelFamily --> MeanRate["MeanRateBaseline"]
    ModelFamily --> PLDS["PLDS / Laplace Smoother"]
    ModelFamily --> VLDM["Variational Latent Dynamics (VLDM)"]
    ModelFamily --> STA["Spatiotemporal Attention (STA)"]
    STA --> LikelihoodLoss["PoissonLoss & Likelihood Math (models/likelihoods.py)"]
    VLDM --> LikelihoodLoss
    PLDS --> LikelihoodLoss
    MeanRate --> LikelihoodLoss
    LikelihoodLoss --> EvalSuite["Evaluation & Metric Calculation (evaluation/)"]
    EvalSuite --> Geometry["Procrustes & Dynamical Geometry (geometry/)"]
    EvalSuite --> Streaming["Online Causal Streaming Engine (streaming/)"]
    Streaming --> Verification["System Verifier & Acceptance (verification/)"]
```

---

## 2. Point-Process Observation Math and Stirling Lookup

Poisson log-likelihood computation evaluates integer spike counts $y \in \mathbb{Z}_{\ge 0}$ against predicted firing rates $\lambda \in \mathbb{R}^+$. A precomputed double-precision lookup table (LUT) eliminates runtime transcendental gamma evaluations for count values $k \le 1024$.

```mermaid
flowchart LR
    SpikeCount["Integer Count y"] --> Branch{"Count Check: y <= 1024?"}
    Branch -->|"Yes (<= 1024)"| PrecomputedLUT["Cached Log-Factorial LUT: ln(y!) (O(1) Memory Index)"]
    Branch -->|"No (> 1024)"| RamanujanApprox["Ramanujan Refinement Formula (High-Order Analytic)"]
    PrecomputedLUT --> PoissonNLL["Poisson Negative Log-Likelihood: -y ln(lambda dt) + lambda dt + ln(y!)"]
    RamanujanApprox --> PoissonNLL
    PoissonNLL --> DevianceCalc["Poisson Deviance: 2 * [y ln(y / mu) - (y - mu)]"]
```

---

## 3. Axial Spatiotemporal Attention Factorization

Standard dense multi-head attention over space-time tokens costs $\mathcal{O}(B T^2 N^2 d)$. Nocturne factorizes attention into alternating temporal and spatial operations, reducing complexity to $\mathcal{O}(B N T^2 d + B T N^2 d)$.

```mermaid
flowchart TD
    InputToken["Input Spatiotemporal Tokens: Tensor (B, T, N, d)"] --> TemporalReshape["Reshape to (B*N, T, d): Group across Neurons"]
    TemporalReshape --> TemporalMHA["Temporal Multi-Head Attention (Causal or Bidirectional Mask)"]
    TemporalMHA --> TempNorm["Residual Add & LayerNorm"]
    TempNorm --> SpatialReshape["Reshape to (B*T, N, d): Group across Time"]
    SpatialReshape --> SpatialMHA["Spatial Multi-Head Attention (Channel Mask across Neurons)"]
    SpatialMHA --> SpatNorm["Residual Add & LayerNorm"]
    SpatNorm --> FeedForward["Pointwise Feedforward Network (Expansion & Dropout)"]
    FeedForward --> BlockOutput["Block Output: Tensor (B, T, N, d)"]
```

---

## 4. Continuous-Time Variational Latent Dynamics (VLDM)

VLDM encodes neural population history into an initial variational latent distribution $q(z_0 \mid x)$ and integrates forward continuously via a parameterized vector field $dz/dt = f_\theta(z) + B u_t$ using fourth-order Runge-Kutta (RK4) integration.

```mermaid
flowchart TD
    ObsHistory["Observed Spike Counts: (B, T, N)"] --> GRUEncoder["Bidirectional GRU Observation Encoder"]
    GRUEncoder --> LatentPosterior["Posterior Parameters: mu_0, logvar_0"]
    LatentPosterior --> Reparameterize["Reparameterization Trick: z_0 ~ N(mu_0, sigma_0^2)"]
    Reparameterize --> ODEIntegration["RK4 Integrator: dz/dt = f_theta(z) + B u_t"]
    ODEIntegration --> LatentTrajectory["Continuous Latents: z(t) over [0, T]"]
    LatentTrajectory --> RateDecoder["MLP Rate Decoder + Softplus Readout"]
    RateDecoder --> PredictedRates["Predicted Firing Rates: lambda(t) > 0"]
    LatentPosterior --> AnalyticKL["Analytic Gaussian KL Divergence: KL(q(z_0) || N(0, I))"]
    PredictedRates --> ELBO["Evidence Lower Bound (ELBO) Objective"]
    AnalyticKL --> ELBO
```

---

## 5. Causal Streaming Engine and Ring Buffer State Machine

The online streaming engine ingests incoming neural packet chunks under real-time deadlines, maintains bounded circular history, and restores valid states upon detecting malformed telemetry.

```mermaid
stateDiagram-v2
    [*] --> Uninitialized: Engine Allocation
    Uninitialized --> Ready: Initialize Buffer (Capacity W_max)
    Ready --> Streaming: Push First Spike Chunk
    Streaming --> Checkpointing: Save Buffer Snapshot
    Checkpointing --> Streaming: Snapshot Validated
    Streaming --> Streaming: Push Next Chunk (Evict Oldest Bin if > W_max)
    Streaming --> FaultDetected: Malformed Chunk or Gap Detected
    FaultDetected --> RollingBack: StreamFaultHandler Intercept
    RollingBack --> Ready: Restore Previous Snapshot & Impute Gaps
    Streaming --> Flushed: End of Stream Session
    Flushed --> [*]
```

---

## 6. Atomic Checkpointing and Fault Recovery Lifecycle

To prevent corrupted model checkpoints caused by premature process termination, Nocturne enforces atomic file writes via POSIX temporary file replacement and multi-RNG capture.

```mermaid
sequenceDiagram
    autonumber
    participant Trainer as Training Engine
    participant CheckpointMgr as AtomicCheckpointManager
    participant Disk as File System (POSIX)
    participant RecoveryMgr as CheckpointRecoveryManager

    Trainer->>CheckpointMgr: Request Save Checkpoint (epoch, state, RNG)
    CheckpointMgr->>CheckpointMgr: Capture RNG (Python, NumPy, PyTorch CPU & MPS)
    CheckpointMgr->>Disk: Write Payload to Temporary File (.tmp_uuid)
    CheckpointMgr->>Disk: Invoke os.fsync() (Flush Hardware Buffers)
    CheckpointMgr->>Disk: Execute os.replace(.tmp_uuid, model.pt) (Atomic Rename)
    Disk-->>Trainer: Checkpoint Write Confirmed
    Note over Trainer,Disk: System Restart or Crash Recovery Event
    Trainer->>RecoveryMgr: Request Checkpoint Load (model.pt)
    RecoveryMgr->>Disk: Read Header and Payload Bytes
    RecoveryMgr->>RecoveryMgr: Verify SHA-256 Checksum & Finite Weight Tensor Check
    RecoveryMgr->>Trainer: Restore Weights, Optimizer State, and Exact RNG
```

---

## 7. Inference Mode Routing and Causal Anti-Leakage Defense

The inference engine dynamically dispatches requests across Co-Smoothing, Causal Filtering, and Autoregressive Forecasting while actively auditing against future-data leakage.

```mermaid
flowchart TD
    InfReq["InferenceRequest (counts, mode, held_out_indices)"] --> ModeRouter{"Inference Mode Dispatcher"}
    ModeRouter -->|"CO_SMOOTHING"| CoSmooth["Bidirectional Spatial Co-Smoothing (Mask Held-Out Neurons)"]
    ModeRouter -->|"FILTERING"| FilterEngine["Causal Filtering (Strict Lower-Triangular Temporal Mask)"]
    ModeRouter -->|"FORECASTING"| Forecaster["Autoregressive Forecaster (Condition on History Prefix)"]
    FilterEngine --> LeakageAuditor["CausalLeakageAuditor (Inject Future Perturbations)"]
    Forecaster --> LeakageAuditor
    LeakageAuditor --> PerturbCheck{"Output at t changes when t' > t perturbed?"}
    PerturbCheck -->|"Yes (Violation)"| LeakageAlert["Raise Fatal CausalLeakageError"]
    PerturbCheck -->|"No (Passed)"| VerifiedResult["Emit Verified InferenceResult"]
    CoSmooth --> VerifiedResult
```

---

## 8. Orthogonal Procrustes Gauge Alignment and Dynamical Geometry

Because unconstrained latent dynamics possess coordinate gauge ambiguity, Nocturne aligns learned trajectories to target systems using orthogonal Procrustes SVD fitted strictly on training data.

```mermaid
flowchart LR
    TrainLatents["Train Latents X_train"] --> SVD["Compute SVD: X^T Y = U Sigma V^T"]
    TargetGroundTruth["Target Coordinates Y_train"] --> SVD
    SVD --> OrthogonalR["Optimal Rotation Matrix: R = U V^T"]
    OrthogonalR --> FrozenAligner["Frozen Aligner Operator (R, mu_X, mu_Y, scale)"]
    TestLatents["Held-Out Test Latents X_test"] --> FrozenAligner
    FrozenAligner --> AlignedTest["Aligned Latents: X_test * R"]
    AlignedTest --> FixedPointFinder["Fixed Point Finder: min ||f_theta(z)||^2"]
    FixedPointFinder --> JacobianAnalysis["Jacobian Autograd: Eigenspectra & Topology"]
```

---

## 9. Acceptance Bundle and Cryptographic Manifest Verification

The verification suite runs system controls, asserts numerical exactness, benchmarks datasets, and compiles a cryptographically signed manifest for reproducible scientific audits.

```mermaid
flowchart TD
    Controls["Negative Controls NC01 - NC12 (verification/controls.py)"] --> SystemVerifier["SystemVerifier (verifier.py)"]
    Datasets["Dataset Fixtures (MC_Maze, MC_RTT, Area2_Bump)"] --> SystemVerifier
    SyntheticCircuits["Synthetic Circuits (4 Controlled Regimes)"] --> SystemVerifier
    StreamingChecks["Causal Streaming Equivalence Checks"] --> SystemVerifier
    SystemVerifier --> BundleGen["AcceptanceBundleGenerator (acceptance.py)"]
    BundleGen --> EnvJSON["ENVIRONMENT.json (Host & GPU specs)"]
    BundleGen --> DataJSON["DATASETS.json (Recording records & counts)"]
    BundleGen --> EvalJSON["EVALUATION_FINGERPRINTS.json (Split hashes)"]
    BundleGen --> VerifJSON["SYSTEM_VERIFICATION.json (All check results)"]
    EnvJSON --> SHA256["Compute SHA-256 Hashes of All Artifacts"]
    DataJSON --> SHA256
    EvalJSON --> SHA256
    VerifJSON --> SHA256
    SHA256 --> Manifest["ACCEPTANCE_MANIFEST.json (Signed Scientific Record)"]
```

---

## 10. Multi-Seed Campaign and Matrix Aggregation Architecture

The campaign package automates multi-seed model evaluation, 5-family ablation matrices, and robustness sweeps, serializing results to LaTeX, Markdown, and CSV tables.

```mermaid
flowchart TD
    SeedController["Seed Controller (Seeds: 42, 43, 44)"] --> BenchmarkRunner["BenchmarkSuiteRunner (campaign/benchmark.py)"]
    SeedController --> AblationRunner["AblationRunner: 5 Families (campaign/ablations.py)"]
    SeedController --> RobustnessRunner["RobustnessCampaignRunner (campaign/robustness.py)"]
    BenchmarkRunner --> Aggregator["MatrixAggregator (campaign/matrix.py)"]
    AblationRunner --> Aggregator
    RobustnessRunner --> Aggregator
    Aggregator --> PairedStats["Paired Student t-Test Hypothesis Engine (scipy.stats)"]
    Aggregator --> MarkdownTables["benchmark_matrix.md (Markdown Reports)"]
    Aggregator --> LaTeXReports["benchmark_table.tex (Publication Booktabs)"]
    Aggregator --> CSVExports["benchmark_runs.csv (Raw Records)"]
    MarkdownTables --> Reg["ArtifactRegistry Registration (artifacts/schema.py)"]
    LaTeXReports --> Reg
    CSVExports --> Reg
```
