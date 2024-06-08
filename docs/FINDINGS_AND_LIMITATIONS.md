# Nocturne Scientific Findings and Architectural Limitations

This document synthesizes empirical observations, theoretical trade-offs, and fundamental limitations uncovered during the design and evaluation of Nocturne neural dynamics architectures.

---

## 1. Point-Process Likelihoods versus Continuous Approximations

### Empirical Finding
Models trained with the exact Poisson point-process objective consistently achieve superior bits-per-spike (BPS) and pseudo-$R^2$ compared to models trained with mean squared error (MSE) or Gaussian log-likelihood surrogates.

### Theoretical Analysis
In electrophysiological recordings with 20 ms bin discretization, spike counts per bin $y \in \{0, 1, 2, \dots\}$ are highly non-negative, discrete, and sparse ($P(y = 0) > 0.85$ for typical cortical channels). 
Continuous Gaussian and quadratic MSE loss formulations assume homoscedastic variance $\sigma^2$:
$$\mathcal{L}_{\text{MSE}} = \frac{1}{2} (y - \mu)^2$$
This formulation suffers from two major pathologies:
1. **Variance Underestimation on Bursts**: Biological neural firing exhibits Poisson-like count variability where variance scales proportionally with the mean ($\text{Var}(y) \approx \lambda \Delta t$). MSE penalizes large-rate bursts disproportionately while under-weighting deviations in quiet baseline regimes.
2. **Negative Rate Artifacts**: MSE loss allows models to produce negative rate predictions ($\lambda < 0$) in sparse channels unless artificial post-hoc clamping is applied, which destroys gradient flow.

The Poisson negative log-likelihood with softplus rate activation guarantees strictly positive rates $\lambda > 0$ and naturally scales precision inversely with expected variance, providing exact maximum likelihood calibration.

---

## 2. Autonomous Dynamics versus Inferred External Drive

### Empirical Finding
Autonomous continuous-time vector fields ($dz/dt = f_\theta(z)$) accurately reconstruct stereotyped delayed reach trajectories in MC_Maze, but exhibit severe dynamical collapse during unexpected limb perturbations in Area2_Bump.

### Mechanism of Perturbation Collapse
In unperturbed motor tasks, neural population activity traverses well-defined low-dimensional manifolds governed by internal recurrent dynamics (e.g. rotational dynamics in motor cortex). However, when a mechanical bump perturbation occurs, external somatosensory feedback injects abrupt, unpredictable energy into the circuit.
An autonomous model forced to explain this perturbation must either:
- Distort its autonomous vector field $f_\theta(z)$ with steep artificial gradients that corrupt baseline reach dynamics; or
- Smooth over the perturbation, causing substantial tracking delay and rate estimation error.

### Solution via Inferred External Drive
Introducing an inferred external drive input $u_t \in \mathbb{R}^{d_u}$:
$$\frac{dz}{dt} = f_\theta(z) + B u_t$$
where $u_t$ is inferred from recent sensory observations via an auxiliary encoder network, separates intrinsic autonomous flow from exogenous sensory shocks. This preserves autonomous attractor topology during reach planning while enabling rapid adaptation during unexpected perturbations.

---

## 3. Axial Spatiotemporal Attention versus Recurrent Latent ODEs

### Factorization Trade-offs
Axial spatiotemporal attention (STA) provides distinct operational advantages over continuous neural ODEs:
- **Parallel Training**: Attention computes over entire trial windows simultaneously without sequential numerical integration steps, accelerating training throughput on Apple Silicon and modern accelerators.
- **Flexible Masking**: The factorized architecture seamlessly switches between causal filtering (lower-triangular temporal mask) and offline bidirectional co-smoothing (full visibility) without retraining.

### Memory and Latency Scaling
Conversely, continuous-time latent ODEs (VLDM) retain key advantages in long-horizon deployment:
- **Constant Memory Footprint**: In streaming regimes, VLDM maintains a compact $d$-dimensional latent vector $z_t$, whereas attention requires a temporal history buffer $W$ that scales linearly with sequence context.
- **Continuous Interpolation**: Neural ODEs can evaluate latent states at arbitrary continuous timestamps $t \in \mathbb{R}$, whereas attention is locked to the discrete grid $\Delta t$.

---

## 4. Gauge Invariance in Neural Manifold Analysis

A crucial scientific finding is that raw Euclidean distance between latent states across different model initializations is meaningless due to coordinate gauge ambiguity:
$$z'_t = M z_t, \quad C' = C M^{-1}$$
By enforcing **Orthogonal Procrustes alignment** fitted strictly on training data, Nocturne establishes valid out-of-sample manifold comparisons. Furthermore, topological metrics:
- Fixed point locations and stability spectra (Jacobian eigenvalues $\text{Re}(\mu) \le 0$);
- Kinetic energy landscapes $q(z) = \frac{1}{2}\|f_\theta(z)\|^2$;
- Canonical Correlation Analysis (CCA) subspace overlaps;
provide gauge-invariant characterizations of learned population dynamics that transcend arbitrary coordinate choices.

---

## 5. Architectural Limitations and Open Directions

1. **Uniform Temporal Discretization**: Nocturne currently operates on uniform bins ($\Delta t \in [5, 50]$ ms). Sub-millisecond spike timing precision, phase-of-firing codes, and continuous-time point processes (e.g. Hawkes processes or continuous neural jump ODEs) remain avenues for future expansion.
2. **Ultra-Large Neural Ensembles**: While axial attention factorizes spatial complexity to $\mathcal{O}(B T N^2 d)$, scaling to $N > 10,000$ channels (e.g. whole-brain Neuropixels recordings) will require sparse graph attention or low-rank spatial factorizations.
3. **Multi-Day Session Non-Stationarity**: Current models assume stationary tuning within each recording. Chronic multi-day recordings exhibit electrode drift and representational turnover, requiring adaptive online recalibration.
