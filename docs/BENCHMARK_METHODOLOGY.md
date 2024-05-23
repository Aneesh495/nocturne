# Nocturne Benchmark Methodology and Evaluation Protocol

## 1. Primary Point-Process Metrics

Evaluating neural population dynamics models on binned spike counts requires likelihood-based metrics that accurately reflect Poisson point-process statistics.

### 1.1 Bits-Per-Spike (BPS)

Bits-Per-Spike quantifies the information gain (in bits) that a candidate model provides about each observed spike over an uninformative baseline (typically the static marginal training mean rate):

$$\text{BPS} = \frac{\ln \mathcal{L}_{\text{model}} - \ln \mathcal{L}_{\text{null}}}{N_{\text{spikes}} \cdot \ln(2)}$$

where:
- $\ln \mathcal{L}_{\text{model}} = \sum_{i \in \Omega} [y_i \ln \mu_i - \mu_i]$
- $\ln \mathcal{L}_{\text{null}} = \sum_{i \in \Omega} [y_i \ln \mu_{\text{null}, i} - \mu_{\text{null}, i}]$
- $\Omega$ denotes the set of evaluated held-out bins and channels.
- $N_{\text{spikes}} = \sum_{i \in \Omega} y_i$ is the total number of spikes in the evaluation set.

#### Strict Zero-Spike Denominator Contract
If an evaluated neuron or sub-interval contains exactly zero spikes ($N_{\text{spikes}} = 0$), the ratio $\frac{0}{0}$ is mathematically undefined. Nocturne strictly returns `None` rather than substituting 0.0 or dividing by zero.

### 1.2 Poisson Deviance and Deviance Pseudo-R2

Poisson deviance measures the discrepancy between the saturated model (where predicted count equals observed count $\mu_i = y_i$) and the fitted model:

$$D(y, \mu) = 2 \sum_{i \in \Omega} \left[ y_i \ln\left(\frac{y_i}{\mu_i}\right) - (y_i - \mu_i) \right]$$

with the standard limit convention that $y_i \ln(y_i / \mu_i) \to 0$ as $y_i \to 0$.

The deviance pseudo-$R^2$ normalizes deviance against the null baseline:

$$R_{\text{dev}}^2 = 1 - \frac{D(y, \mu_{\text{model}})}{D(y, \mu_{\text{null}})}$$

---

## 2. Behavioral Decoding Probes

To test whether inferred latent representations capture behaviorally relevant motor dynamics:

1. **Model Freezing**: The generative neural dynamics model is frozen.
2. **Train-Only Regression**: Ridge regression decoders are fit strictly on training latents $\hat{Z}_{\text{train}}$ and training behavioral trajectories $V_{\text{train}}$ (e.g. hand velocity $v_x, v_y$):
   $$W^* = (\hat{Z}_{\text{train}}^T \hat{Z}_{\text{train}} + \lambda I)^{-1} \hat{Z}_{\text{train}}^T V_{\text{train}}$$
3. **Out-of-Sample Evaluation**: Decoded velocities $\hat{V}_{\text{test}} = \hat{Z}_{\text{test}} W^*$ are evaluated on held-out test trials, reporting kinematic coefficient of determination $R^2$ and Pearson $r$.

---

## 3. Cryptographic Evaluation Integrity

To guard against subtle evaluation leakage:
- Split auditors guarantee disjointness: $\text{Trials}_{\text{train}} \cap \text{Trials}_{\text{test}} = \emptyset$.
- Predictions and ground truth target count arrays are fingerprinted with SHA-256 hashes embedded in the final evaluation bundle.
