# Nocturne Causal Leakage Rules and Verification Protocol

## 1. Information Flow Constraints

In computational neuroscience and neural interface engineering, models deployed for online decoding and closed-loop stimulation must operate causally. A model evaluated in an online setting must never condition on future time steps.

### 1.1 Mathematical Definition of Causal Prefix Invariance

Let $Y = [y_1, y_2, \dots, y_T] \in \mathbb{R}^{T \times N}$ denote an observed neural sequence. Let $f_{\theta}: \mathbb{R}^{T \times N} \to \mathbb{R}^{T \times N}$ denote the causal neural dynamics model producing estimated firing rates $\hat{\Lambda} = [\hat{\lambda}_1, \dots, \hat{\lambda}_T]$ and latent states $\hat{Z} = [\hat{z}_1, \dots, \hat{z}_T]$.

A model $f_\theta$ satisfies **Causal Prefix Invariance** if and only if for all sequence lengths $T$, all cutoff indices $t^* \in \{1, \dots, T-1\}$, and all arbitrary future perturbations $\Delta Y$ restricted to $\{t > t^*\}$:

$$\hat{\lambda}_t(Y + \Delta Y) = \hat{\lambda}_t(Y) \quad \forall t \le t^*$$
$$\hat{z}_t(Y + \Delta Y) = \hat{z}_t(Y) \quad \forall t \le t^*$$

Equivalently, the functional Jacobian between past outputs and future inputs is strictly zero:

$$\frac{\partial \hat{\lambda}_t}{\partial y_{t'}} = 0, \quad \frac{\partial \hat{z}_t}{\partial y_{t'}} = 0 \quad \forall t' > t$$

---

## 2. Common Structural Causes of Causal Leakage

1. **Bidirectional Recurrent Encoders**: Running bidirectional GRUs or LSTMs over the full sequence duration couples time step $t$ to future states $h_{t+1 \dots T}^{(bwd)}$.
2. **Unmasked Multi-Head Attention**: Standard self-attention allows query token at time $t$ to compute dot products with key tokens at time $t' > t$.
3. **Improper Normalization along Sequence Axis**: Batch Normalization or standard 1D Instance Normalization across the temporal dimension $T$ computes empirical mean $\mu_T = \frac{1}{T} \sum_{t=1}^T y_t$ and variance $\sigma_T^2$, leaking summary statistics of future bins into past bins. Nocturne avoids this by using LayerNorm strictly across the channel/embedding dimension $d_{\text{model}}$ per time step.
4. **Non-Causal Convolutions**: Standard 1D convolutions center the filter kernel around time step $t$, looking ahead by $\lfloor K/2 \rfloor$ bins.

---

## 3. Perturbation Auditing Protocol

To guarantee acceptance gate BA07, `CausalLeakageAuditor` subjects the model to an automated adversarial perturbation engine:

1. **Sample Diversity**: Evaluates $\ge 1,000$ independent trials across varying sequence lengths, channel counts, and cutoff boundaries $t^* \sim \text{Uniform}(1, T-2)$.
2. **Adversarial Future Inputs**:
   - High-amplitude spike bursts: $y_{t' > t^*} \gets y_{t'} + 20.0$.
   - Sudden quiescent clamping: $y_{t' > t^*} \gets 0.0$.
   - High-variance uniform noise: $y_{t' > t^*} \sim \text{DiscreteUniform}(0, 25)$.
   - Alternating high-frequency pulse trains.
3. **Verification Tolerance**:
   $$\max_{t \le t^*} |\hat{\lambda}_t^{\text{orig}} - \hat{\lambda}_t^{\text{pert}}| \le 10^{-5}$$
   Any deviation exceeding this tolerance immediately fails the causal audit and generates a diagnostic report with the violating time index and tensor diff.
