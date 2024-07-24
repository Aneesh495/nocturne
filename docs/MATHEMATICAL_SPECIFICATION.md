# Nocturne Mathematical Specification

This document provides the formal mathematical specification for the generative models, continuous dynamical systems, geometric alignments, and information-theoretic evaluation metrics in Nocturne.

---

## 1. Generative Neural Point-Process Observation Model

Let $y_{t, n} \in \mathbb{N}_0$ denote the integer spike count observed on electrode channel $n \in \{1, \dots, N\}$ in discrete temporal bin $t \in \{1, \dots, T\}$ with bin width $\Delta t > 0$.

Observations follow an conditionally independent inhomogeneous Poisson process:
$$y_{t, n} \mid \lambda_{t, n} \sim \text{Poisson}(\lambda_{t, n} \Delta t)$$

with log-likelihood given by:
$$\ln p(y_{t, n} \mid \lambda_{t, n}) = y_{t, n} \ln(\lambda_{t, n} \Delta t) - \lambda_{t, n} \Delta t - \ln(y_{t, n}!)$$

The instantaneous conditional firing rate $\lambda_{t, n} > 0$ is parameterized via a smooth, strictly positive Softplus nonlinearity:
$$\lambda_{t, n} = \frac{1}{\beta} \ln(1 + \exp(\beta \cdot \eta_{t, n}))$$
where $\eta_{t, n} = (W z_t + b)_n \in \mathbb{R}$ is the linear decoding projection from latent state $z_t \in \mathbb{R}^D$ with readout matrix $W \in \mathbb{R}^{N \times D}$ and baseline bias $b \in \mathbb{R}^N$.

---

## 2. Continuous Latent Dynamics & Symplectic Integrators

### 2.1 Continuous Neural ODE
Latent states evolve continuously according to a parameter-conditioned vector field:
$$\frac{dz(t)}{dt} = f_\theta(z(t), u(t))$$
where $u(t) \in \mathbb{R}^K$ represents external or inferred driving inputs.

State rollouts from initial state $z(0)$ to time $T$ are computed via numerical integration:
$$z(t) = z(0) + \int_0^t f_\theta(z(\tau), u(\tau)) d\tau$$

### 2.2 Symplectic Störmer-Verlet Integration
For Hamiltonian mechanical circuits with separable Hamiltonian $H(q, p) = \frac{1}{2} p^T M^{-1} p + V(q)$, Nocturne implements symplectic Störmer-Verlet integration:
$$p_{t + 1/2} = p_t - \frac{\Delta t}{2} \nabla V(q_t)$$
$$q_{t + 1} = q_t + \Delta t M^{-1} p_{t + 1/2}$$
$$p_{t + 1} = p_{t + 1/2} - \frac{\Delta t}{2} \nabla V(q_{t + 1})$$

Symplectic integration preserves the canonical symplectic 2-form $dq \wedge dp$ and bounds energy drift $\|H(q_T, p_T) - H(q_0, p_0)\| \le \mathcal{O}(\Delta t^2)$ over arbitrarily long trajectories.

---

## 3. Spatiotemporal Rotary Positional Embeddings (RoPE)

Temporal attention blocks encode relative time differences between token bins via orthogonal rotation operators $R_{\Theta, m} \in SO(d_{\text{head}})$ acting on query and key vectors:

$$R_{\Theta, m} = \text{diag}\left( R(\theta_1 m), R(\theta_2 m), \dots, R(\theta_{d/2} m) \right)$$
where each 2D planar rotation block is given by:
$$R(\theta_k m) = \begin{pmatrix} \cos(m \theta_k) & -\sin(m \theta_k) \\ \sin(m \theta_k) & \cos(m \theta_k) \end{pmatrix}$$
with frequencies $\theta_k = 10000^{-2(k - 1)/d}$.

Under this transformation, attention logits depend strictly on the relative lag $(m - n)$:
$$\langle R_{\Theta, m} q, R_{\Theta, n} k \rangle = q^T R_{\Theta, m - n} k = g(q, k, m - n)$$
guaranteeing temporal shift invariance across arbitrary sequence lengths.

---

## 4. Orthogonal Procrustes Gauge Alignment

Let $Z_1, Z_2 \in \mathbb{R}^{T \times D}$ denote two latent trajectory embeddings up to unobservable linear gauge transformations. Nocturne computes the optimal orthogonal alignment:
$$\min_{R \in O(D), s > 0, t \in \mathbb{R}^D} \| s (Z_1 - \mu_1) R + t - Z_2 \|_F^2$$
subject to $R^T R = I_D$.

Centering $\tilde{Z}_1 = Z_1 - \mu_1$ and $\tilde{Z}_2 = Z_2 - \mu_2$, the singular value decomposition of the cross-covariance matrix is computed:
$$M = \tilde{Z}_1^T \tilde{Z}_2 = U \Sigma V^T$$
The optimal rotation matrix is $R^* = U V^T$ (with reflection correction if $\det(R^*) < 0$), and optimal isotropic scaling is:
$$s^* = \frac{\text{tr}(\Sigma)}{\|\tilde{Z}_1\|_F^2}$$

The normalized Procrustes disparity residual is defined as:
$$\delta(Z_1, Z_2) = \frac{\| s^* \tilde{Z}_1 R^* + \mu_2 - Z_2 \|_F}{\| Z_2 \|_F}$$

---

## 5. Information-Theoretic Evaluation Metrics

### 5.1 Bits Per Spike (BPS)
Measures predictive improvement over a constant mean-rate homogeneous Poisson null model per emitted action potential:
$$\text{BPS} = \frac{\mathcal{L}_{\text{model}} - \mathcal{L}_{\text{null}}}{(\sum_{t, n} y_{t, n}) \cdot \ln 2}$$
where $\mathcal{L}_{\text{model}} = \sum_{t, n} [y_{t, n} \ln(\hat{\lambda}_{t, n} \Delta t) - \hat{\lambda}_{t, n} \Delta t]$ and $\mathcal{L}_{\text{null}} = \sum_{t, n} [y_{t, n} \ln(\bar{\lambda}_n \Delta t) - \bar{\lambda}_n \Delta t]$.

### 5.2 Poisson Deviance
Quantifies goodness-of-fit relative to a saturated model:
$$D(y, \hat{y}) = 2 \sum_{t, n} \left[ y_{t, n} \ln\left(\frac{y_{t, n}}{\hat{y}_{t, n}}\right) - (y_{t, n} - \hat{y}_{t, n}) \right]$$
with $y \ln(y / \hat{y}) \to 0$ when $y = 0$.

### 5.3 Generalized Rate I-Divergence (Distillation)
For knowledge distillation between teacher rates $\lambda^T$ and student rates $\lambda^S$:
$$D_I(\lambda^T \parallel \lambda^S) = \sum_{t, n} \left[ \lambda_{t, n}^T \ln\left(\frac{\lambda_{t, n}^T}{\lambda_{t, n}^S}\right) - (\lambda_{t, n}^T - \lambda_{t, n}^S) \right]$$
which corresponds to the continuous Bregman divergence generated by the negative Poisson entropy.
