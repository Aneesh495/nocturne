# Nocturne Latent Geometry, Gauge Alignment, and Fixed Point Diagnostics

## 1. Latent Gauge Symmetry

In latent variable models of neural population activity, the underlying coordinates $z_t \in \mathbb{R}^D$ are defined only up to an unidentifiable gauge transformation (e.g. arbitrary rotation, reflection, or affine change of basis). Two models trained with different random seeds or architectures can learn identical computational dynamics while spanning rotated coordinate frames.

Direct Euclidean comparison of unaligned trajectories $\|Z_1 - Z_2\|$ yields large spurious discrepancies. Nocturne resolves this using gauge-aware alignment protocols.

---

## 2. Orthogonal Procrustes Alignment

Given two centered trajectory matrices $Z_1, Z_2 \in \mathbb{R}^{T \times D}$:

$$\min_{R \in \mathcal{O}(D), s > 0} \|s Z_1 R - Z_2\|_F^2 \quad \text{subject to } R^T R = I_D$$

### 2.1 Closed-Form SVD Solution

Form the cross-covariance matrix $M = Z_1^T Z_2 \in \mathbb{R}^{D \times D}$. Computing its Singular Value Decomposition:

$$M = U \Sigma V^T$$

The optimal orthogonal matrix is given by:

$$R = U V^T$$

When reflections are disallowed, if $\det(R) = -1$, the sign of the column corresponding to the smallest singular value in $V$ is flipped:

$$V_{:,-1} \gets -V_{:,-1}, \quad R = U V^T$$

The optimal isotropic scaling factor $s$ is:

$$s = \frac{\operatorname{tr}(Z_1 R Z_2^T)}{\|Z_1\|_F^2} = \frac{\sum_{i=1}^D \Sigma_{ii}}{\|Z_1\|_F^2}$$

### 2.2 Strict Train-Only Fitting Protocol

To prevent test data leakage, the transformation parameters $(R, s, \mu_1, \mu_2)$ are fitted strictly on the training partition. Out-of-sample evaluation trials are transformed using the frozen training transformation:

$$\hat{Z}_{\text{test}} = s (Z_{\text{test}} - \mu_1) R + \mu_2$$

---

## 3. Fixed Point and Slow Point Identification

For continuous-time autonomous latent dynamics $\dot{z} = f_\theta(z)$, computational mechanisms (e.g. working memory attractors, line attractors, decision boundaries) correspond to topological fixed points and slow manifolds.

### 3.1 Kinetic Energy Objective

True fixed points satisfy $f_\theta(z^*) = 0$. In deep neural network vector fields, exact roots are identified by minimizing the kinetic energy:

$$q(z) = \frac{1}{2} \|f_\theta(z)\|^2$$

Starting from candidate states sampled along empirical trajectories, gradient descent updates:

$$z^{(k+1)} = z^{(k)} - \eta \left( \left. \frac{\partial f_\theta}{\partial z} \right|_{z^{(k)}} \right)^T f_\theta(z^{(k)})$$

Converged points are categorized:
- **True Fixed Point**: $\|f_\theta(z^*)\| \le 10^{-3}$
- **Slow Point**: $\|f_\theta(z^*)\| \le 10^{-1}$

Points within clustering radius $\delta_{\text{cluster}}$ are consolidated into unique topological representatives.

---

## 4. Local Jacobian Linearization and Stability

At an identified fixed point $z^*$, the local dynamics are linearized:

$$\delta \dot{z} \approx J(z^*) \, \delta z, \quad J(z^*) = \left. \frac{\partial f_\theta(z)}{\partial z} \right|_{z = z^*}$$

The Jacobian $J \in \mathbb{R}^{D \times D}$ is computed exactly via PyTorch autograd.

### 4.1 Eigenspectrum Classification

Let $\{\lambda_i = \alpha_i + i \omega_i\}_{i=1}^D$ denote the eigenvalues of $J$:

1. **Stable Node**: $\alpha_i < 0 \; \forall i$ and $\omega_i = 0 \; \forall i$.
2. **Stable Focus (Spiral Attractor)**: $\alpha_i < 0 \; \forall i$ and $\exists \omega_j \ne 0$.
3. **Saddle Point**: $\exists \alpha_i > 0$ and $\exists \alpha_j < 0$.
4. **Unstable Source**: $\alpha_i > 0 \; \forall i$.
