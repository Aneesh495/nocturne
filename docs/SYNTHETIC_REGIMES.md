# Controlled Synthetic Neural Circuits and Ground-Truth Latent Regimes

This document details the mathematical formulations, stability characteristics, Jacobian eigenspectra, and emission mappings for the four controlled dynamical regimes implemented in Nocturne (`src/nocturne/synthetic/`).

---

## 1. Regime 1: Linear Damped Harmonic Oscillator

The linear damped oscillator evaluates whether latent dynamical models can recover rotation planes and continuous exponential decay rates under point-process observations.

### 1.1 Differential Equations

The continuous-time state vector $z(t) = \begin{pmatrix} z_1(t) \\ z_2(t) \end{pmatrix} \in \mathbb{R}^2$ evolves according to:

$$\frac{dz_1}{dt} = -\gamma z_1 + \omega z_2$$
$$\frac{dz_2}{dt} = -\omega z_1 - \gamma z_2$$

In matrix form:

$$\frac{dz}{dt} = A z, \quad A = \begin{pmatrix} -\gamma & \omega \\ -\omega & -\gamma \end{pmatrix}$$

where $\gamma \ge 0$ is the damping coefficient and $\omega > 0$ is the angular frequency.

### 1.2 Analytic Properties and Eigenspectrum

- **Eigenvalues**: The characteristic equation $\det(A - \lambda I) = (\lambda + \gamma)^2 + \omega^2 = 0$ yields complex conjugate eigenvalues:
  $$\lambda_{1,2} = -\gamma \pm i \omega$$
- **Transition Semigroup**: The matrix exponential $e^{At}$ decomposes into radial decay and planar rotation:
  $$e^{At} = e^{-\gamma t} \begin{pmatrix} \cos(\omega t) & \sin(\omega t) \\ -\sin(\omega t) & \cos(\omega t) \end{pmatrix}$$
- **Fixed Point**: The origin $z^* = (0, 0)$ is a stable spiral sink (stable focus) for $\gamma > 0$.
- **Constant Jacobian**: The Jacobian matrix $J(z) = \frac{\partial f}{\partial z} = A$ is uniform across the entire phase space.

---

## 2. Regime 2: Nonlinear Stuart-Landau Limit Cycle Oscillator

The Stuart-Landau oscillator tests model recovery of nonlinear limit cycle attractors, amplitude saturation, and non-trivial rotational flow.

### 2.1 Differential Equations

In Cartesian coordinates $z = (z_1, z_2)$:

$$\frac{dz_1}{dt} = \left(\mu - (z_1^2 + z_2^2)\right) z_1 - \left(\omega + b (z_1^2 + z_2^2)\right) z_2$$
$$\frac{dz_2}{dt} = \left(\omega + b (z_1^2 + z_2^2)\right) z_1 + \left(\mu - (z_1^2 + z_2^2)\right) z_2$$

In polar coordinates $(r, \theta)$ with $z_1 = r \cos \theta, z_2 = r \sin \theta$:

$$\frac{dr}{dt} = r (\mu - r^2)$$
$$\frac{d\theta}{dt} = \omega + b r^2$$

### 2.2 Phase Space Geometry

- **Supercritical Hopf Bifurcation**: Occurs at $\mu = 0$. For $\mu > 0$, the origin loses stability and sheds a circular limit cycle.
- **Limit Cycle Manifold**: An isolated closed orbit at radius:
  $$r^* = \sqrt{\mu}$$
  with constant angular velocity $\Omega^* = \omega + b \mu$.
- **Transverse Floquet Exponent**: Linearizing the radial equation around $r^*$:
  $$\left. \frac{d}{dr}[r(\mu - r^2)] \right|_{r = \sqrt{\mu}} = \mu - 3\mu = -2\mu$$
  Trajectories perturbed away from the limit cycle contract exponentially at rate $\lambda_r = -2\mu$.
- **Unstable Equilibrium**: Origin $(0, 0)$ with eigenvalues $\mu \pm i\omega$.

---

## 3. Regime 3: Bistable Double-Well Memory Circuit

This regime simulates working memory persistence, discrete attractor dynamics, and input-triggered state transitions.

### 3.1 Differential Equations

The state $z = (z_1, z_2) \in \mathbb{R}^2$ evolves in a double-well potential with linear damping on the orthogonal axis:

$$\frac{dz_1}{dt} = \alpha (z_1 - z_1^3) + u_1(t)$$
$$\frac{dz_2}{dt} = -\lambda_2 z_2 + u_2(t)$$

where $\alpha > 0$ governs the potential barrier height and $\lambda_2 > 0$ provides transverse contraction.

### 3.2 Fixed Points and Invariants

For zero input ($u = 0$):
1. **Memory State A**: $z^* = (+1.0, 0.0)$. Jacobian:
   $$J(z^*) = \begin{pmatrix} -2\alpha & 0 \\ 0 & -\lambda_2 \end{pmatrix}, \quad \text{Eigenvalues: } -2\alpha, -\lambda_2$$
   Both eigenvalues are strictly negative; State A is a stable node.
2. **Memory State B**: $z^* = (-1.0, 0.0)$. Identical stable node eigenvalues $-2\alpha, -\lambda_2$.
3. **Separatrix Saddle**: $z^* = (0.0, 0.0)$. Jacobian has eigenvalues $+\alpha$ and $-\lambda_2$. Trajectories are repelled along $z_1$ and attracted along $z_2$.
- **Switching Invariant**: A brief input pulse $u_1(t)$ exceeding the separatrix threshold drives transitions between memory states A and B.

---

## 4. Regime 4: Driven Continuous-Time Recurrent Circuit (CTRNN)

This regime investigates the interaction between autonomous recurrent flow and external driving inputs, specifically examining whether inferred drive over-explains population activity.

### 4.1 Differential Equations

For a $D$-dimensional latent vector $z(t) \in \mathbb{R}^D$ and external input $u(t) \in \mathbb{R}^M$:

$$\tau \frac{dz}{dt} = -z + W \tanh(z) + B u(t)$$

where $\tau > 0$ is the membrane/network time constant, $W \in \mathbb{R}^{D \times D}$ is the recurrent connectivity matrix, and $B \in \mathbb{R}^{D \times M}$ is the input projection matrix.

### 4.2 State-Dependent Jacobian

$$\frac{\partial}{\partial z_j} \left( \frac{dz_i}{dt} \right) = \frac{1}{\tau} \left( -\delta_{ij} + W_{ij} (1 - \tanh^2(z_j)) \right)$$

In matrix form:

$$J(z) = \frac{1}{\tau} \left( -I + W \operatorname{diag}\left(1 - \tanh^2(z)\right) \right)$$

As $z$ explores phase space, local Jacobian eigenvalues vary smoothly with state, enabling transitions between stable fixed points, oscillatory modes, and input-dominated drive regimes.
