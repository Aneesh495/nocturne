# ADR-005: Gauge Invariance and Orthogonal Procrustes Alignment

## Status
Accepted

## Context
In latent dynamical models (such as PLDS, variational latent neural ODEs, and deep state-space systems), the internal coordinate system of the latent representation $z_t \in \mathbb{R}^d$ is fundamentally unidentifiable up to an invertible linear transformation:
$$z'_t = M z_t, \quad C' = C M^{-1}$$
Because two models trained with different random initializations or seeds will converge to arbitrarily rotated, reflected, or scaled coordinate frames, raw element-wise comparisons (e.g. Euclidean distances between latent trajectories $\hat{z}$ and true ground-truth $z^*$) are mathematically uninterpretable and fail to reflect true dynamical fidelity.

## Decision
1. **Orthogonal Procrustes Alignment**: Implement `OrthogonalProcrustesAligner` (`src/nocturne/geometry/alignment.py`). Given centered source latents $X \in \mathbb{R}^{T \times d}$ and target coordinates $Y \in \mathbb{R}^{T \times d}$, compute the optimal orthogonal rotation/reflection matrix $R \in \mathcal{O}(d)$ minimizing Frobenius distance $\|X R - Y\|_F^2$ via Singular Value Decomposition:
   $$X^T Y = U \Sigma V^T \implies R = U V^T$$
   with singular sign correction $\det(R)$ if pure rotation $SO(d)$ is strictly enforced.
2. **Train-Only Alignment Contract**: The alignment transformation $(R, \mu_X, \mu_Y, s)$ must be fitted strictly on the training partition. The fitted operator is then frozen and evaluated out-of-sample on held-out test trajectories.
3. **Gauge-Invariant Dynamics Metrics**: Downstream dynamical properties (such as autonomous Jacobian eigenspectra, fixed point topologies, and kinetic energy landscapes) are computed using gauge-aware coordinate-independent differential geometric operations (`src/nocturne/geometry/jacobians.py`).

## Consequences
- Enables rigorous quantitative benchmarking against known ground-truth synthetic circuits (such as Lorenz, Duffing, and limit-cycle attractors).
- Guarantees zero test-set leakage during latent space alignment.
- Eliminates spurious coordinate discrepancy while strictly preserving topological properties of the underlying neural manifold.
