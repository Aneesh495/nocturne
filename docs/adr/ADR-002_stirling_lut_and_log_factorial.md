# ADR-002: Exact Stirling Lookup Table and Ramanujan Log-Factorial Approximation

## Status
Accepted

## Context
Poisson log-likelihood computation requires evaluating the log-factorial of discrete integer spike counts:
$$\ln \mathcal{L}(y \mid \lambda) = y \ln(\lambda \Delta t) - \lambda \Delta t - \ln(y!)$$
In high-throughput training loops and evaluation sweeps across millions of neuron-bin observations, repeatedly calling `torch.lgamma(y + 1)` or `math.lgamma(y + 1)` introduces measurable GPU kernel launch overhead and CPU arithmetic latency. Furthermore, standard Stirling approximations without higher-order corrections deviate from exact integer factorials, distorting deviance and bits-per-spike metrics on low-count regimes ($y \in \{1, 2, 3\}$).

## Decision
1. **Precomputed Exact LUT**: A precomputed double-precision lookup table of exact values $\ln(k!)$ for $k \in \{0, 1, \dots, 1024\}$ is initialized once at startup (`src/nocturne/binning/likelihood_math.py`). Because single-bin spike counts for 20 ms bins in neocortical recordings virtually never exceed 100 spikes, over 99.999% of queries hit the exact $O(1)$ lookup table.
2. **Ramanujan Refinement for $k > 1024$**: For extremal counts exceeding 1024 (e.g. synthetic burst controls), Nocturne uses Ramanujan's high-order refinement:
   $$\ln(k!) \approx k \ln k - k + \frac{1}{6}\ln\left(k(1 + 4k(1 + 2k))\right) + \frac{1}{2}\ln \pi$$
   providing relative error below $10^{-12}$ across the entire domain.
3. **Dual CPU and GPU Support**: The table is instantiated as a cached PyTorch 1D tensor buffer (`torch.Tensor`) and NumPy array (`np.ndarray`), enabling zero-allocation vectorized indexing on both Apple Silicon MPS and host CPU.

## Consequences
- Evaluation of Poisson deviance is acceleration-ready without invoking expensive transcendental functions in inner loops.
- Exact arithmetic agreement between PyTorch tensor losses and NumPy evaluation metrics up to $10^{-10}$ tolerance.
- Zero floating-point division by zero when $y = 0$, since $\ln(0!) = 0.0$.
