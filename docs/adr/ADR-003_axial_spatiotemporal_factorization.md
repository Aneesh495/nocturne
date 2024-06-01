# ADR-003: Axial Spatiotemporal Attention Factorization

## Status
Accepted

## Context
Neural population recordings present data along two distinct orthogonal axes: time $T$ (temporal steps) and neurons $N$ (spatial electrode channels). Standard full dense multi-head self-attention treats the population sequence as $B$ flattened matrices of size $(T \times N) \times d$, yielding attention maps of size $(T \cdot N) \times (T \cdot N)$.
The computational complexity and memory footprint of standard attention scale as:
$$\mathcal{O}\left(B \cdot (T \cdot N)^2 \cdot d\right) = \mathcal{O}\left(B \cdot T^2 \cdot N^2 \cdot d\right)$$
For standard neural benchmarks with $T = 100$ bins and $N = 182$ neurons (MC_Maze), $T \cdot N = 18,200$ tokens per trial. A dense attention matrix requires $(18,200)^2 \approx 3.31 \times 10^8$ floating-point elements per head, which immediately exhausts VRAM on workstations and consumer GPUs.

## Decision
1. **Factorized Axial Attention Decomposition**: Decompose the representation into alternating axial attention blocks (`src/nocturne/models/attention/axial_block.py`):
   - **Temporal Attention**: Reshapes tensor to $(B \cdot N, T, d)$. Computes attention strictly across time steps for each neuron independently.
   - **Spatial Attention**: Reshapes tensor to $(B \cdot T, N, d)$. Computes attention strictly across neuron channels for each time step independently.
2. **Computational Complexity**:
   $$\text{Complexity} = \mathcal{O}\left(B \cdot N \cdot T^2 \cdot d + B \cdot T \cdot N^2 \cdot d\right)$$
   For $T = 100$ and $N = 182$, this reduces computation by a factor of:
   $$\frac{T \cdot N}{T + N} = \frac{18,200}{282} \approx 64.5\times$$
   rendering high-channel transformer modeling computationally tractable and memory-efficient.
3. **Causal Masking Isolation**: Strict causal masking is applied exclusively within the temporal attention mechanism (lower-triangular masking), while spatial attention remains bidirectional across channels at each causal time slice, preserving instant population co-dependency without future temporal leakage.

## Consequences
- Enables training and inference of deep multi-layer transformer architectures on multi-electrode arrays with $N > 100$ channels.
- Factorization separates temporal dynamical modeling (filtering, oscillation, delay) from instantaneous population co-dependency (low-rank manifold coordination).
- Linear memory scaling enables full trial sequence processing without batch-size truncation.
