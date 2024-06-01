# ADR-001: Temporal Discretization and Binning Contract

## Status
Accepted

## Context
Electrophysiological spike recordings arrive as continuous-time timestamp events $t_i \in \mathbb{R}^+$. Statistical models and neural network backbones in Nocturne (including Poisson factor analysis, continuous-time VLDM, and axial spatiotemporal transformers) require temporal discretization into discrete bin counts $y_{b, n} \in \mathbb{Z}_{\ge 0}$ over uniform intervals $\Delta t$.

Ambiguities in interval boundary conventions (half-open $[t, t+\Delta t)$ vs $(t, t+\Delta t]$ vs closed $[t, t+\Delta t]$) frequently introduce one-bin temporal misalignments across disparate analyses. Furthermore, behavioral kinematics (e.g., cursor positions, hand velocities) are sampled at irregular or different frequencies and must be aligned synchronously without future data leakage.

## Decision
1. **Half-Open Interval Contract**: Every bin interval is strictly defined as half-open $[t_{\text{start}}, t_{\text{end}})$ where $t_{\text{end}} = t_{\text{start}} + \Delta t$. A spike timestamp $t$ belongs to bin $b$ if and only if $t_{\text{start}} \le t < t_{\text{end}}$.
2. **Explicit Temporal Grid**: All discretization operations must utilize `TemporalGrid` and `BinConfig` defined in `src/nocturne/contracts/time.py`.
3. **Exact Spike Count Accumulation**: Spike counts are aggregated via `SpikeAccumulator` (`src/nocturne/binning/accumulator.py`) using binary search searchsorted routines with strict validation:
   $$\sum_{b=0}^{B-1} y_{b, n} \equiv N_{\text{spikes, window}}$$
   Any spike discarded outside the declared trial window is explicitly audited and accounted for.
4. **Behavioral Resampling**: Kinematic trajectories are resampled to match bin center timestamps:
   $$t_{\text{center}, b} = t_{\text{start}} + \frac{1}{2}\Delta t$$
   using causal linear interpolation strictly bounded within the recorded trial epoch.

## Consequences
- Guaranteed zero boundary overlap between consecutive temporal bins.
- Exact conservation of spike counts across all transformations.
- Elimination of off-by-one synchronization errors between behavioral kinematics and neural population firing rates.
