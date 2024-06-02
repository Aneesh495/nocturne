# ADR-006: Zero-Spike Denominator Contract in Bits-Per-Spike Metrics

## Status
Accepted

## Context
The standard metric for quantifying point-process neural model quality relative to a homogeneous baseline firing rate is Bits Per Spike (BPS):
$$\text{BPS} = \frac{\ln \mathcal{L}(\hat{\lambda}) - \ln \mathcal{L}(\lambda_0)}{N_{\text{spikes}} \cdot \ln 2}$$
where $N_{\text{spikes}} = \sum_{b, n} y_{b, n}$ is the total number of observed spikes in the evaluation window.
In quiescent behavioral trials, brief test windows, or sparse neural sub-populations (e.g., quiet baseline periods before reach onset), $N_{\text{spikes}}$ can evaluate to exactly zero ($N_{\text{spikes}} = 0$). In naive metric libraries, evaluating this expression triggers an IEEE 754 division by zero, yielding `nan` or `inf`. Ad-hoc solutions (such as adding an arbitrary epsilon $N_{\text{spikes}} + \epsilon$) corrupt numerical comparability across benchmarks and mask pathological quiet trials.

## Decision
1. **Strict Optional Return Contract**: In `BitsPerSpikeEvaluator` (`src/nocturne/evaluation/bits_per_spike.py`), when evaluated spike count $N_{\text{spikes}} = 0$, the function returns `None` for that neuron or population partition rather than a floating-point number.
2. **Explicit Zero Handling in Results**: The `BPSResult` dataclass declares `population_bps: Optional[float]` and `per_neuron_bps: List[Optional[float]]`.
3. **Downstream Accounting**: Aggregators, tables, and reporting modules format `None` values as `-` or `N/A`, and compute unweighted macro-averages exclusively over valid active channels with $N_{\text{spikes}} \ge 1$, recording the active neuron count alongside the metric.
4. **Negative Control Verification**: Negative Control NC04 (`src/nocturne/verification/controls.py`) executes a zero-spike synthetic trial and asserts that BPS returns `None` with zero warnings, zero NaNs, and zero exceptions.

## Consequences
- Total elimination of spurious `NaN` values and division-by-zero crashes in evaluation pipelines.
- Standardized, reproducible handling of quiescent trials and sparse neural units.
- Absolute numerical integrity when benchmarking across short temporal windows.
