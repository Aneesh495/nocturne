# Data Contracts, Units, and Discretization Specifications

This specification defines the foundational data contracts, unit definitions, observation masks, and split auditing policies governing all electrophysiology data in Nocturne.

## 1. Units and Mathematical Conventions

Electrophysiological observations are recorded either as point processes (discrete action potential timestamps) or binned event counts. Nocturne enforces strict non-negative integer count representations for spike counts and strictly positive real numbers for firing rates.

### 1.1 Fundamental Physical Units

- **Time ($t$)**: Seconds ($s$). Millisecond conversions ($ms$) are derived via $t_{ms} = 1000 \times t_s$.
- **Spike Counts ($y_{t,n}$)**: Non-negative integers $\mathbb{N}_0 = \{0, 1, 2, \dots\}$. Floating-point count representations are strictly forbidden in count likelihood modules.
- **Instantaneous Firing Rates ($\lambda_{t,n}$)**: Hertz ($Hz$, spikes per second), $\lambda \in [0, \infty)$.
- **Expected Binned Poisson Rate ($\mu_{t,n}$)**: Expected spikes per bin, computed as:
  $$\mu_{t,n} = \lambda_{t,n} \times \Delta t$$
  where $\Delta t$ is the bin width in seconds.

### 1.2 Point Process Likelihood Accounting

Under an inhomogeneous Poisson process assumption with rate $\lambda_n(t)$, the probability of observing $y_{t,n}$ spikes in a bin of duration $\Delta t$ centered at $t$ is:

$$P(Y_{t,n} = y_{t,n} \mid \mu_{t,n}) = \frac{\mu_{t,n}^{y_{t,n}} e^{-\mu_{t,n}}}{y_{t,n}!}$$

The log-likelihood for neuron $n$ at bin $t$ is:

$$\ln P(Y_{t,n} = y_{t,n} \mid \mu_{t,n}) = y_{t,n} \ln \mu_{t,n} - \mu_{t,n} - \ln(y_{t,n}!)$$

**Unit Transformation Rules**:
1. When a model predicts firing rate in spikes/second $\lambda_{t,n}$, it must be scaled by bin duration $\Delta t$ before computing the Poisson loss.
2. In log-rate parameterization where model predicts $\eta_{t,n} = \ln \lambda_{t,n}$, the log-mean per bin is:
   $$\ln \mu_{t,n} = \ln \lambda_{t,n} + \ln \Delta t = \eta_{t,n} + \ln \Delta t$$
3. The factorial term $\ln(y!)$ is required when computing exact held-out log-likelihood, model evidence, or comparing models across disparate count transforms.

## 2. Core Contract Schemas

### 2.1 Temporal Contracts (`src/nocturne/contracts/time.py`)

- `TimeInterval`: Continuous half-open interval $[start, stop)$ in seconds with finite boundary validation.
- `BinConfig`: Immutable binning configuration defining bin width $\Delta t$, alignment to origin ($t=0$), and rate-to-count scaling operators.
- `TemporalGrid`: Uniform grid of bin edges and bin centers derived from a continuous interval. Accounts for floating-point epsilon rounding to ensure bin count determinism.

### 2.2 Neuron and Population Metadata (`src/nocturne/contracts/neurons.py`)

- `NeuronMetadata`: Attributes of an individual recorded neural unit, including unique session identifier `neuron_id`, zero-indexed array channel `channel_index`, anatomical brain region `brain_area`, isolation quality `quality` (`SUA`, `MUA`, `UNSORTED`), probe coordinates in micrometers, and benchmark role `role`.
- `PopulationMetadata`: Immutable population collection with mandatory uniqueness validation for `neuron_id` and `channel_index`. Provides accessors for held-in versus held-out channels.

### 2.3 Observation and Evaluation Masks (`src/nocturne/contracts/masks.py`)

Nocturne separates model inputs from evaluation targets using explicit multi-dimensional boolean masks:

- `observed_mask` ($[B, T, N]$): True for channels and time bins provided as inputs to the model.
- `target_mask` ($[B, T, N]$): True for channels and time bins evaluated in the loss or benchmark metrics.
- `padding_mask` ($[B, T]$): True for genuine recording bins, False for zero-padded tail bins in variable-length batches.
- `channel_mask` ($[B, N]$): True for valid active channels, False for disabled or missing channels.

**Information Boundaries**:
- In **Offline Co-Smoothing**, held-in channels are observed across all valid time bins ($observed = True$), while held-out channels are strictly unobserved ($observed = False$) and evaluated ($target = True$).
- In **Causal Filtering**, observation masks only expose information up to the current time bin $t$, preventing future observations from leaking into historical state estimates.
- In **Forecasting**, observations are restricted to prefix bins $t < T_{obs}$, and targets are evaluated over future horizon bins $t \in [T_{obs}, T_{obs} + H)$.

### 2.4 Split Integrity and Zero-Leakage Guarantees (`src/nocturne/contracts/splits.py`)

- `DatasetSplit`: Immutable partition mapping each trial or continuous interval to `train`, `val`, or `test`.
- Enforces strict pairwise disjointness:
  $$Train \cap Val = \emptyset, \quad Train \cap Test = \emptyset, \quad Val \cap Test = \emptyset$$
- Prohibits interval overlap for continuous recording splits.
- Test evaluation data is strictly isolated during training and hyperparameter selection.
