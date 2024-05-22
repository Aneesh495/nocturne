# Nocturne Training and Recovery Runbook

## 1. Optimization Architecture

Nocturne implements a modular optimization engine designed for point-process models of neural population recordings.

### 1.1 Decoupled Parameter Grouping

Standard weight decay improperly shrinks biases and normalization parameters (such as LayerNorm scale and bias), shifting zero-firing rate offsets and distorting activation variance. Nocturne segments parameters into two disjoint groups:

1. **Decayed Parameters**: Weight matrices of 2D or higher dimensionality ($W_{\text{in}}$, projection layers, linear dynamics matrices, attention query/key/value projections).
2. **Non-Decayed Parameters**: 1D biases and normalization gains where weight decay is strictly set to 0.0.

### 1.2 Learning Rate Scheduling

Training utilizes a two-phase schedule:

1. **Linear Warmup**: Over $S_{\text{warmup}}$ steps, the learning rate scales linearly from $\eta_0 = \eta_{\max} / S_{\text{warmup}}$ to $\eta_{\max}$. This prevents unstable updates when randomly initialized latent embeddings produce erratic Poisson gradients.
2. **Cosine Annealing**: Decays smoothly from $\eta_{\max}$ to $\eta_{\min} = \eta_{\max} \cdot \alpha_{\min}$ over the remaining steps:

$$\eta_s = \eta_{\min} + \frac{1}{2} (\eta_{\max} - \eta_{\min}) \left( 1 + \cos\left( \pi \frac{s - S_{\text{warmup}}}{S_{\text{total}} - S_{\text{warmup}}} \right) \right)$$

### 1.3 Gradient Norm Clipping

Spike count bursts can generate large negative log-likelihood gradients. All parameter updates are stabilized via global L2 gradient clipping:

$$\tilde{g} = \begin{cases} g & \text{if } \|g\|_2 \le \gamma_{\max} \\ \frac{\gamma_{\max}}{\|g\|_2} g & \text{otherwise} \end{cases}$$

Default threshold is $\gamma_{\max} = 1.0$.

---

## 2. Atomic Checkpointing and State Recovery

### 2.1 Atomic Replacement Guarantee

Process interruptions (such as node preemption or hardware faults) during standard checkpoint serialization produce truncated or corrupt binary files. Nocturne enforces atomic file replacement using temporary staging files and atomic filesystem replacement:

1. State dictionary is serialized to a PID-tagged temporary file: `tmp_checkpoint_<pid>_<step>.pt`.
2. Upon complete, verified disk flush, `os.replace` atomically substitutes the target path: `checkpoint_epoch_XXXX_step_YYYYYY.pt`.
3. An identical protocol maintains pointers to `latest.pt` and `best.pt`.

At no point does a half-written file exist at the destination path.

### 2.2 Complete State Serialization

Every checkpoint records full computational context:
- `epoch`: Current training epoch index.
- `step`: Global step counter.
- `model_state_dict`: Model parameters.
- `optimizer_state_dict`: Optimizer momentum and variance buffers.
- `scheduler_state_dict`: Learning rate schedule step.
- `metrics`: Evaluation dictionary.
- `rng_states`: Snapshot of Python `random`, NumPy `np.random`, and PyTorch CPU/CUDA/MPS RNG generators.

### 2.3 Resumption Verification

`TrainingRecovery.verify_checkpoint_integrity` inspects:
1. Presence of all mandatory dictionary keys.
2. Absence of NaN or Infinite values across all model tensor buffers.
3. Restoration of identical pseudo-random number generator states, guaranteeing reproducible subsequent trial sampling.
