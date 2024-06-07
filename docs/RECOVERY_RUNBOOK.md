# Nocturne Fault Recovery and Operational Runbook

This runbook outlines operational procedures for diagnosing errors, recovering interrupted training workflows, handling telemetry faults during causal streaming, and rectifying verification gate failures.

---

## 1. Training Interruption and Checkpoint Recovery

### Scenario: Process Preemption or Out-Of-Memory Crash
When a training worker terminates prematurely, the checkpoint directory may contain partially written files or obsolete state.

### Recovery Workflow
1. **Locate Checkpoint Directory**: Navigate to the experiment run directory (e.g. `checkpoints/{experiment_id}/`).
2. **Scan for Temporary Files**: Remove any orphaned `.tmp_*` files left by incomplete POSIX atomic writes:
   ```bash
   rm -f checkpoints/*/.tmp_*
   ```
3. **Verify Candidate Checkpoints**: Run `CheckpointRecoveryManager` to validate candidate `.pt` snapshots:
   ```python
   from nocturne.training.recovery import CheckpointRecoveryManager
   recovery = CheckpointRecoveryManager("checkpoints/exp_vldm_01")
   valid_ckpt = recovery.find_latest_valid_checkpoint()
   ```
4. **Inspect Validation Diagnostics**: If a checkpoint contains `NaN` or `Inf` weights, `find_latest_valid_checkpoint()` automatically discards it and falls back to the preceding valid top-$K$ snapshot.
5. **Resume Training**: Pass `resume_from_checkpoint=valid_ckpt` to `NeuralDynamicsTrainer`. The engine automatically restores model weights, optimizer momentum buffers, learning rate scheduler state, and exact RNG states across Python, NumPy, and PyTorch.

---

## 2. Online Streaming Telemetry Faults

### Scenario: Network Packet Drops and Irregular Arrivals
During closed-loop neural streaming, client telemetry may drop packets, send irregular chunk sizes ($\Delta c \in [1, 32]$), or encounter network timeouts.

### Handling Procedures
- **Bounded Buffer Saturation**: If packets arrive faster than consumption, `StreamingBufferState` strictly caps its queue at $W_{\max} = 128$ bins, evicting the oldest bins in $\mathcal{O}(1)$ time to preserve constant memory.
- **Malformed Packet Interception**: If a chunk contains negative counts, non-finite floats, or channel dimension mismatches ($N \ne N_{\text{expected}}$), `StreamFaultHandler` catches the fault, logs an incident, and rolls back the buffer state to the last verified snapshot without process termination.
- **Gap Imputation**: When timestamp gaps occur ($t_{\text{next}} > t_{\text{curr}} + \Delta t$), the streamer imputes missing bins with baseline rates and flags the observation mask as invalid ($M_{\text{obs}} = \text{False}$), preventing downstream decoders from misinterpreting network dropouts as neural quiescence.

---

## 3. Verification Gate and Negative Control Failures

### Scenario: Acceptance Gate Failure during `nocturne verify`
When `SystemVerifier` reports a failure, follow the targeted remediation steps:

| Failing Check / Control | Root Cause | Remediation Procedure |
| :--- | :--- | :--- |
| **NC01 (Temporal Interval)** | Inverted or zero bin width in configuration | Ensure $t_{\text{end}} > t_{\text{start}}$ and $\Delta t > 0.0$ in `BinConfig`. |
| **NC02 (Split Overlap)** | Train/val/test intervals share timestamps | Re-run `SplitAuditor` to inspect trial boundaries and resolve split intersections. |
| **NC04 (Zero-Spike BPS)** | BPS function returned float instead of `None` | Verify `BitsPerSpikeEvaluator` adheres to the strict zero-spike contract. |
| **NC05 (Causal Leakage)** | Future tokens influencing earlier predictions | Check attention mask direction: ensure temporal MHA is set to `causal=True`. |
| **NC06 (Poisson Negatives)** | Firing rates $\le 0.0$ fed to Poisson loss | Verify rate decoder applies `softplus` activation with positive offset. |
| **NC08 (Buffer Overflow)** | Ring buffer unbounded accumulation | Verify `StreamingBufferState` capacity clamping $W \le W_{\max}$. |
| **NC10 (Checkpoint Corruption)** | Non-finite tensor accepted by loader | Ensure `CheckpointRecoveryManager` executes finite weight tensor audits. |

---

## 4. Disaster Recovery Checklist

Prior to deploying Nocturne models in closed-loop streaming or submitting benchmark runs:
1. Execute full verification suite: `python -m nocturne.cli verify --all`.
2. Generate fresh cryptographic acceptance bundle: `python -m nocturne.cli acceptance --output-dir acceptance_bundle/`.
3. Verify all SHA-256 signatures in `ACCEPTANCE_MANIFEST.json`.
