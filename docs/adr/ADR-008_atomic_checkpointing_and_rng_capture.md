# ADR-008: Atomic Checkpoint Serialization and Multi-Library RNG State Capture

## Status
Accepted

## Context
Long-running training campaigns and streaming model evaluations are vulnerable to system interrupts, hardware crashes, memory evictions, or node preemption. If a process terminates while writing a checkpoint directly to disk (e.g. `torch.save(state, "model.pt")`), the destination file is left truncated and corrupted, destroying previously valid state.
Furthermore, resuming training without exact random number generator (RNG) state preservation causes non-deterministic divergence in data augmentation, dropout masks, and stochastic gradient descent batches.

## Decision
1. **POSIX Atomic Write via Temporary File**: `AtomicCheckpointManager` (`src/nocturne/training/checkpoint.py`) writes all serialized payload bytes to a unique temporary file in the same directory (`.tmp_{uuid}`), forces OS buffer sync (`os.fsync`), and executes an atomic POSIX rename via `os.replace`. The target file path transitions instantaneously from old valid state to new valid state.
2. **Multi-Library RNG State Capture**: Every saved checkpoint payload captures the exact pseudo-random generator states across:
   - Python standard library (`random.getstate()`)
   - NumPy RNG (`np.random.get_state()`)
   - PyTorch CPU RNG (`torch.get_rng_state()`)
   - PyTorch MPS/CUDA RNG (`torch.mps` or `torch.cuda.get_rng_state_all()`)
3. **Corrupted Checkpoint Detection**: `CheckpointRecoveryManager` (`src/nocturne/training/recovery.py`) verifies the file header, checks unpickling safety, scans all weight matrices for non-finite values (NaN/Inf), and verifies internal SHA-256 checksums before loading parameters into live memory.
4. **Top-K Retention and Atomic Pruning**: Retains only the top-$K$ best checkpoints (by validation deviance) and atomic cleans obsolete snapshots to conserve disk space.

## Consequences
- Total immunity against truncated or corrupted checkpoint files from interrupted execution.
- Bit-exact training resumption from saved checkpoints.
- Automated validation of checkpoint integrity during cold starts and disaster recovery.
