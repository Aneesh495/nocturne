# Nocturne Reproducibility Standards and Verification Guide

This document establishes the reproducibility protocols, deterministic execution guarantees, and audit procedures implemented across the Nocturne neural dynamics codebase.

---

## 1. Pinned Hardware and Software Environment

Nocturne is developed and verified against the following target environment:
- **Operating System**: macOS arm64 (Darwin Kernel, Apple Silicon)
- **Python Version**: 3.12.14 (managed via `uv`)
- **PyTorch**: 2.14.1 with Metal Performance Shaders (MPS) acceleration and CPU fallback
- **NumPy**: 2.5.4
- **SciPy**: 1.18.1
- **nlb-tools**: 0.0.4

All environment dependencies are declared in `pyproject.toml` and pinned in local virtual environments.

---

## 2. Multi-Library Determinism and RNG Management

To guarantee bit-exact numerical reproducibility across randomized runs, Nocturne implements centralized multi-library seed coordination:

### Seed Initialization Protocol
Whenever a seed $S$ is provided (standard suite: $S \in \{42, 43, 44\}$), the runner executes:
```python
import random
import numpy as np
import torch

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    torch.mps.manual_seed(seed)
```

### Checkpoint RNG State Capture
During checkpoint serialization (`AtomicCheckpointManager`), the exact internal state tuples of Python `random`, NumPy RNG, and PyTorch CPU/MPS devices are bundled directly into the `.pt` dictionary payload. Resuming training restores these state tuples prior to loading data iterators, guaranteeing identical mini-batch sequences.

---

## 3. Deterministic Algorithmic Formulations

Key mathematical routines are designed with deterministic invariants:
1. **Precomputed Stirling Lookup Table**: Exact values of $\ln(k!)$ for $k \le 1024$ are compiled statically at initialization, avoiding floating-point drift across platforms.
2. **Orthogonal Procrustes SVD**: SVD decomposition ($X^T Y = U \Sigma V^T$) guarantees sign consistency and deterministic rotation matrix derivation.
3. **Runge-Kutta 4 Integrator**: Fixed step-size RK4 integration of neural ODE vector fields produces deterministic continuous trajectories independent of variable-step solver tolerances.
4. **Causal Mask Lower-Triangular Indices**: Temporal attention masks are statically constructed with exact boolean lower-triangular masks (`torch.tril`).

---

## 4. Cryptographic Provenance and Acceptance Manifest

Every acceptance run generates a cryptographic audit trail:
- **Partition Fingerprints**: SHA-256 hashes of trial timestamps, event indices, and behavioral tensors are logged to `EVALUATION_FINGERPRINTS.json`.
- **System Verification Report**: `SYSTEM_VERIFICATION.json` documents the pass/fail status and execution timestamps of all 12 negative controls and baseline checks.
- **Master Acceptance Manifest**: `ACCEPTANCE_MANIFEST.json` aggregates SHA-256 digests of all generated artifacts. Modifying any source input or metric output invalidates the manifest signature.
