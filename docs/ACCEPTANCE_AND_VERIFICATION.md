# Acceptance Bundle and Verification Engine Architecture

This document describes the cryptographic verification engine, executable negative controls, and acceptance bundle generation pipeline implemented in `nocturne.verification`.

---

## 1. Overview and Standards Compliance

The Nocturne verification subsystem provides end-to-end certification that all scientific models, data contracts, streaming loops, and evaluation metrics satisfy acceptance gates BA01 through BA12.

The subsystem consists of three core components:
1. `NegativeControlSuite` (`nocturne.verification.controls`): A collection of 12 executable test cases demonstrating that illegal operations, temporal inversions, corrupted weights, causal leakage, and mathematical edge cases are actively trapped.
2. `SystemVerifier` (`nocturne.verification.verifier`): Regression auditor that checks mathematical exactness (Ramanujan Stirling bounds, Poisson deviance), dataset fixtures, synthetic circuit generation, and online streaming equivalence.
3. `AcceptanceBundleGenerator` (`nocturne.verification.acceptance`): Orchestrator that compiles complete provenance documents, structural datasets summaries, evaluation fingerprints, and SHA-256 manifests into an immutable distribution bundle.

---

## 2. Executable Negative Controls (NC01 to NC12)

| Control ID | Target Subsystem | Failure Injected | Catching Invariant |
|---|---|---|---|
| NC01 | `contracts.time` | Inverted interval where `end_s <= start_s` | `TimeInterval` validation rejects non-positive duration. |
| NC02 | `contracts.splits` | Leaked trial ID present in both train and test | `DatasetSplit.verify_zero_overlap()` rejects partition collision. |
| NC03 | `evaluation.integrity` | Non-finite values or negative firing rates in predictions | `EvaluationLabelIntegrityAuditor` flags violations. |
| NC04 | `evaluation.bits_per_spike` | Zero total spikes evaluated in denominator | `BitsPerSpikeEvaluator` returns `None` without division by zero. |
| NC05 | `inference.causality` | Non-causal future-to-past reverse time dependencies | `CausalLeakageAuditor` traps prefix variance across 10 cases. |
| NC06 | `binning.likelihood_math` | Negative spike count input | `get_log_factorial` rejects negative integers. |
| NC07 | `binning.likelihood_math` | NaN spike count input | `get_log_factorial` rejects non-finite values. |
| NC08 | `contracts.masks` | Channel dimension mismatch in `ObservationMasks` | `ObservationMasks` post-init validator enforces exact shape. |
| NC09 | `streaming.state` | Non-positive sliding window size | `StreamingBufferState` rejects `max_window_bins < 1`. |
| NC10 | `geometry.alignment` | Dimensionality mismatch between source and target | `OrthogonalProcrustesAligner` rejects shape mismatch. |
| NC11 | `training.recovery` | Checkpoint dictionary containing NaN weight values | `TrainingRecovery.verify_checkpoint_integrity` rejects NaNs. |
| NC12 | `models.attention` | Observation mask sequence length incompatible with input | Attention model rejects mismatched sequence dimensions. |

All 12 negative controls are executed programmatically via `NegativeControlSuite.run_all()`.

---

## 3. Cryptographic Acceptance Bundle Structure

Running `make acceptance` or calling `AcceptanceBundleGenerator().build_bundle()` generates the following artifacts in `artifacts/acceptance_bundle/`:

### File Manifest
- `ENVIRONMENT.json`:
  Contains machine architecture, OS kernel, Python version, PyTorch version, MPS/CUDA accelerator state, and exact Git commit hash.
- `SOURCE_INPUTS.json`:
  Contains DANDI archive identifiers (`000128` MC_Maze, `000129` MC_RTT, `000127` Area2_Bump), licenses (CC-BY-4.0), and synthetic circuit declarations.
- `DATASETS.json`:
  Contains trial counts, neuron counts, duration, temporal bin width ($\Delta t = 0.02$ s), split distributions, and zero-overlap verification flags.
- `EVALUATION_FINGERPRINTS.json`:
  Contains SHA-256 binary digests of raw binned spike counts and test predictions.
- `SYSTEM_VERIFICATION.json`:
  Full diagnostic execution log of all checks and negative controls.
- `ACCEPTANCE_MANIFEST.json`:
  Cryptographic index containing individual SHA-256 digests and byte counts for every file in the bundle.
