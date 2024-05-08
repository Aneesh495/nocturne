# Nocturne Build Status and Capability Ledger

**Current Phase**: Phase 0 — Environment, Repository Initialization, and Project Charter  
**Local Date/Time**: 2026-10-10  
**Current Git Commit**: Pending initial commit  
**Remote Repository**: `Aneesh495/nocturne-neural`  
**Host Target**: macOS arm64 (Apple Silicon), PyTorch 2.14.1 with MPS support, Python 3.12  

---

## Capability Register (B01 - B32)

| ID | Description | Source Path | Test Path | Status |
|---|---|---|---|---|
| B01 | Versioned recording/trial/neuron/time/unit contracts | `src/nocturne/contracts/` | `tests/contracts/` | passed |
| B02 | Bounded NWB/DANDI/NLB source acquisition and provenance | `src/nocturne/recordings/` | `tests/recordings/` | unverified |
| B03 | Exact binning, alignment, validity, and spike-count accounting | `src/nocturne/binning/` | `tests/binning/` | unverified |
| B04 | Official and grouped/chronological split auditing | `src/nocturne/recordings/splits.py` | `tests/recordings/` | passed (split contracts validated) |
| B05 | Explicit observation/held-out/padding/forecast masks | `src/nocturne/contracts/masks.py` | `tests/contracts/` | passed |
| B06 | Original independent Poisson-rate baseline | `src/nocturne/models/baselines/` | `tests/models/` | unverified |
| B07 | Original low-rank point-process baseline | `src/nocturne/models/baselines/` | `tests/models/` | unverified |
| B08 | Original linear/Poisson latent-state inference baseline | `src/nocturne/models/baselines/` | `tests/models/` | unverified |
| B09 | Original variational nonlinear latent dynamics | `src/nocturne/models/vldm/` | `tests/models/` | unverified |
| B10 | Explicit inferred-drive versus autonomous dynamics modes | `src/nocturne/models/vldm/` | `tests/models/` | unverified |
| B11 | Original axial neuron/time masked attention model | `src/nocturne/models/attention/` | `tests/models/` | unverified |
| B12 | Positive-rate readout and correct count likelihood | `src/nocturne/models/likelihoods.py` | `tests/models/` | unverified |
| B13 | Real masked/variational training and validation selection | `src/nocturne/training/` | `tests/training/` | unverified |
| B14 | Separate smoothing, filtering, and forecasting contracts | `src/nocturne/inference/` | `tests/inference/` | unverified |
| B15 | Causal prefix-invariance and future-data leakage checks | `src/nocturne/inference/causality.py` | `tests/inference/` | unverified |
| B16 | Causal streaming state, reset, and checkpoint lifecycle | `src/nocturne/streaming/` | `tests/streaming/` | unverified |
| B17 | Original controlled circuits with known latent truth | `src/nocturne/synthetic/` | `tests/synthetic/` | unverified |
| B18 | Synthetic noise/dropout/input/misspecification regimes | `src/nocturne/synthetic/noise.py` | `tests/synthetic/` | unverified |
| B19 | Independent bits-per-spike and likelihood recomputation | `src/nocturne/evaluation/` | `tests/evaluation/` | unverified |
| B20 | Frozen train-only behavior/representation probes | `src/nocturne/evaluation/probes.py` | `tests/evaluation/` | unverified |
| B21 | Gauge-aware latent alignment and geometry checks | `src/nocturne/geometry/` | `tests/geometry/` | unverified |
| B22 | Declared fixed-point/Jacobian/dynamics diagnostics | `src/nocturne/geometry/fixed_points.py` | `tests/geometry/` | unverified |
| B23 | Three-seed public-recording comparison campaign | `src/nocturne/campaign/` | `tests/campaign/` | unverified |
| B24 | Future prediction and missing-neuron robustness | `src/nocturne/inference/robustness.py` | `tests/inference/` | unverified |
| B25 | Likelihood/coverage and shift diagnostics | `src/nocturne/evaluation/calibration.py` | `tests/evaluation/` | unverified |
| B26 | Objective/input/depth/mask ablation experiments | `src/nocturne/campaign/ablations.py` | `tests/campaign/` | unverified |
| B27 | Atomic training/stream state recovery and bounded windows | `src/nocturne/training/recovery.py` | `tests/training/` | unverified |
| B28 | Genuine raster/rate/latent/error/scientific viewer | `src/nocturne/viewer/` | `tests/viewer/` | unverified |
| B29 | Independent evaluation-label and metric integrity checks | `src/nocturne/evaluation/integrity.py` | `tests/evaluation/` | unverified |
| B30 | Actual streaming latency, memory, and throughput measurements | `src/nocturne/streaming/benchmarks.py`| `tests/streaming/` | unverified |
| B31 | Derivations, data/model cards, source walkthroughs, diagrams | `docs/` | `tests/docs/` | unverified |
| B32 | Fresh acceptance and genuinely executed verifier controls | `src/nocturne/verification/` | `tests/verification/` | unverified |

---

## Acceptance Gates (BA01 - BA12)

| Gate | Description | Target Contract | Status |
|---|---|---|---|
| BA01 | At least 400 authored numerical/semantic cases and 10,000 generated bin/mask/contract cases | Mathematical/accounting unit test suite | unverified |
| BA02 | All three required NLB/DANDI datasets and released test-evaluation identities validated | Official splits and chronological audit | unverified |
| BA03 | At least 5,000 independent synthetic trials in each of 4 regimes | Circuit validation and analytic checks | unverified |
| BA04 | Original latent and attention models plus baselines trained with 3 seeds | Optimizer updates and raw histories | unverified |
| BA05 | Every eligible official local test unit has frozen predictions; ≥1,000,000 valid neuron-bin observations | Public test evaluation pass | unverified |
| BA06 | Positive public bits-per-spike learning gate on ≥2 datasets and synthetic improvement | Held-out BPS verification | unverified |
| BA07 | ≥1,000 future-input/prefix/held-out perturbation cases; causal prefix invariance verified | Perturbation engine pass | unverified |
| BA08 | ≥5 prediction horizons and ≥5 neuron-dropout profiles evaluated | Forecast & dropout robustness matrix | unverified |
| BA09 | ≥200 held-out circuit trials per regime for gauge-aware dynamics analysis | Procrustes/fixed-point audit | unverified |
| BA10 | 5 ablation families with 3 seeds on 2 recordings + 2 circuits | Ablation results compiled | unverified |
| BA11 | 50 interruption cases, 100 malformed/gap cases, 60-min bounded stream run | Fault & benchmark verification | unverified |
| BA12 | Genuine trained demo, complete docs, fresh evidence, ≥12 verifier negative controls | Acceptance bundle & verifier pass | unverified |

---

## Executed Work and Verified Checkpoints

- **2026-10-10**: Verified host environment (Python 3.12.14, uv, PyTorch 2.14.1 with MPS, NumPy 2.5.4, nlb-tools 0.0.4).
- **2026-10-10**: Initialized project skeleton, `pyproject.toml`, `.gitignore`, `LICENSE`, `README.md`, and capability tracking ledger.

## Next Executable Action
Initialize Git repository, configure remote `Aneesh495/nocturne-neural`, verify push to remote, and build core Makefile targets.
