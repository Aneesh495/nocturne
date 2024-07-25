# Contributing to Nocturne

Nocturne is an original research laboratory for probabilistic learning of neural population dynamics, Lie group representations, continuous-time differential geometry, and real-time closed-loop neuromodulation.

Contributions that advance mathematical rigor, algorithmic efficiency, or benchmark breadth are welcome.

---

## 1. Engineering Principles

### Mathematical Integrity
All algorithms must implement published or formally derived mathematical formulations. Approximations or heuristics must be explicitly documented and accompanied by negative controls.

### Deterministic Bitwise Reproducibility
Every workflow, model initialization, and data generator must accept explicit pseudorandom seeds and reproduce bitwise identical outputs across executions under identical hardware environments.

### Zero External Service Dependency
Benchmark evaluations, unit tests, and CLI subcommands must execute entirely locally using fixtures and offline data adapters without connecting to external servers or closed cloud platforms.

### Strictly No Em Dashes
Prose, docstrings, code comments, and commit messages must strictly avoid em dashes. Use standard hyphens, colons, or parentheses instead.

---

## 2. Development Setup

Clone the repository and set up a clean Python virtual environment (Python 3.10+):

```bash
git clone https://github.com/Aneesh495/nocturne-neural.git
cd Nocturne
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Verify your environment using the doctor diagnostic probe:
```bash
nocturne doctor
```

---

## 3. Testing Standards

Every change must be accompanied by comprehensive unit tests under `tests/`.

Run the full automated test suite:
```bash
pytest
```

Run test suite with coverage reporting:
```bash
pytest --cov=nocturne --cov-report=term-missing
```

All unit tests must pass before submitting contributions.

---

## 4. Architectural Conventions

### Generative Models (`src/nocturne/models/`)
- Must subclass `torch.nn.Module`.
- Must output tuple `(predicted_rates_hz, latents, loss, metrics_dict)`.
- Rates must be strictly non-negative, guaranteed via Softplus activation with $\beta > 0$.

### Synthetic Circuits (`src/nocturne/synthetic/`)
- Must subclass `SyntheticCircuit` from `nocturne.synthetic.base`.
- Must implement analytical or numerical integration via `integrate_trajectory`.

### Workflows (`src/nocturne/workflows/`)
- Must return an immutable frozen dataclass containing scalar summary metrics and visualizations.

---

## 5. Git Commit Standards

- Commits must represent atomic, substantive mathematical mechanisms, algorithms, tests, or documentation.
- Do not create empty, artificial, or squashed commits.
- Follow conventional commit prefixes: `feat`, `fix`, `test`, `docs`, `refactor`, `perf`.
