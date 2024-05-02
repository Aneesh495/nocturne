# Nocturne Research Laboratory Makefile
SHELL := /bin/bash
PYTHON ?= .venv/bin/python
UV ?= uv

.PHONY: bootstrap doctor build test test-likelihood test-causality synthetic infer-stream latent-study evaluate-nlb data train evaluate ablate stress bench docs demo acceptance verify release clean

bootstrap:
	@echo "==> Bootstrapping virtual environment and dependencies..."
	$(UV) venv --python 3.12 .venv || true
	$(UV) pip install --python $(PYTHON) torch numpy scipy pandas h5py pynwb pydantic pyyaml matplotlib scikit-learn pytest pytest-cov
	$(UV) pip install --python $(PYTHON) --no-deps nlb-tools || true
	$(PYTHON) -m pip install -e . --no-deps || true
	@echo "==> Bootstrap complete."

doctor:
	@echo "==> Running environment diagnostic..."
	@$(PYTHON) -c "import torch, numpy, scipy, h5py, pydantic; print(f'Python: 3.12, PyTorch: {torch.__version__}, MPS: {torch.backends.mps.is_available()}, NumPy: {numpy.__version__}')"
	@which $(PYTHON)
	@git status --short

build:
	@echo "==> Building Nocturne package..."
	$(PYTHON) -m pip install -e . --no-deps

test:
	@echo "==> Running mathematical and unit tests..."
	$(PYTHON) -m pytest tests/ -v --tb=short

test-likelihood:
	@echo "==> Running point-process likelihood and numerical gradient tests..."
	$(PYTHON) -m pytest tests/binning/test_likelihood_math.py tests/models/test_poisson_loss.py -v

test-causality:
	@echo "==> Running causal prefix invariance and leakage tests..."
	$(PYTHON) -m pytest tests/inference/test_causal_invariance.py -v

synthetic:
	@echo "==> Generating and verifying controlled synthetic circuits..."
	$(PYTHON) -m nocturne.cli synthetic generate --all --trials 5000

infer-stream:
	@echo "==> Running online causal streaming test..."
	$(PYTHON) -m nocturne.cli stream run --dataset mc_maze_synthetic --duration 60

latent-study:
	@echo "==> Running gauge-aware latent geometry and fixed-point analysis..."
	$(PYTHON) -m nocturne.cli geometry analyze --all

evaluate-nlb:
	@echo "==> Evaluating models on released local NLB test units..."
	$(PYTHON) -m nocturne.cli evaluate nlb --datasets mc_maze,mc_rtt,area2_bump

data:
	@echo "==> Acquiring and validating dataset manifests..."
	$(PYTHON) -m nocturne.cli data prepare --all

train:
	@echo "==> Training primary models with 3 seeds..."
	$(PYTHON) -m nocturne.cli train run --config configs/primary_campaign.yaml

evaluate:
	@echo "==> Evaluating frozen checkpoints on held-out splits..."
	$(PYTHON) -m nocturne.cli evaluate run --config configs/primary_campaign.yaml

ablate:
	@echo "==> Running ablation study matrix..."
	$(PYTHON) -m nocturne.cli ablate run --config configs/ablations.yaml

stress:
	@echo "==> Executing fault injection and stream recovery stress suite..."
	$(PYTHON) -m pytest tests/streaming/test_fault_recovery.py -v

bench:
	@echo "==> Running throughput, memory, and latency benchmarks..."
	$(PYTHON) -m nocturne.cli bench run --output artifacts/benchmarks.json

docs:
	@echo "==> Verifying documentation, derivations, and links..."
	$(PYTHON) -m nocturne.cli docs verify

demo:
	@echo "==> Running end-to-end trained model demo..."
	$(PYTHON) -m nocturne.cli demo run

acceptance:
	@echo "==> Executing full fresh acceptance campaign and emitting evidence bundle..."
	$(PYTHON) -m nocturne.verification.acceptance run --output evidence/latest

verify:
	@echo "==> Verifying evidence bundle integrity..."
	$(PYTHON) -m nocturne.verification.verifier verify --evidence evidence/latest

release:
	@echo "==> Packaging verified release artifacts..."
	$(PYTHON) -m nocturne.cli release package

clean:
	@rm -rf build dist *.egg-info .pytest_cache .coverage
	@find . -name "__pycache__" -exec rm -rf {} +
