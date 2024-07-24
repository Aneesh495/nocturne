# Nocturne CLI Reference Manual

Nocturne exposes a unified command-line interface (`nocturne`) providing complete terminal access to synthetic circuit simulation, dynamics estimation, ablation campaigns, geometric audits, and visualization rendering.

```bash
nocturne <subcommand> [flags]
```

---

## Subcommands Overview

| Subcommand | Description | Primary Output |
|:---|:---|:---|
| `doctor` | System environment and hardware probe | Terminal diagnostic report |
| `synthetic` | Circuit simulation runner across 4 regimes | Simulation counts and summary |
| `verify` | Negative controls and subsystem audit | Pass/Fail audit table |
| `acceptance` | Acceptance gate validation bundle | Signed JSON acceptance manifest |
| `benchmark` | Multi-seed benchmark suite execution | Bits-per-spike and deviance metrics |
| `ablate` | 5-family ablation study matrix runner | Ablation comparison table |
| `stream` | Online streaming latency and throughput benchmark | Latency percentiles and throughput |
| `raster` | Spike raster ASCII and SVG generator | ASCII raster or standalone SVG |
| `phase_portrait` | Latent phase portrait and vector flow visualizer | Vector SVG phase portrait |
| `matrix` | Scientific matrix aggregator | LaTeX, Markdown, or CSV table |
| `falsify` | Adversarial attractor falsification runner | Surrogate null hypothesis report |
| `sensitivity` | Hyperparameter sensitivity sweep runner | Dimensionality saturation knee report |
| `spectrogram` | Time-frequency spectrogram visualizer | Vector SVG power spectrogram |
| `manifold` | 3D isometric manifold mesh visualizer | Vector SVG 3D manifold orbit |

---

## Subcommand Details

### 1. `doctor`
Inspects CPU, Apple Silicon MPS / NVIDIA CUDA accelerators, PyTorch version, thread allocation, and workspace integrity.
```bash
nocturne doctor
```

### 2. `synthetic`
Simulates synthetic neural populations across the 4 dynamical regimes.
```bash
nocturne synthetic --regime nonlinear_oscillator --trials 5 --neurons 16 --duration 1.0 --bin-width 20
```
- `--regime`: Choice of `linear_oscillator`, `nonlinear_oscillator`, `bistable_memory`, `driven_recurrent`.
- `--trials`: Number of independent realizations.
- `--neurons`: Monitored population channel count.
- `--duration`: Trial duration in seconds.
- `--bin-width`: Bin width in milliseconds.

### 3. `verify`
Executes negative controls and architectural subsystem verification audits.
```bash
nocturne verify
```

### 4. `acceptance`
Executes full acceptance test suite across capability registers B01 through B32 and gates BA01 through BA12, generating cryptographic manifest.
```bash
nocturne acceptance --output-manifest acceptance_manifest.json
```

### 5. `benchmark`
Runs multi-seed benchmark evaluation on synthetic and recording benchmark corpora.
```bash
nocturne benchmark --dataset mc_maze --seeds 42,43,44 --models attention,plds,poisson_factor
```

### 6. `ablate`
Runs 5-family ablation study matrix across objective, masking, capacity, causality, and dynamics drive variants.
```bash
nocturne ablate --family family_1_objective --dataset mc_maze --steps 20
```

### 7. `stream`
Benchmarks real-time causal streaming processing latency against hardware deadlines.
```bash
nocturne stream --chunk-size 2 --buffer-size 50 --num-chunks 100 --deadline 20.0
```

### 8. `raster`
Renders neural spike counts into lightweight ASCII terminal displays or standalone vector SVG files.
```bash
nocturne raster --neurons 16 --bins 50 --ascii
nocturne raster --neurons 16 --bins 100 --output-svg raster.svg
```

### 9. `phase_portrait`
Renders continuous vector field flows and latent trajectory orbits into vector SVG documents.
```bash
nocturne phase_portrait --regime nonlinear_oscillator --output-svg phase_portrait.svg
```

### 10. `matrix`
Aggregates experimental and ablation records into publications tables.
```bash
nocturne matrix --format latex --input-json results.json
nocturne matrix --format markdown --input-json results.json
```

### 11. `falsify`
Runs adversarial attractor falsification tests using phase-randomized Fourier surrogate data and stochastic diffusion.
```bash
nocturne falsify --surrogates 50 --seeds 25
```

### 12. `sensitivity`
Executes hyperparameter sensitivity sweeps and identifies the dimensionality saturation knee.
```bash
nocturne sensitivity --dims 2,4,8,16,32 --output-knee
```

### 13. `spectrogram`
Computes short-time Fourier transform (STFT) power spectrograms and renders vector SVG visual representations.
```bash
nocturne spectrogram --window-size 32 --hop-size 8 --output-svg spectrogram.svg
```

### 14. `manifold`
Projects 3D neural population trajectories onto 2D viewports using orthographic isometric projections.
```bash
nocturne manifold --elevation 30.0 --azimuth 45.0 --output-svg manifold_3d.svg
```
