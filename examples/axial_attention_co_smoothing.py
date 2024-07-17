#!/usr/bin/env python3
"""Spatiotemporal axial attention neural population co-smoothing example."""

from __future__ import annotations

import sys
import numpy as np
import torch

from nocturne.inference.co_smoothing import CoSmoother
from nocturne.inference.contracts import InferenceMode, InferenceRequest, InferenceResult
from nocturne.models.attention import AttentionConfig, SpatiotemporalAttentionModel
from nocturne.synthetic.generator import SyntheticDatasetGenerator


def main() -> int:
    print("=" * 60)
    print("Nocturne Neural Dynamics: Axial Attention Co-Smoothing")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    total_neurons = 20
    held_out_count = 4
    held_in_count = total_neurons - held_out_count
    held_in_indices = list(range(held_in_count))
    held_out_indices = list(range(held_in_count, total_neurons))
    bin_width_s = 0.02
    seq_len = 30
    batch_size = 4

    print(f"Total neural population (N)    : {total_neurons}")
    print(f"Held-in observed channels       : {len(held_in_indices)} {held_in_indices[:5]}...")
    print(f"Held-out target channels        : {len(held_out_indices)} {held_out_indices}")

    # 1. Generate synthetic benchmark recording
    generator = SyntheticDatasetGenerator(seed=42)
    sims, _ = generator.generate_regime_corpus(
        regime_name="nonlinear_oscillator",
        num_trials=batch_size,
        num_neurons=total_neurons,
        bin_width_s=bin_width_s,
        trial_duration_s=seq_len * bin_width_s,
    )
    counts = np.stack([s.spike_counts[:seq_len] for s in sims], axis=0)  # [B, T, N]
    true_rates = np.stack([s.true_rates_hz[:seq_len] for s in sims], axis=0)  # [B, T, N]
    counts_th = torch.from_numpy(counts).float()

    # 2. Instantiate Spatiotemporal Axial Attention Model
    config = AttentionConfig(
        obs_dim=total_neurons,
        d_model=32,
        num_heads=2,
        num_layers=2,
        dim_feedforward=64,
        dropout=0.0,
        bin_width_s=bin_width_s,
        max_seq_len=max(64, seq_len),
        causal=False,
    )
    model = SpatiotemporalAttentionModel(config, latent_dim=4)
    print(f"Axial attention model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    # 3. Setup CoSmoother engine
    co_smoother = CoSmoother(model=model)
    request = InferenceRequest(
        counts=counts_th,
        mode=InferenceMode.OFFLINE_SMOOTHING,
        held_in_indices=held_in_indices,
        held_out_indices=held_out_indices,
        bin_width_s=bin_width_s,
    )

    # 4. Perform offline co-smoothing inference
    result: InferenceResult = co_smoother.infer(request)
    pred_rates = result.predicted_rates_hz.cpu().numpy()
    held_out_pred_rates = result.held_out_rates_hz.cpu().numpy()
    held_out_true_rates = true_rates[:, :, held_out_indices]

    # 5. Compute evaluation metrics on held-out channels
    held_out_rmse = float(np.sqrt(np.mean((held_out_pred_rates - held_out_true_rates) ** 2)))
    mean_held_out_rate = float(np.mean(held_out_pred_rates))

    print("\nCo-Smoothing Inference Results:")
    print(f"  Inference latency wall-time   : {result.latency_seconds * 1000.0:.2f} ms")
    print(f"  Predicted rates shape         : {list(result.predicted_rates_hz.shape)}")
    print(f"  Held-out predicted rates shape: {list(result.held_out_rates_hz.shape)}")
    print(f"  Latent representations shape  : {list(result.latents.shape)}")
    print(f"  Held-out channels mean rate   : {mean_held_out_rate:.2f} Hz")
    print(f"  Held-out reconstruction RMSE  : {held_out_rmse:.4f} Hz")

    print("\nAxial attention co-smoothing completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
