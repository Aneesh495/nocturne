#!/usr/bin/env python3
"""Continuous-time Variational Latent Dynamics Model (VLDM) training example."""

from __future__ import annotations

import sys
import numpy as np
import torch

from nocturne.models.vldm.config import VLDMConfig
from nocturne.models.vldm.model import VariationalLatentDynamicsModel
from nocturne.synthetic.generator import SyntheticDatasetGenerator


def main() -> int:
    print("=" * 60)
    print("Nocturne Neural Dynamics: Continuous-Time VLDM Training")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Synthesize neural population spikes from limit cycle oscillator
    num_neurons = 16
    latent_dim = 4
    bin_width_s = 0.02
    trial_duration_s = 0.6
    batch_size = 4

    generator = SyntheticDatasetGenerator(seed=42)
    sims, _ = generator.generate_regime_corpus(
        regime_name="nonlinear_oscillator",
        num_trials=batch_size,
        num_neurons=num_neurons,
        bin_width_s=bin_width_s,
        trial_duration_s=trial_duration_s,
    )
    spike_counts = np.stack([s.spike_counts for s in sims], axis=0)  # [B, T, N]
    counts_th = torch.from_numpy(spike_counts).float()
    B, T, N = counts_th.shape

    print(f"Data batch shape: [B={B}, T={T}, N={N}]")
    print(f"Total spikes in batch: {int(counts_th.sum().item())}")

    # 2. Configure and instantiate VLDM
    config = VLDMConfig(
        obs_dim=num_neurons,
        latent_dim=latent_dim,
        encoder_hidden_dim=32,
        decoder_hidden_dim=32,
        use_inferred_drive=False,
        bin_width_s=bin_width_s,
    )
    model = VariationalLatentDynamicsModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)

    print(f"VLDM architecture initialized with latent dim {config.latent_dim}")
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters())}")

    # 3. Optimization loop
    print("\nExecuting continuous latent dynamics optimization:")
    num_steps = 10
    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        rates_hz, latents, loss, metrics = model(counts_th)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        recon_loss = metrics.get("loss_recon", 0.0)
        kl_z0 = metrics.get("loss_kl_z0", 0.0)
        print(
            f"  Step {step:02d}/{num_steps:02d} | Total ELBO: {loss.item():.4f} | "
            f"Recon: {recon_loss:.4f} | KL(z0): {kl_z0:.4f}"
        )

    # 4. Out-of-sample forward evaluation
    model.eval()
    with torch.no_grad():
        rates_hz, latents, eval_loss, eval_metrics = model(counts_th)
        mean_rate = float(rates_hz.mean().item())
        min_rate = float(rates_hz.min().item())
        max_rate = float(rates_hz.max().item())

    print("\nPost-training diagnostic evaluation:")
    print(f"  Final Evaluation Loss: {eval_loss.item():.4f}")
    print(f"  Latent trajectory shape: {list(latents.shape)}")
    print(f"  Predicted firing rates : min={min_rate:.2f} Hz, mean={mean_rate:.2f} Hz, max={max_rate:.2f} Hz")
    print("\nVLDM continuous training completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
