"""Verification of equivalence between batch causal inference and chunked streaming."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import torch
import torch.nn as nn

from nocturne.streaming.streamer import OnlineCausalStreamer


@dataclass(frozen=True)
class EquivalenceReport:
    """Report comparing chunked streaming outputs against full-prefix batch reference."""

    chunk_sizes_evaluated: List[int]
    max_rate_deviations: Dict[int, float]
    max_latent_deviations: Dict[int, float]
    is_equivalent: bool


class StreamingEquivalenceAuditor:
    """Verifies that chunked streaming produces identical outputs to batch causal filtering."""

    def __init__(self, tolerance: float = 5e-5):
        self.tolerance = tolerance

    @torch.no_grad()
    def audit_equivalence(
        self,
        model: nn.Module,
        counts: torch.Tensor,
        chunk_sizes: List[int] | None = None,
    ) -> EquivalenceReport:
        """Test equivalence across multiple chunk sizes (e.g. 1 bin, 2 bins, 5 bins).

        Args:
            model: PyTorch model.
            counts: Input spike counts [1, T, N].
            chunk_sizes: List of chunk lengths to evaluate.

        Returns:
            EquivalenceReport summarizing deviations for each chunk size.
        """
        model.eval()
        if chunk_sizes is None:
            chunk_sizes = [1, 2, 5]

        T = counts.shape[1]
        obs_dim = counts.shape[2]

        # 1. Run full-batch causal reference
        batch_rates, batch_latents, _, _ = model(counts, causal=True)

        rate_devs: Dict[int, float] = {}
        latent_devs: Dict[int, float] = {}
        all_passed = True

        for chunk_sz in chunk_sizes:
            streamer = OnlineCausalStreamer(
                model=model,
                obs_dim=obs_dim,
                max_window_bins=T + 10,  # Ensure full history is retained for comparison
                device=counts.device,
            )

            streamed_rates_list = []
            streamed_latents_list = []

            for start in range(0, T, chunk_sz):
                end = min(start + chunk_sz, T)
                chunk = counts[:, start:end, :]
                res = streamer.push_chunk(chunk)
                streamed_rates_list.append(res.predicted_rates_hz)
                streamed_latents_list.append(res.latents)

            concatenated_rates = torch.cat(streamed_rates_list, dim=1)
            concatenated_latents = torch.cat(streamed_latents_list, dim=1)

            max_rate_diff = (batch_rates - concatenated_rates).abs().max().item()
            max_latent_diff = (batch_latents - concatenated_latents).abs().max().item()

            rate_devs[chunk_sz] = max_rate_diff
            latent_devs[chunk_sz] = max_latent_diff

            if max_rate_diff > self.tolerance or max_latent_diff > self.tolerance:
                all_passed = False

        return EquivalenceReport(
            chunk_sizes_evaluated=chunk_sizes,
            max_rate_deviations=rate_devs,
            max_latent_deviations=latent_devs,
            is_equivalent=all_passed,
        )
