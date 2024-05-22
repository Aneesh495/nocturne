"""Empirical benchmarks measuring streaming latency, throughput, and memory bounds."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn

from nocturne.streaming.streamer import OnlineCausalStreamer


@dataclass(frozen=True)
class StreamingBenchmarkMetrics:
    """Benchmark performance metrics for online causal neural stream processing."""

    num_chunks_evaluated: int
    chunk_size_bins: int
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    throughput_chunks_per_sec: float
    throughput_bins_per_sec: float
    is_realtime_capable: bool
    memory_bounded_verified: bool


class StreamingBenchmarkRunner:
    """Executes empirical benchmarks measuring latency distributions and memory bounds."""

    def __init__(self, bin_duration_ms: float = 20.0):
        self.bin_duration_ms = bin_duration_ms

    def run_benchmark(
        self,
        streamer: OnlineCausalStreamer,
        num_chunks: int = 200,
        chunk_size_bins: int = 1,
        warmup_chunks: int = 20,
    ) -> StreamingBenchmarkMetrics:
        """Measure per-chunk execution latencies and memory stability.

        Args:
            streamer: OnlineCausalStreamer instance.
            num_chunks: Number of chunks to evaluate.
            chunk_size_bins: Temporal bins per chunk (default 1 bin).
            warmup_chunks: Number of initial iterations to disregard for cache warmup.

        Returns:
            StreamingBenchmarkMetrics containing statistical latency distribution.
        """
        streamer.reset()
        obs_dim = streamer.obs_dim
        device = streamer.device

        # Warmup phase
        for _ in range(warmup_chunks):
            dummy = torch.randint(0, 3, (1, chunk_size_bins, obs_dim), dtype=torch.float32, device=device)
            streamer.push_chunk(dummy)

        streamer.reset()

        latencies_ms: List[float] = []
        buffer_sizes: List[int] = []

        total_t0 = time.perf_counter()

        for _ in range(num_chunks):
            chunk = torch.randint(0, 3, (1, chunk_size_bins, obs_dim), dtype=torch.float32, device=device)
            res = streamer.push_chunk(chunk)
            latencies_ms.append(res.latency_ms)
            buffer_sizes.append(streamer.state.window_counts.shape[1])

        total_elapsed = time.perf_counter() - total_t0

        mean_lat = float(np.mean(latencies_ms))
        p50_lat = float(np.percentile(latencies_ms, 50))
        p95_lat = float(np.percentile(latencies_ms, 95))
        p99_lat = float(np.percentile(latencies_ms, 99))
        max_lat = float(np.max(latencies_ms))

        throughput_chunks = num_chunks / max(1e-6, total_elapsed)
        throughput_bins = (num_chunks * chunk_size_bins) / max(1e-6, total_elapsed)

        # Realtime capable if p95 latency is strictly under the temporal bin duration (e.g. 20ms)
        is_realtime = p95_lat < (chunk_size_bins * self.bin_duration_ms)

        # Buffer memory bounded: buffer size must stabilize at max_window_bins once reached
        max_seen = max(buffer_sizes)
        memory_bounded = max_seen <= streamer.max_window_bins

        return StreamingBenchmarkMetrics(
            num_chunks_evaluated=num_chunks,
            chunk_size_bins=chunk_size_bins,
            mean_latency_ms=mean_lat,
            p50_latency_ms=p50_lat,
            p95_latency_ms=p95_lat,
            p99_latency_ms=p99_lat,
            max_latency_ms=max_lat,
            throughput_chunks_per_sec=throughput_chunks,
            throughput_bins_per_sec=throughput_bins,
            is_realtime_capable=is_realtime,
            memory_bounded_verified=memory_bounded,
        )
