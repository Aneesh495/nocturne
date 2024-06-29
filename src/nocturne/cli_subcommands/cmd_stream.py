"""Online causal streaming simulation and latency benchmark CLI subcommand."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn

from nocturne.models.attention.config import AttentionConfig
from nocturne.models.attention.model import SpatiotemporalAttentionModel
from nocturne.streaming.benchmarks import StreamingBenchmarkMetrics, StreamingBenchmarkRunner
from nocturne.streaming.streamer import OnlineCausalStreamer


def run_stream_command(
    num_neurons: int = 24,
    chunk_size_bins: int = 1,
    num_chunks: int = 100,
    window_bins: int = 50,
    bin_width_ms: int = 20,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> int:
    """Execute online causal streaming latency and throughput benchmark."""
    if verbose:
        print("=== Nocturne Online Causal Streaming Simulation ===")
        print(f"Neurons: {num_neurons} | Chunk size: {chunk_size_bins} bin(s) | Window: {window_bins} bins | Bin width: {bin_width_ms}ms")
        print(f"Simulating {num_chunks} streaming chunks...\n")

    cfg = AttentionConfig(
        obs_dim=num_neurons,
        d_model=32,
        n_heads=2,
        num_layers=1,
        causal=True,
    )
    model = SpatiotemporalAttentionModel(cfg)

    streamer = OnlineCausalStreamer(
        model=model,
        obs_dim=num_neurons,
        max_window_bins=window_bins,
    )

    runner = StreamingBenchmarkRunner(bin_duration_ms=float(bin_width_ms))
    metrics: StreamingBenchmarkMetrics = runner.run_benchmark(
        streamer=streamer,
        num_chunks=num_chunks,
        chunk_size_bins=chunk_size_bins,
        warmup_chunks=min(15, num_chunks // 4),
    )

    if verbose:
        print("Streaming Performance Distribution:")
        print(f"  Mean Latency:    {metrics.mean_latency_ms:6.2f} ms")
        print(f"  P50  Latency:    {metrics.p50_latency_ms:6.2f} ms")
        print(f"  P95  Latency:    {metrics.p95_latency_ms:6.2f} ms")
        print(f"  P99  Latency:    {metrics.p99_latency_ms:6.2f} ms")
        print(f"  Max  Latency:    {metrics.max_latency_ms:6.2f} ms")
        print(f"  Throughput:      {metrics.throughput_bins_per_sec:6.1f} bins/sec ({metrics.throughput_chunks_per_sec:6.1f} chunks/sec)")
        print(f"  Budget Limit:    {bin_width_ms * chunk_size_bins:6.2f} ms")
        print(f"  Realtime Status: {'REALTIME CAPABLE (P99 < Budget)' if metrics.is_realtime_capable else 'DEADLINE EXCEEDED'}")
        print(f"  Memory Bounded:  {'VERIFIED (Sliding Buffer Stable)' if metrics.memory_bounded_verified else 'FAILED'}")

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "num_chunks": metrics.num_chunks_evaluated,
            "chunk_size_bins": metrics.chunk_size_bins,
            "mean_latency_ms": metrics.mean_latency_ms,
            "p50_latency_ms": metrics.p50_latency_ms,
            "p95_latency_ms": metrics.p95_latency_ms,
            "p99_latency_ms": metrics.p99_latency_ms,
            "max_latency_ms": metrics.max_latency_ms,
            "throughput_bins_per_sec": metrics.throughput_bins_per_sec,
            "is_realtime_capable": metrics.is_realtime_capable,
            "memory_bounded_verified": metrics.memory_bounded_verified,
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        if verbose:
            print(f"\nStreaming benchmark metrics saved to: {p}")

    return 0 if metrics.is_realtime_capable else 1


def add_stream_arguments(parser: argparse.ArgumentParser) -> None:
    """Register CLI arguments for the stream subcommand."""
    parser.add_argument(
        "--neurons",
        type=int,
        default=24,
        help="Number of observed neurons in stream",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1,
        help="Chunk size in temporal bins",
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=100,
        help="Number of streaming chunks to evaluate",
    )
    parser.add_argument(
        "--window-bins",
        type=int,
        default=50,
        help="Bounded history buffer window capacity in bins",
    )
    parser.add_argument(
        "--bin-width-ms",
        type=int,
        default=20,
        help="Bin width in milliseconds defining realtime deadline",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional path to write streaming metrics JSON",
    )
