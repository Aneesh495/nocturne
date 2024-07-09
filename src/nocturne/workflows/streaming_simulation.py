"""End-to-end scientific workflow: Online causal streaming simulation and closed-loop stimulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn

from nocturne.models.attention import AttentionConfig, SpatiotemporalAttentionModel
from nocturne.streaming.closed_loop import ClosedLoopStimulator, StimulationEvent
from nocturne.streaming.streamer import OnlineCausalStreamer
from nocturne.synthetic.generator import SyntheticDatasetGenerator
from nocturne.viewer.streaming_monitor import StreamingMonitorVisualizer


@dataclass(frozen=True)
class StreamingSimulationResult:
    """Quantitative outcome of real-time causal streaming and closed-loop stimulation."""

    total_chunks: int
    total_bins: int
    mean_latency_ms: float
    p95_latency_ms: float
    max_latency_ms: float
    deadline_violations: int
    num_stim_events: int
    num_blanked_bins: int
    reconstruction_rmse: float
    ascii_summary: str
    svg_monitor: str
    stim_events: List[StimulationEvent]


def run_streaming_simulation(
    num_neurons: int = 16,
    latent_dim: int = 4,
    total_bins: int = 100,
    chunk_size_bins: int = 2,
    bin_width_ms: int = 20,
    rate_threshold_hz: float = 35.0,
    phase_trigger_target: Optional[float] = None,
    min_isi_s: float = 0.05,
    blanking_duration_s: float = 0.02,
    circuit_name: str = "nonlinear_oscillator",
    seed: int = 42,
    custom_model: Optional[nn.Module] = None,
) -> StreamingSimulationResult:
    """Execute online causal streaming simulation with closed-loop neurostimulation feedback.

    Args:
        num_neurons: Number of monitored neural channels.
        latent_dim: Latent state dimensionality.
        total_bins: Total duration in discrete time bins.
        chunk_size_bins: Number of bins arriving per streaming hardware tick.
        bin_width_ms: Sampling bin width in milliseconds.
        rate_threshold_hz: Spiking rate trigger threshold for closed-loop modulation.
        phase_trigger_target: Optional angular phase target in latent space.
        min_isi_s: Refractory lockout duration between stimulation pulses in seconds.
        blanking_duration_s: Sensor artifact blanking window duration in seconds.
        circuit_name: Name of synthetic circuit regime for ground-truth generation.
        seed: Pseudorandom seed for repeatable circuit dynamics.
        custom_model: Optional pre-trained PyTorch causal model.

    Returns:
        StreamingSimulationResult containing latency profiles, events, and diagnostics.
    """
    bin_width_s = bin_width_ms / 1000.0
    deadline_ms = float(chunk_size_bins * bin_width_ms)

    # 1. Generate ground-truth neural counts from synthetic circuit
    generator = SyntheticDatasetGenerator(seed=seed)
    total_duration_s = total_bins * bin_width_s
    sims, _ = generator.generate_regime_corpus(
        regime_name=circuit_name,
        num_trials=1,
        num_neurons=num_neurons,
        bin_width_s=bin_width_s,
        trial_duration_s=total_duration_s,
    )
    sim = sims[0]
    raw_spike_counts = sim.spike_counts[:total_bins]  # [T, N]
    ground_truth_rates = sim.true_rates_hz[:total_bins]  # [T, N]

    # 2. Initialize causal streaming model
    if custom_model is not None:
        model = custom_model
    else:
        config = AttentionConfig(
            obs_dim=num_neurons,
            d_model=16,
            num_heads=2,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            bin_width_s=bin_width_s,
            max_seq_len=max(64, total_bins),
            causal=True,
        )
        model = SpatiotemporalAttentionModel(config, latent_dim=latent_dim)

    streamer = OnlineCausalStreamer(
        model=model,
        obs_dim=num_neurons,
        max_window_bins=50,
    )

    stimulator = ClosedLoopStimulator(
        rate_threshold_hz=rate_threshold_hz,
        phase_trigger_target=phase_trigger_target,
        min_inter_stim_interval_s=min_isi_s,
        blanking_duration_s=blanking_duration_s,
    )

    # 3. Step through incoming causal chunks
    latencies_ms: List[float] = []
    buffer_sizes: List[int] = []
    predicted_rates_list: List[np.ndarray] = []
    num_blanked_bins = 0
    num_chunks = int(np.ceil(total_bins / chunk_size_bins))

    for chunk_idx in range(num_chunks):
        start_bin = chunk_idx * chunk_size_bins
        end_bin = min(start_bin + chunk_size_bins, total_bins)
        curr_chunk_size = end_bin - start_bin

        chunk_spikes = raw_spike_counts[start_bin:end_bin].copy()

        # Check for sensor artifact blanking
        t_start = start_bin * bin_width_s
        for b in range(curr_chunk_size):
            t_bin = t_start + b * bin_width_s
            if stimulator.is_blanking_active(t_bin):
                chunk_spikes[b, :] = 0.0
                num_blanked_bins += 1

        # Push chunk through causal streaming engine
        chunk_th = torch.from_numpy(chunk_spikes).unsqueeze(0).float()
        res = streamer.push_chunk(chunk_th)

        latencies_ms.append(res.latency_ms)
        buffer_sizes.append(int(streamer.state.window_counts.shape[1]))

        rates_np = res.predicted_rates_hz.squeeze(0).cpu().numpy()
        latents_np = res.latents.squeeze(0).cpu().numpy()
        predicted_rates_list.append(rates_np)

        # Evaluate closed-loop triggers for each bin in chunk
        for b in range(curr_chunk_size):
            t_eval = t_start + b * bin_width_s
            latent_vec = latents_np[b]
            rate_vec = rates_np[b]
            stimulator.evaluate_step(
                current_time_s=t_eval,
                latent_state=latent_vec,
                predicted_rates_hz=rate_vec,
            )

    # 4. Synthesize diagnostic metrics
    lat_arr = np.array(latencies_ms, dtype=float)
    mean_lat = float(np.mean(lat_arr))
    p95_lat = float(np.percentile(lat_arr, 95))
    max_lat = float(np.max(lat_arr))
    violations = int(np.sum(lat_arr > deadline_ms))

    all_predicted_rates = np.concatenate(predicted_rates_list, axis=0)[:total_bins]
    rmse = float(np.sqrt(np.mean((all_predicted_rates - ground_truth_rates) ** 2)))

    ascii_summary = StreamingMonitorVisualizer.render_ascii(
        latencies_ms=latencies_ms,
        buffer_sizes=buffer_sizes,
        deadline_ms=deadline_ms,
    )

    svg_monitor = StreamingMonitorVisualizer.render_svg(
        latencies_ms=latencies_ms,
        buffer_sizes=buffer_sizes,
        deadline_ms=deadline_ms,
        title=f"Streaming Simulation: {circuit_name} ({num_neurons} channels)",
    )

    return StreamingSimulationResult(
        total_chunks=num_chunks,
        total_bins=total_bins,
        mean_latency_ms=mean_lat,
        p95_latency_ms=p95_lat,
        max_latency_ms=max_lat,
        deadline_violations=violations,
        num_stim_events=len(stimulator.event_history),
        num_blanked_bins=num_blanked_bins,
        reconstruction_rmse=rmse,
        ascii_summary=ascii_summary,
        svg_monitor=svg_monitor,
        stim_events=list(stimulator.event_history),
    )
