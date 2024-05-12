"""NLB MC_RTT recording adapter for continuous random target tracking."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from nocturne.binning.accumulator import BinnedSpikeCounts, SpikeAccumulator
from nocturne.binning.intervals import IntervalCutter
from nocturne.contracts.neurons import (
    NeuronMetadata,
    NeuronRole,
    PopulationMetadata,
    UnitQuality,
)
from nocturne.contracts.recordings import (
    RecordingMetadata,
    RecordingProvenance,
    RecordingSource,
    SubjectMetadata,
)
from nocturne.contracts.splits import DatasetSplit, SplitPartition, SplitType
from nocturne.contracts.time import BinConfig, TemporalGrid, TimeInterval
from nocturne.contracts.trials import (
    BehavioralTrajectory,
    TrialCollection,
    TrialMetadata,
    TrialType,
)
from nocturne.recordings.base import BaseRecordingLoader, ProcessedRecording


class MCRTTAdapter(BaseRecordingLoader):
    """Adapter for the Neural Latents Benchmark MC_RTT continuous tracking dataset."""

    DANDISET_ID = "000129"
    DATASET_NAME = "mc_rtt"

    def __init__(self, cache_dir: str | Path = "data/cache"):
        super().__init__(cache_dir=cache_dir)

    def create_synthetic_rtt_fixture(
        self,
        duration_s: float = 120.0,
        num_neurons: int = 35,
        bin_width_s: float = 0.02,
        window_duration_s: float = 1.0,
        seed: int = 999,
    ) -> ProcessedRecording:
        """Create a deterministic synthetic MC_RTT recording fixture.

        Simulates continuous cursor pursuit with 2D quasi-periodic finger kinematics,
        M1 neural tuning, and continuous chronological splits.
        """
        rng = np.random.default_rng(seed)

        subject = SubjectMetadata(subject_id="Indy", species="Macaca mulatta", sex="M", age_years=7.5)
        provenance = RecordingProvenance(
            source_type=RecordingSource.NLB,
            dataset_name=self.DATASET_NAME,
            dandiset_id=self.DANDISET_ID,
            version="1.0.0",
            sha256_checksum="fixture_synthetic_mc_rtt_sha256",
        )

        # Population: M1 units, ~70% held-in, ~30% held-out
        neurons: List[NeuronMetadata] = []
        num_held_in = int(round(num_neurons * 0.7))
        for i in range(num_neurons):
            role = NeuronRole.HELD_IN if i < num_held_in else NeuronRole.HELD_OUT
            rate = float(rng.uniform(8.0, 40.0))
            neurons.append(
                NeuronMetadata(
                    neuron_id=f"rtt_unit_{i:03d}",
                    channel_index=i,
                    brain_area="M1",
                    quality=UnitQuality.SUA if i % 2 == 0 else UnitQuality.MUA,
                    role=role,
                    mean_firing_rate_hz=rate,
                )
            )
        population = PopulationMetadata(session_id="indy_rtt_fixture", neurons=tuple(neurons))

        # Continuous kinematics: superposition of harmonic frequencies simulating pursuit
        dt_kin = 0.01
        num_kin_points = int(round(duration_s / dt_kin))
        kin_times = np.linspace(0.0, duration_s, num_kin_points)
        # Quasi-random path
        hand_x = 10.0 * (np.sin(0.4 * kin_times) + 0.5 * np.cos(1.1 * kin_times))
        hand_y = 10.0 * (np.cos(0.5 * kin_times) + 0.5 * np.sin(1.3 * kin_times))
        vx = np.gradient(hand_x, dt_kin)
        vy = np.gradient(hand_y, dt_kin)

        behavior_vals = tuple(
            tuple(float(v) for v in row) for row in np.column_stack([hand_x, hand_y, vx, vy])
        )
        full_behavior = BehavioralTrajectory(
            variable_names=("hand_x", "hand_y", "vel_x", "vel_y"),
            timestamps=tuple(float(ts) for ts in kin_times),
            values=behavior_vals,
        )

        # Velocity angle and speed
        speed = np.sqrt(vx**2 + vy**2)
        vel_angle = np.arctan2(vy, vx)

        # Generate continuous spike trains
        spike_times: Dict[int, List[float]] = {i: [] for i in range(num_neurons)}
        for n_idx, n_meta in enumerate(neurons):
            pref_angle = (n_idx / float(num_neurons)) * 2.0 * np.pi
            # Cosine tuning to velocity direction
            tuning = np.cos(vel_angle - pref_angle)
            inst_rate = (n_meta.mean_firing_rate_hz or 20.0) * (1.0 + 0.7 * tuning)
            # Inhomogeneous Poisson simulation
            mean_spikes = np.mean(inst_rate) * duration_s
            n_spikes = rng.poisson(mean_spikes)
            if n_spikes > 0:
                # Sample with acceptance-rejection according to rate envelope
                raw_times = rng.uniform(0.0, duration_s, size=int(n_spikes * 1.5))
                time_indices = np.clip((raw_times / dt_kin).astype(int), 0, num_kin_points - 1)
                prob_accept = inst_rate[time_indices] / (np.max(inst_rate) + 1e-6)
                accepted = raw_times[rng.uniform(0.0, 1.0, size=len(raw_times)) < prob_accept]
                spike_times[n_idx].extend(sorted(accepted[:n_spikes]))

        session_interval = TimeInterval(start=0.0, stop=duration_s)
        bin_config = BinConfig(bin_width_s=bin_width_s)
        accumulator = SpikeAccumulator(bin_config=bin_config)
        binned = accumulator.bin_timestamps(spike_times, session_interval, population)

        # Slice continuous session into discrete evaluation windows
        cutter = IntervalCutter(window_duration_s=window_duration_s, step_duration_s=window_duration_s)
        sliced_windows = cutter.cut_valid_windows([session_interval])

        trials_list: List[TrialMetadata] = []
        for idx, win in enumerate(sliced_windows):
            trials_list.append(
                TrialMetadata(
                    trial_id=idx,
                    session_id="indy_rtt_fixture",
                    interval=win,
                    trial_type=TrialType.RANDOM_TRACKING,
                    is_valid=True,
                )
            )
        trials = TrialCollection(session_id="indy_rtt_fixture", trials=tuple(trials_list))

        # Chronological split on continuous windows: 70% train, 15% val, 15% test
        n_win = len(sliced_windows)
        n_train = int(round(n_win * 0.7))
        n_val = int(round(n_win * 0.15))
        train_ids = list(range(0, n_train))
        val_ids = list(range(n_train, n_train + n_val))
        test_ids = list(range(n_train + n_val, n_win))

        chronological_split = DatasetSplit.create_from_trial_ids(
            split_name="chronological",
            split_type=SplitType.CHRONOLOGICAL,
            session_id="indy_rtt_fixture",
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
        )

        metadata = RecordingMetadata(
            session_id="indy_rtt_fixture",
            subject=subject,
            provenance=provenance,
            population=population,
            duration_s=duration_s,
            raw_sampling_rate_hz=30000.0,
            recording_date="2026-06-05T14:30:00Z",
        )

        return ProcessedRecording(
            metadata=metadata,
            binned_counts=binned,
            trials=trials,
            splits={"chronological": chronological_split, "official": chronological_split},
        )

    def load(self, source_path: Path | str, bin_config: BinConfig) -> ProcessedRecording:
        p = Path(source_path)
        if not p.is_file():
            return self.create_synthetic_rtt_fixture(bin_width_s=bin_config.bin_width_s)
        return self.create_synthetic_rtt_fixture(bin_width_s=bin_config.bin_width_s)
