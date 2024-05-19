"""Batch generator simulating controlled synthetic neural population dynamics across four regimes."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from nocturne.binning.accumulator import BinnedSpikeCounts, SpikeAccumulator
from nocturne.contracts.neurons import PopulationMetadata
from nocturne.contracts.recordings import (
    RecordingMetadata,
    RecordingProvenance,
    RecordingSource,
    SubjectMetadata,
)
from nocturne.contracts.splits import DatasetSplit, SplitType
from nocturne.contracts.time import BinConfig, TemporalGrid, TimeInterval
from nocturne.contracts.trials import TrialCollection, TrialMetadata, TrialType
from nocturne.recordings.base import ProcessedRecording
from nocturne.synthetic.base import CircuitSimulation, SyntheticCircuit
from nocturne.synthetic.bistable_memory import BistableMemoryCircuit
from nocturne.synthetic.driven_recurrent import DrivenRecurrentCircuit
from nocturne.synthetic.emissions import SyntheticEmissionModel
from nocturne.synthetic.linear_oscillator import LinearDampedOscillator
from nocturne.synthetic.noise import MisspecificationCorruptor, NoiseRegimeConfig
from nocturne.synthetic.nonlinear_oscillator import NonlinearLimitCycleOscillator


class SyntheticDatasetGenerator:
    """Generates large-scale synthetic benchmark corpora across four distinct dynamical regimes."""

    SUPPORTED_REGIMES = [
        "linear_oscillator",
        "nonlinear_oscillator",
        "bistable_memory",
        "driven_recurrent",
    ]

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def create_circuit(self, regime_name: str) -> SyntheticCircuit:
        """Instantiate canonical synthetic dynamical circuit for a regime."""
        if regime_name == "linear_oscillator":
            return LinearDampedOscillator(damping_gamma=0.5, frequency_omega=2.0 * math.pi)
        elif regime_name == "nonlinear_oscillator":
            return NonlinearLimitCycleOscillator(mu=1.0, omega=2.0 * math.pi, b=0.5)
        elif regime_name == "bistable_memory":
            return BistableMemoryCircuit(alpha=2.0, decay2=1.0)
        elif regime_name == "driven_recurrent":
            return DrivenRecurrentCircuit(latent_dim=4, input_dim=2, tau=0.05, seed=self.seed)
        else:
            raise ValueError(f"Unknown regime: {regime_name}. Supported: {self.SUPPORTED_REGIMES}")

    def generate_trial(
        self,
        circuit: SyntheticCircuit,
        emission_model: SyntheticEmissionModel,
        trial_id: int,
        duration_s: float = 1.0,
        bin_width_s: float = 0.02,
        corruptor: Optional[MisspecificationCorruptor] = None,
    ) -> CircuitSimulation:
        """Simulate a single trial with ground-truth trajectory, rates, and spikes."""
        dt_sim = 0.001
        latent_dim = circuit.latent_dim

        # Initial state distribution z0
        if circuit.regime_name == "linear_oscillator":
            # Random initial displacement away from origin
            r = self.rng.uniform(1.0, 2.5)
            theta = self.rng.uniform(0.0, 2.0 * math.pi)
            z0 = np.array([r * math.cos(theta), r * math.sin(theta)])
            input_fn = None
        elif circuit.regime_name == "nonlinear_oscillator":
            # Initial condition near or inside limit cycle
            r = self.rng.uniform(0.2, 1.8)
            theta = self.rng.uniform(0.0, 2.0 * math.pi)
            z0 = np.array([r * math.cos(theta), r * math.sin(theta)])
            input_fn = None
        elif circuit.regime_name == "bistable_memory":
            # Start near one of the attractors, apply pulse at t=0.4s
            start_in_A = self.rng.choice([True, False])
            base_z1 = 1.0 if start_in_A else -1.0
            z0 = np.array([base_z1 + self.rng.normal(0.0, 0.1), self.rng.normal(0.0, 0.1)])

            # Switching pulse
            pulse_dir = -3.5 if start_in_A else 3.5

            def input_fn(t: float) -> np.ndarray:
                if 0.4 <= t <= 0.55:
                    return np.array([pulse_dir, 0.0])
                return np.array([0.0, 0.0])

        elif circuit.regime_name == "driven_recurrent":
            z0 = self.rng.normal(0.0, 0.5, size=(latent_dim,))
            step_val = self.rng.choice([-1.0, 1.0])

            def input_fn(t: float) -> np.ndarray:
                if 0.2 <= t <= 0.8:
                    return np.array([step_val, 0.5 * step_val])
                return np.array([0.0, 0.0])
        else:
            z0 = self.rng.normal(0.0, 1.0, size=(latent_dim,))
            input_fn = None

        latent = circuit.integrate_rk4(z0, duration_s=duration_s, dt_s=dt_sim, input_fn=input_fn)

        # Downsample continuous states to temporal bin centers
        bin_config = BinConfig(bin_width_s=bin_width_s)
        grid = TemporalGrid.from_interval(TimeInterval(start=0.0, stop=duration_s), bin_config)
        bin_centers = grid.bin_centers

        # Interpolate states to bin centers
        step_indices = np.clip((bin_centers / dt_sim).astype(int), 0, latent.num_timepoints - 1)
        binned_states = latent.states[step_indices]  # [num_bins, latent_dim]

        true_rates = emission_model.compute_rates_hz(binned_states)

        if corruptor is not None:
            true_rates = corruptor.corrupt_firing_rates(true_rates)

        spikes = emission_model.sample_spikes(true_rates, bin_width_s=bin_width_s, rng=self.rng)

        population = emission_model.create_population_metadata(session_id=f"sim_{circuit.regime_name}")

        return CircuitSimulation(
            regime_name=circuit.regime_name,
            trial_id=trial_id,
            latent=latent,
            true_rates_hz=true_rates,
            spike_counts=spikes,
            population=population,
            bin_config=bin_config,
        )

    def generate_regime_corpus(
        self,
        regime_name: str,
        num_trials: int = 5000,
        num_neurons: int = 30,
        bin_width_s: float = 0.02,
        trial_duration_s: float = 1.0,
    ) -> Tuple[List[CircuitSimulation], ProcessedRecording]:
        """Generate full corpus of independent trials for a given regime, returning ProcessedRecording."""
        circuit = self.create_circuit(regime_name)
        emission = SyntheticEmissionModel(
            latent_dim=circuit.latent_dim,
            num_neurons=num_neurons,
            link_function="softplus",
            seed=self.seed,
        )

        sims: List[CircuitSimulation] = []
        trials_list: List[TrialMetadata] = []
        all_counts: List[np.ndarray] = []

        current_time = 0.0
        bin_config = BinConfig(bin_width_s=bin_width_s)
        bins_per_trial = int(round(trial_duration_s / bin_width_s))

        train_ids: List[int] = []
        val_ids: List[int] = []
        test_ids: List[int] = []

        for tid in range(num_trials):
            sim = self.generate_trial(
                circuit=circuit,
                emission_model=emission,
                trial_id=tid,
                duration_s=trial_duration_s,
                bin_width_s=bin_width_s,
            )
            sims.append(sim)

            start_t = current_time
            stop_t = current_time + trial_duration_s
            all_counts.append(sim.spike_counts)

            trials_list.append(
                TrialMetadata(
                    trial_id=tid,
                    session_id=f"corpus_{regime_name}",
                    interval=TimeInterval(start=start_t, stop=stop_t),
                    trial_type=TrialType.STANDARD,
                    condition_id=int(tid % 4),
                )
            )

            # Split (70% train, 15% val, 15% test)
            if tid < int(num_trials * 0.7):
                train_ids.append(tid)
            elif tid < int(num_trials * 0.85):
                val_ids.append(tid)
            else:
                test_ids.append(tid)

            current_time = stop_t

        combined_counts = np.vstack(all_counts)  # [total_bins, num_neurons]
        total_interval = TimeInterval(start=0.0, stop=current_time)
        grid = TemporalGrid.from_interval(total_interval, bin_config)
        binned = BinnedSpikeCounts(grid=grid, population=sims[0].population, counts=combined_counts)

        split = DatasetSplit.create_from_trial_ids(
            split_name="official",
            split_type=SplitType.OFFICIAL_NLB,
            session_id=f"corpus_{regime_name}",
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
        )

        metadata = RecordingMetadata(
            session_id=f"corpus_{regime_name}",
            subject=SubjectMetadata(subject_id=f"synthetic_{regime_name}", species="DynamicalSystem"),
            provenance=RecordingProvenance(
                source_type=RecordingSource.SYNTHETIC,
                dataset_name=regime_name,
                version="1.0.0",
                sha256_checksum=f"synthetic_manifest_seed_{self.seed}",
            ),
            population=sims[0].population,
            duration_s=current_time,
            raw_sampling_rate_hz=1000.0,
        )

        trials = TrialCollection(session_id=f"corpus_{regime_name}", trials=tuple(trials_list))

        processed = ProcessedRecording(
            metadata=metadata,
            binned_counts=binned,
            trials=trials,
            splits={"official": split},
        )

        return sims, processed
