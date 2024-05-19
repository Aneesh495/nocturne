"""Unit tests for synthetic emissions, noise regimes, and dataset generation."""

import numpy as np
import pytest

from nocturne.synthetic.emissions import SyntheticEmissionModel
from nocturne.synthetic.generator import SyntheticDatasetGenerator
from nocturne.synthetic.noise import MisspecificationCorruptor, NoiseRegimeConfig


def test_emission_model_rates_and_spikes():
    emission = SyntheticEmissionModel(
        latent_dim=2,
        num_neurons=10,
        link_function="softplus",
        seed=42,
    )

    states = np.array([[0.0, 0.0], [1.0, -1.0]])
    rates = emission.compute_rates_hz(states)
    assert rates.shape == (2, 10)
    assert np.all(rates > 0.0)

    # Sample Poisson spikes
    spikes = emission.sample_spikes(rates, bin_width_s=0.02)
    assert spikes.shape == (2, 10)
    assert np.issubdtype(spikes.dtype, np.integer)
    assert np.all(spikes >= 0)

    # Overdispersed negative binomial
    emission_nb = SyntheticEmissionModel(
        latent_dim=2,
        num_neurons=10,
        dispersion_theta=2.0,
        seed=42,
    )
    spikes_nb = emission_nb.sample_spikes(rates, bin_width_s=0.02)
    assert spikes_nb.shape == (2, 10)
    assert np.all(spikes_nb >= 0)


def test_noise_regime_dropout_and_jitter():
    config = NoiseRegimeConfig(
        dropout_fraction=0.4,
        hide_external_input=True,
        rate_jitter_sigma=0.2,
    )
    corruptor = MisspecificationCorruptor(config, seed=123)

    obs_idx, drop_idx = corruptor.apply_neuron_dropout(num_neurons=20)
    assert len(drop_idx) == 8  # 40% of 20 = 8
    assert len(obs_idx) == 12
    assert set(obs_idx).isdisjoint(set(drop_idx))

    # Input drive filtering
    dummy_input = np.ones((10, 2))
    assert corruptor.filter_external_inputs(dummy_input) is None

    # Multiplicative jitter
    rates = np.full((10, 5), 20.0)
    jittered = corruptor.corrupt_firing_rates(rates)
    assert not np.allclose(rates, jittered)
    assert np.all(jittered > 0.0)


def test_synthetic_dataset_generator_all_regimes():
    gen = SyntheticDatasetGenerator(seed=999)

    for regime in SyntheticDatasetGenerator.SUPPORTED_REGIMES:
        sims, processed = gen.generate_regime_corpus(
            regime_name=regime,
            num_trials=10,
            num_neurons=15,
            trial_duration_s=0.5,
            bin_width_s=0.02,
        )

        assert len(sims) == 10
        assert processed.num_neurons == 15
        assert processed.trials is not None
        assert processed.trials.num_trials == 10

        split = processed.get_split("official")
        split.verify_zero_overlap()
        assert split.train.num_trials == 7
        assert split.val.num_trials == 1
        assert split.test.num_trials == 2
