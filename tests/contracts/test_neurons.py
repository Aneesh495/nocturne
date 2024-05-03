"""Unit tests for neuron and population metadata contracts."""

import pytest

from nocturne.contracts.neurons import (
    NeuronMetadata,
    NeuronRole,
    PopulationMetadata,
    UnitQuality,
)


def test_neuron_metadata_creation():
    neuron = NeuronMetadata(
        neuron_id="unit_001",
        channel_index=0,
        brain_area="M1",
        quality=UnitQuality.SUA,
        role=NeuronRole.PRIMARY_OBSERVED,
        coordinates_um=(120.0, 340.0, 50.0),
        mean_firing_rate_hz=14.2,
        snr=4.5,
    )
    assert neuron.neuron_id == "unit_001"
    assert neuron.channel_index == 0
    assert neuron.brain_area == "M1"
    assert neuron.quality == UnitQuality.SUA
    assert neuron.role == NeuronRole.PRIMARY_OBSERVED
    assert neuron.coordinates_um == (120.0, 340.0, 50.0)


def test_population_metadata_methods():
    n0 = NeuronMetadata(neuron_id="n0", channel_index=0, brain_area="M1", role=NeuronRole.HELD_IN)
    n1 = NeuronMetadata(neuron_id="n1", channel_index=1, brain_area="M1", role=NeuronRole.HELD_OUT)
    n2 = NeuronMetadata(neuron_id="n2", channel_index=2, brain_area="PMd", role=NeuronRole.HELD_IN)
    n3 = NeuronMetadata(neuron_id="n3", channel_index=3, brain_area="PMd", role=NeuronRole.HELD_OUT)

    pop = PopulationMetadata(session_id="session_test", neurons=(n0, n1, n2, n3))

    assert pop.num_neurons == 4
    assert pop.channel_indices == [0, 1, 2, 3]
    assert pop.neuron_ids == ["n0", "n1", "n2", "n3"]

    assert pop.held_in_indices == [0, 2]
    assert pop.held_out_indices == [1, 3]

    assert pop.get_indices_by_area("M1") == [0, 1]
    assert pop.get_indices_by_area("PMd") == [2, 3]

    assert pop.get_by_id("n2").channel_index == 2
    with pytest.raises(KeyError):
        pop.get_by_id("n999")


def test_population_metadata_duplicates():
    n0 = NeuronMetadata(neuron_id="n0", channel_index=0, brain_area="M1")
    n0_dup_id = NeuronMetadata(neuron_id="n0", channel_index=1, brain_area="M1")
    with pytest.raises(ValueError, match="Duplicate neuron_id"):
        PopulationMetadata(session_id="s1", neurons=(n0, n0_dup_id))

    n1_dup_chan = NeuronMetadata(neuron_id="n1", channel_index=0, brain_area="M1")
    with pytest.raises(ValueError, match="Duplicate channel_index"):
        PopulationMetadata(session_id="s1", neurons=(n0, n1_dup_chan))


def test_population_filter_by_indices():
    n0 = NeuronMetadata(neuron_id="n0", channel_index=0, brain_area="M1")
    n1 = NeuronMetadata(neuron_id="n1", channel_index=1, brain_area="M1")
    n2 = NeuronMetadata(neuron_id="n2", channel_index=2, brain_area="PMd")

    pop = PopulationMetadata(session_id="s1", neurons=(n0, n1, n2))
    sub = pop.filter_by_indices([0, 2])
    assert sub.num_neurons == 2
    assert sub.channel_indices == [0, 2]
    assert sub.neuron_ids == ["n0", "n2"]
