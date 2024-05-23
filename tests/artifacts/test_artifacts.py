"""Tests for experimental artifact metadata and registry."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from pydantic import ValidationError

from nocturne.artifacts.schema import ArtifactMetadata, ArtifactRegistry


def test_artifact_metadata_immutability():
    meta = ArtifactMetadata(
        artifact_id="art_001",
        artifact_type="figure",
        dataset_name="mc_maze",
        split_name="val",
        model_name="vldm",
        file_path="figures/fig1.svg",
        metrics={"bps": 0.25},
    )
    assert meta.artifact_id == "art_001"
    assert meta.metrics["bps"] == 0.25

    with pytest.raises(ValidationError):
        meta.artifact_id = "modified"


def test_artifact_registry_lifecycle(tmp_path: Path):
    reg = ArtifactRegistry(tmp_path)
    assert reg.list_artifacts() == []

    meta1 = ArtifactMetadata(
        artifact_id="art_001",
        artifact_type="figure",
        dataset_name="mc_maze",
        split_name="val",
        model_name="vldm",
        file_path="figures/fig1.svg",
    )
    reg.register(meta1)

    listed = reg.list_artifacts()
    assert len(listed) == 1
    assert listed[0]["artifact_id"] == "art_001"

    retrieved = reg.get_artifact("art_001")
    assert retrieved is not None
    assert retrieved.artifact_id == "art_001"
    assert retrieved.dataset_name == "mc_maze"

    # Register updated metadata with identical ID
    meta1_updated = ArtifactMetadata(
        artifact_id="art_001",
        artifact_type="figure",
        dataset_name="mc_maze",
        split_name="val",
        model_name="vldm",
        file_path="figures/fig1_updated.svg",
    )
    reg.register(meta1_updated)
    assert len(reg.list_artifacts()) == 1
    assert reg.get_artifact("art_001").file_path == "figures/fig1_updated.svg"

    # Nonexistent ID lookup
    assert reg.get_artifact("nonexistent") is None
