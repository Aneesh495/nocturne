"""Base electrophysiology recording loader, SHA-256 cryptographic verification, and caching."""

from __future__ import annotations

import abc
import hashlib
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from nocturne.binning.accumulator import BinnedSpikeCounts
from nocturne.contracts.neurons import PopulationMetadata
from nocturne.contracts.recordings import RecordingMetadata, RecordingProvenance, RecordingSource
from nocturne.contracts.splits import DatasetSplit
from nocturne.contracts.time import BinConfig, TimeInterval
from nocturne.contracts.trials import TrialCollection


class ProcessedRecording(BaseModel):
    """Complete in-memory processed recording containing metadata, binned counts, trials, and splits."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    metadata: RecordingMetadata = Field(..., description="Top-level recording session metadata")
    binned_counts: BinnedSpikeCounts = Field(..., description="Binned spike counts matrix [num_bins, num_neurons]")
    trials: Optional[TrialCollection] = Field(default=None, description="Trial collection if trial-structured")
    splits: Dict[str, DatasetSplit] = Field(default_factory=dict, description="Named dataset splits (e.g. official, chronological)")

    @property
    def session_id(self) -> str:
        return self.metadata.session_id

    @property
    def num_neurons(self) -> int:
        return self.metadata.population.num_neurons

    @property
    def num_bins(self) -> int:
        return self.binned_counts.num_bins

    def get_split(self, split_name: str) -> DatasetSplit:
        if split_name not in self.splits:
            raise KeyError(f"Split '{split_name}' not found in recording {self.session_id} (available: {list(self.splits.keys())})")
        return self.splits[split_name]


class BaseRecordingLoader(abc.ABC):
    """Abstract base recording loader with caching, integrity verification, and provenance tracking."""

    def __init__(self, cache_dir: str | Path = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_sha256(file_path: Path | str) -> str:
        """Compute SHA-256 hash of a file on disk."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                sha256.update(chunk)
        return sha256.hexdigest()

    def verify_file_checksum(self, file_path: Path | str, expected_checksum: str) -> bool:
        """Verify that the file on disk matches the expected SHA-256 hash."""
        computed = self.compute_sha256(file_path)
        return computed.lower() == expected_checksum.lower()

    @abc.abstractmethod
    def load(self, source_path: Path | str, bin_config: BinConfig) -> ProcessedRecording:
        """Load, parse, bin, and partition an electrophysiological recording."""
        pass
