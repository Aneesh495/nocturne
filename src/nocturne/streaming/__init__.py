"""Online causal streaming engine, bounded buffers, equivalence checks, and benchmarks."""

from __future__ import annotations

from nocturne.streaming.state import StreamingBufferState
from nocturne.streaming.streamer import (
    StreamChunkResult,
    OnlineCausalStreamer,
)
from nocturne.streaming.equivalence import (
    EquivalenceReport,
    StreamingEquivalenceAuditor,
)
from nocturne.streaming.fault_tolerance import (
    FaultToleranceReport,
    StreamFaultHandler,
)
from nocturne.streaming.benchmarks import (
    StreamingBenchmarkMetrics,
    StreamingBenchmarkRunner,
)
from nocturne.streaming.closed_loop import (
    ClosedLoopStimulator,
    StimulationEvent,
)

__all__ = [
    "StreamingBufferState",
    "StreamChunkResult",
    "OnlineCausalStreamer",
    "EquivalenceReport",
    "StreamingEquivalenceAuditor",
    "FaultToleranceReport",
    "StreamFaultHandler",
    "StreamingBenchmarkMetrics",
    "StreamingBenchmarkRunner",
    "ClosedLoopStimulator",
    "StimulationEvent",
]
