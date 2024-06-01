# ADR-004: Bounded Streaming Ring Buffer and Fault Recovery

## Status
Accepted

## Context
Online brain-machine interface (BMI) decoding and closed-loop neural stimulation require streaming causal inference with hard real-time latency deadlines (e.g., $< 10$ ms per bin). Naively appending incoming spike counts to an unbounded in-memory list causes linear memory growth $\mathcal{O}(T_{\text{stream}})$ and quadratic computation latency $\mathcal{O}(T_{\text{stream}}^2)$ over time. In a multi-hour session, unbounded accumulation triggers garbage collection spikes, out-of-memory errors, and severe latency degradation.
Furthermore, physical telemetry streams suffer from packet drops, variable chunk sizes, transmission jitter, and unexpected socket disconnections.

## Decision
1. **Bounded Circular Ring Buffer**: Implement `StreamingBufferState` (`src/nocturne/streaming/state.py`) with a strictly enforced maximum capacity $W_{\max} = 128$ bins. As new chunks arrive, oldest bins beyond $W_{\max}$ are evicted in $O(1)$ time, guaranteeing constant memory usage regardless of streaming duration.
2. **Chunk-Size Invariance**: Incoming spike packets of arbitrary chunk sizes $\Delta c \in [1, 32]$ are normalized into uniform discrete bins. `OnlineCausalStreamer` (`src/nocturne/streaming/streamer.py`) guarantees mathematical output equivalence across varying chunk arrival schedules.
3. **State Checkpointing and Rollback**: Bounded buffer states support lightweight serialized snapshots (`save_checkpoint()` / `restore_checkpoint()`). If an invalid payload (NaNs, negative counts, mismatched channel dimension) is detected, `StreamFaultHandler` (`src/nocturne/streaming/fault_tolerance.py`) immediately rolls back to the last valid verified state without crashing the host process.
4. **Gap Imputation**: When timestamps indicate dropped packets, missing intervals are imputed with empirical marginal mean firing rates or zeros accompanied by explicit observation mask invalidation ($M_{\text{obs}} = \text{False}$).

## Consequences
- Bounded $\mathcal{O}(1)$ memory consumption and sub-millisecond per-chunk inference latency verified over continuous multi-thousand bin streams.
- Robust fault tolerance against network drops, malformed data, and unexpected process interruptions.
- Clean separation between streaming buffer state management and deep neural network inference engines.
