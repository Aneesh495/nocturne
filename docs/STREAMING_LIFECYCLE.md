# Nocturne Online Causal Streaming Architecture and Lifecycle

## 1. Real-Time Streaming Requirements

Closed-loop brain-computer interfaces (BCIs) and neuroprosthetic decoders require online state inference where predictions are emitted synchronously with incoming neural spike packets.

### 1.1 Timing and Budget Specifications

- **Bin Duration**: Standard temporal binning $\Delta t = 20\text{ ms}$ (50 Hz streaming frequency).
- **Processing Deadline**: Per-chunk computational latency must remain strictly below the bin duration ($t_{\text{latency}} < 20\text{ ms}$), with a target p95 latency under 5 ms on Apple Silicon / GPU targets to ensure ample headroom for downstream decoder actuation.
- **Causality Guarantee**: Future information must have zero influence on present or past state estimates.

---

## 2. Bounded Ring Buffer Mechanics

Naively accumulating neural counts across a multi-hour recording session leads to linear memory growth $\mathcal{O}(T_{\text{session}} \cdot N \cdot d_{\text{model}})$, which eventually exhausts resident memory and slows quadratic attention calculations.

Nocturne employs a fixed-capacity sliding ring buffer:
1. `max_window_bins`: Maximum temporal history retained (typically $W = 50$ to $100$ bins, corresponding to 1 to 2 seconds of recent neural activity).
2. When incoming chunk $\Delta T$ causes buffer length to exceed $W$, oldest bins are rolled off:

$$W_{t + \Delta t} = W_t[\Delta T:] \cup \text{chunk}_{\text{new}}$$

3. Memory consumption remains strictly bounded at $\mathcal{O}(W \cdot N \cdot d_{\text{model}})$ regardless of session length.

---

## 3. Streaming Lifecycle State Machine

A streaming session transitions across explicit states:

```
[INITIALIZING] --> [STREAMING] <--> [FAULT_RECOVERING]
                         |
                         V
                   [TERMINATED]
```

1. **INITIALIZING**: Buffer memory allocated, model set to eval mode, causal attention masks precomputed.
2. **STREAMING**: Successive chunks ingested, rates and latents predicted, wall-clock latencies profiled.
3. **FAULT_RECOVERING**: Triggered by network packet drops, timestamp gaps, or non-finite inputs. The handler fills missing intervals with neutral counts or sanitizes corrupted tokens without session restart.
4. **TERMINATED**: Session concluded, final diagnostic latency distribution and throughput metrics emitted.

---

## 4. Atomic Stream Checkpointing

In mission-critical BCI experiments, host system restarts or communication hiccups must not corrupt decoding continuity:
- `streamer.checkpoint()` serializes active buffer tokens, cumulative step counter, and latent state.
- `streamer.restore(state)` reconstitutes the identical internal state on a fresh stream instance, enabling instant resumption.
