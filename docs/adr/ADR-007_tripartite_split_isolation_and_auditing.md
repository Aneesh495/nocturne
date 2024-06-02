# ADR-007: Tripartite Split Isolation and Chronological Auditing

## Status
Accepted

## Context
Neural dynamics models can easily memorize trial sequences or exploit autocorrelation leakage across continuous time if training, validation, and test partitions share overlapping temporal segments or if cross-validation splits are randomly shuffled across time. In non-stationary motor tasks (e.g. MC_RTT continuous tracking) and perturbation paradigms (Area2_Bump), random trial shuffling permits future behavioral state information to leak into earlier predictions.

## Decision
1. **Strict Tripartite Partitioning**: Dataset partitions are represented by immutable `SplitPartition` records (`src/nocturne/contracts/splits.py`) strictly segregating `train`, `val`, and `test` intervals.
2. **Chronological Group Enforcement**: For continuous tracking sessions, partitions enforce chronological blocks: all training trials precede validation trials, and all validation trials precede test trials.
3. **Automated Zero-Overlap Auditing**: `SplitAuditor` (`src/nocturne/recordings/split_audit.py`) and `EvaluationLabelIntegrityAuditor` (`src/nocturne/evaluation/integrity.py`) compute interval intersections across all declared trial pairs:
   $$[t_{\text{start}, i}, t_{\text{end}, i}) \cap [t_{\text{start}, j}, t_{\text{end}, j}) = \emptyset \quad \forall i \in \text{Split}_A, j \in \text{Split}_B$$
   Any non-empty intersection immediately raises a fatal `SplitOverlapError`.
4. **Cryptographic Fingerprinting**: Each partition emits a canonical SHA-256 digest of its trial timestamps, spike event hashes, and behavioral tensors, logged to `EVALUATION_FINGERPRINTS.json`.

## Consequences
- Guaranteed zero information leakage between training, tuning, and evaluation datasets.
- Reliable out-of-sample generalization metrics that reflect true online decoding performance.
- Automated audit catches temporal overlap, split inversion, or duplicate trials during pipeline initialization.
