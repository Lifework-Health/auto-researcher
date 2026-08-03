# ADR 013: Explicit execution modes and semantic replay safety

## Decision

Expose `START`, `RESUME`, and `REPLAY_INSPECT` through `run-execution-v2`.
Persist a canonical execution identity in graph state. A new input dictionary
is accepted only by `START` on a checkpoint-free thread; continuation uses
`None` or `Command(resume=...)`; terminal replay is read-only inspection.

Lifecycle provenance uses `provenance-events-v2` semantic identities rather
than timestamps or actors. The provenance store returns an identical existing
event and rejects a changed scientific payload at the same identity.

`evaluation-reuse-v1` stores completed evaluation and verification snapshots
within a run. Evaluation reuse requires the same immutable experiment and a
complete untampered artefact bundle. Verification reuse additionally requires
the same evaluation hash, verifier version, and task policy version.

## Consequences

An operator cannot accidentally execute a terminal run by submitting its
initial payload again through the supported runtime or CLI. Inspection has no
graph or scientific side effects. Direct graph callers still receive
defence-in-depth deduplication, while conflicts fail closed. Cross-run result
reuse is deliberately excluded.
