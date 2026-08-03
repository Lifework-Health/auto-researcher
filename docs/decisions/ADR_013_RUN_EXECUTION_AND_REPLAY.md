# ADR 013: Explicit execution modes and semantic replay safety

## Decision

Expose `START`, `RESUME`, and `REPLAY_INSPECT` through `run-execution-v2`.
Persist a canonical execution identity in graph state. A new input dictionary
is accepted only by `START` on a checkpoint-free thread; continuation uses
`None` or `Command(resume=...)`; terminal replay is read-only inspection.

Lifecycle provenance uses `provenance-events-v2` semantic identities rather
than timestamps or actors. The provenance store returns an identical existing
event and rejects a changed scientific payload at the same identity.

`evaluation-reuse-v2` stores only successful, completed evaluation snapshots
within a run after the four-file bundle has been published and verified. The
record binds the immutable experiment and result to the original bundle hash,
bundle schema, result encoding, exact references, evaluator-manifest payload
hash, and completion timestamp. Internal validity is necessary but
insufficient: a different valid bundle at the same scientific identity is a
conflict.

Verification reuse remains bound to the evaluation payload, verifier version,
and task policy version, and also references the authoritative evaluation-reuse
record identity. It does not maintain a second independent bundle identity.
`evaluation-reuse-v1` rows are explicitly legacy and non-reusable; current
artefacts are never used to infer missing historical identity fields.

## Consequences

An operator cannot accidentally execute a terminal run by submitting its
initial payload again through the supported runtime or CLI. Inspection has no
graph or scientific side effects. Direct graph callers still receive
defence-in-depth deduplication, while conflicts fail closed. Cross-run result
reuse is deliberately excluded. Public START identity conflicts use
`run-execution-errors-v1`: `conflicting_run_identity`,
`conflicting_contract_identity`, `conflicting_task_identity`, and
`conflicting_initial_input_identity`.
