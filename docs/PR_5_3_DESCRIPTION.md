# PR 5.3: Terminal-run replay safety and semantic provenance idempotency

Checkpoint 03 proved the complete live scientific path but reproduced an
operator error: submitting fresh input to the same terminal LangGraph thread.
This corrective PR formalises execution semantics and prevents that operation
before any node runs.

The runtime now provides `start_run`, `resume_run`, and
`inspect_terminal_run`, backed by `run-execution-v2` identity validation. The
CLI exposes matching `run start`, `run resume`, and `run inspect` commands.

Lifecycle events use `provenance-events-v2` semantic keys. A persistent
`evaluation-reuse-v1` originally prevented repeated per-run evaluator or
verifier work and verified the current artefact bundle before returning a
result. Corrective PR 5.4 supersedes it with `evaluation-reuse-v2`, which also
binds reuse to the original bundle identity. This is defence in depth; the
terminal-thread guard remains primary.

All validation is offline. This PR does not change iCCA science, scientific
JSON, the artefact bundle, grounding, model prompts, or task configuration.
Anthropic, Aura, genuine patient data, and OpenEvolve are not invoked.

## Proposed final live replay-only checkpoint

Checkpoint 03 predates `run-execution-v2` and must not be mutated or silently
migrated. Once a terminal live checkpoint has been created through the new
protocol, run `run inspect` twice and compare its state, provenance rows, result
hashes, bundle hash, file contents, and modification times. Expected deltas are
zero Aura queries, Claude calls, evaluator/verifier calls, artefact writes,
checkpoint writes, and provenance events. Do not submit the initial payload.
