# ADR 012: Durable exact knowledge bundles

Status: accepted for PR 5.

## Context

A graph may change between two identical reads. A process can also stop after a
provider accepted a request but before LangGraph checkpoints the result.
Automatically querying again can silently change the evidence supplied to a
replayed model call.

## Decision

A deterministic retrieval ID binds run/cycle, task/version, contract, provider
and adapter versions, graph alias, configured schema/content versions, and the
query-plan hash. A separate append-only store records `RESERVED`, `COMPLETED`,
`FAILED` and `INDETERMINATE` snapshots.

The exact completed, validated `KnowledgeBundle` is stored with its bundle
hash and reused on replay; the provider is not called again. Five atomic JSON
artefacts preserve the request, plan, graph snapshot metadata, bundle and
validation summary. Only a compact reference enters checkpoint state.

A started reservation without an outcome becomes `INDETERMINATE`. The runtime
will not infer whether the external read completed. An operator must authorise
a linked child attempt with `knowledge retrievals retry`; original records are
never changed.

## Consequences

Replay is evidence-stable and provenance events are idempotent. Storage is
larger than re-querying, but it avoids treating a later graph state as the
original scientific context.
