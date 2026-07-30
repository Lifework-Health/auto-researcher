# ADR 008: Durable model-call reservations and explicit recovery

Status: accepted for PR 4.

## Context

A live provider request is a nondeterministic paid side effect. A process may
die after the provider accepted a request but before LangGraph checkpointed the
node result. Automatically repeating it risks duplicate cost and conflicting
proposals.

## Decision

Every logical call has a deterministic identity over run, cycle, role, prompt
version, context hash, schema fingerprint, provider and explicit model ID.
Before invoking the provider, the runtime appends a `RESERVED` record to a store
separate from the graph checkpointer. It later appends `COMPLETED` or `FAILED`;
records are never updated or deleted.

An identical completed call is reused. A discovered started reservation without
an outcome becomes `INDETERMINATE` and fails closed. Only
`agent-calls retry --call-id ...` may append a new authorised attempt linked to
the original call. The original history is preserved.

## Consequences

Graph replay cannot silently duplicate a paid call. Operators must adjudicate
uncertain calls, and all call states can be inspected without storing rendered
prompts, credentials or hidden reasoning. Model-call provenance events use
record-derived deterministic IDs, so replay is idempotent.
