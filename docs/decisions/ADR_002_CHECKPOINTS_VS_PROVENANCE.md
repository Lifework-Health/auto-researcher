# ADR 002: Execution checkpoints and scientific provenance are separate

- Status: Accepted
- Date: 2026-07-30

## Context

Execution recovery answers “where can this computation resume?” Scientific
provenance answers “what was proposed, run, measured, verified, and concluded?”
They have different retention, query, and integrity requirements.

## Decision

Use a LangGraph checkpointer keyed by `thread_id` for executable graph state. Use
a separate append-only provenance protocol keyed by `run_id` for immutable typed
`DecisionEvent` records. PR 1 supplies SQLite implementations and refuses to use
the same file for both.

## Consequences

Graph implementations and checkpoint backends can change without rewriting the
scientific audit trail. Provenance remains queryable independently of checkpoint
retention. Two stores and identifiers must be managed explicitly, which the
runtime dependency factory centralises.
