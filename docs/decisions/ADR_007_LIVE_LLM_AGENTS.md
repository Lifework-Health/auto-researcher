# ADR 007: Bounded live hypothesis and planning agents

Status: accepted for PR 4.

## Context

Hypothesis generation and experiment planning benefit from model proposals, but
the research contract, search limits, evaluator, verifier and lifecycle cannot
be delegated to a nondeterministic model.

## Decision

The existing hypothesis and planning nodes may use injected live agents. Each
agent makes one bounded structured request (with a small correction/retry
allowance) through `StructuredModelClient`. The response is an untrusted typed
`HypothesisProposal` or `PlannerProposal`. Deterministic reconcilers validate
references, budgets, task schemas, installed capabilities and approval
requirements, then construct platform-owned IDs and final contracts.

Mock implementations remain the default. No graph edge changes between modes.
The model never invokes evaluators, tools or routers and cannot set evidence
status, provenance, measured scores or the research contract.

## Consequences

Live creativity is bounded by typed contracts and task-owned affordances.
Provider code stays outside graph nodes, and tasks remain domain plugins.
Invalid output fails closed after bounded charged retries.

## Alternatives

- Keeping all agents deterministic would not provide the requested live mode.
- A full ReAct graph would delegate deterministic control and side effects.
- Evaluator tools exposed to the model would blur proposal and measurement.
- Free-text parsing would be less strict and replayable than Pydantic output.
- Task-specific live agents without a shared provider boundary would couple
  provider plumbing to scientific domains.
