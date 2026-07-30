# ADR 001: LangGraph is the execution control plane

- Status: Accepted
- Date: 2026-07-30

## Context

Auto Researcher needs explicit lifecycle control, durable state, deterministic
routing, human pauses, and future scientific component substitution. Scientific
judgement and deterministic truth must remain separate.

## Decision

Represent every major execution step as a LangGraph node and every control
decision as typed deterministic routing. LLM-shaped agents are limited to
hypothesis and plan proposals. Evaluators, verifiers, budgets, approval, routing,
and provenance remain deterministic dependencies or nodes.

## Alternatives considered

### Extend the existing manual loop

This would preserve working v2 code, but would keep pause/resume, branch visibility,
and lifecycle persistence coupled to bespoke orchestration. v2 remains available
behind adapters instead.

### Use a generic multi-agent chat framework

Chat-centric message histories are a poor fit for compact scientific state,
deterministic truth gates, and explicit resumable control flow.

### Make every component an LLM agent

This would allow proposers to influence measurement, verification, budget, and
audit rules. Those constraints must be enforced by schemas and deterministic code.

## Consequences

The graph is inspectable and resumable, and agents cannot bypass verification.
LangGraph becomes a core dependency; nodes must remain replay-safe and state must
stay compact.
