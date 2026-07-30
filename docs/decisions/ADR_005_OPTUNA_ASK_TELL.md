# ADR 005: LangGraph controls Optuna through ask and tell

Status: accepted for PR 3.

## Context

Auto Researcher must checkpoint between proposal, scientific execution,
verification and reporting. Different task plugins must use one control plane,
and a failed process must not silently repeat an experiment or report a trial
twice.

## Decision

LangGraph owns the optimisation lifecycle through explicit prepare, ask, create,
evaluate, verify, tell, record, decide and finalise nodes. Optuna is a durable
sampler and study ledger. The platform never calls `Study.optimize`.

Tasks optionally implement `OptunaCapableTask` and return a bounded immutable
study specification. The core contains no scientific parameter names. Task
configuration is normalised before an `ExperimentSpec` is evaluated, and every
result passes mandatory structural and task-policy verification before tell.

## Consequences

Each transition can be interrupted and resumed. Failed evaluations become
`FAIL`, valid constraint violations remain `COMPLETE` but infeasible, and
selection distinguishes the best feasible result from the best overall
diagnostic result. Sequential execution is deliberate in PR 3.

## Alternatives

- `study.optimize` inside one node would hide trial lifecycle and replay.
- Task-specific loops would couple the graph to scientific domains.
- An immediate Optuna HTTP service adds unnecessary deployment complexity.
- LLM-selected hyperparameters are neither the deterministic sampler nor
  scientifically authoritative search-space validation required here.
- Copying the old iCCA loop would duplicate scientific integration and retain a
  non-resumable control flow.
