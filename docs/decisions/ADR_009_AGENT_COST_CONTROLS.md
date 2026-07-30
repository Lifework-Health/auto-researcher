# ADR 009: Model usage and cost are first-class budgets

Status: accepted for PR 4.

## Context

Agent requests and retries consume tokens and money before scientific
evaluation begins. Ignoring that spend would make the research contract's cost
ceiling incomplete.

## Decision

Live mode requires explicit, versioned pricing plus provider/model/call limits.
`AgentBudgetPolicy` limits calls per role and cycle, attempts, input context,
output tokens, cost per logical call and total provider calls. The runtime
checks remaining budget before reservation and reconciles returned
input/output/cache usage afterward.

`BudgetState` tracks model and evaluator costs separately; `cost_used` is their
combined total and is governed by `ResearchContract.maximum_cost`. Retry usage
is accumulated, not hidden. Pricing is runtime configuration rather than a
permanent provider fact.

Token-only budgeting was considered but is not implemented in PR 4; explicit
positive pricing is required.

## Consequences

Runs are auditable across model and experiment spend. Pricing must be reviewed
by the operator and supplied with a version. Actual provider billing can still
differ from the recorded estimate, so estimates are not invoices.
