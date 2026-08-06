# Durable model-call state machine

`RESERVED → DISPATCHING → COMPLETED` is the successful path. SQLite uses
`BEGIN IMMEDIATE`, semantic-key uniqueness, and compare-and-append transitions,
so one call reservation and one dispatch owner exist across processes.

Terminal alternatives are `FAILED_BEFORE_DISPATCH`, `FAILED_CONFIRMED`,
`OUTCOME_UNKNOWN`, and `REJECTED`. Existing v1 `FAILED` and `INDETERMINATE`
records remain valid for hypothesis/planner compatibility. Completed reuse is
read-only and consumes no additional call or cost. `DISPATCHING` and
`OUTCOME_UNKNOWN` are never automatically dispatched again.

Maximum cost is reserved once. Completion records actual token usage and cost.
Failures after possible invocation remain conservatively charged; known
pre-dispatch failures are excluded from subsequent budget reservation totals.
