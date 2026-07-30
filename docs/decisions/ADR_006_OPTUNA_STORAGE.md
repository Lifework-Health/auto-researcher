# ADR 006: Separate SQLite storage for sequential Optuna

Status: accepted for PR 3.

## Decision

Local Optuna studies use Optuna's SQLite RDB storage. Its file must differ from
the LangGraph checkpoint database and the append-only provenance database;
runtime assembly rejects any collision.

Only a safe filename is exposed outside runtime. Study identity user attributes
bind the task, objective, constraints, evaluator, dataset, code, request,
direction, seed and normalised search-space hash. All attributes are checked on
resume.

## Consequences

A new process can reconstruct the graph from its checkpoint and the Optuna
study from its own ledger. Only one running trial is permitted, tagged with run,
request and slot ownership. Untagged, foreign or multiple running trials fail
with an actionable recovery error.

PostgreSQL, parallel workers and distributed optimisation are deferred.
