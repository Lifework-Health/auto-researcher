# PR 3: Generic resumable Optuna ask/tell search

## Summary

This PR adds an optional, domain-neutral Optuna 4 backend without changing the
scientific task boundary. LangGraph owns the iterative lifecycle and calls
`Study.ask` and `Study.tell`; it never calls `Study.optimize`. Synthetic and
iCCA NBS supply different study specifications, evaluators and policies to the
same graph.

## Architecture and topology

The OPTUNA route is:

`prepare study → ask trial → create ExperimentSpec → shared evaluate → shared
verify → tell trial → record trial provenance → decide → ask or finalise`.

The backend is a durable adapter over Optuna storage, while LangGraph remains
responsible for orchestration, checkpoints, budgets and verification. The
compact graph state contains only the current trial and study summary. Trial
payloads required to restore a selected result are durable Optuna user
attributes.

## Contracts and search-space policy

Immutable discriminated Pydantic contracts represent float, integer and
categorical parameters, optimisation direction, study specification, current
trial, compact state, outcome and final result. The runtime-checkable
`OptunaCapableTask` is optional.

Tasks register maximum ranges, fixed context, direction, metric and
search-space version. Planner input may narrow numeric bounds, select an
ordered categorical subset, or pin a sampled parameter. It cannot widen a
range, introduce or reorder choices, add parameter names, mutate type/log/step
semantics, or replace registered fixed scientific context.

## Identity, storage and reconstruction

The readable study name has a short SHA256 suffix over immutable identity
attributes. Identity binds run, task/version, objective/version,
constraints/version, evaluator, dataset, code, request, direction, seed, trial
budget and the normalised search-space hash. Every attribute is compared when a
study is loaded.

PR 3 uses sequential Optuna SQLite RDB storage. Runtime rejects a path collision
between Optuna, LangGraph checkpoint and provenance databases. A process can
reconstruct with the same thread ID and three databases. A tagged RUNNING trial
for the current slot is recovered; foreign, untagged or multiple running trials
fail rather than being guessed away.

Ask is idempotent by `(run, request, slot)`. Tell stores a canonical report
digest and accepts only an identical replay; conflicting reports fail. The
restart integration test interrupts after ask, closes every connection,
reconstructs runtime, completes three trials, and observes no duplicate trial
or provenance event.

## Budgets, trial states and selection

Effective trial budget is the minimum of the task/request study budget and
remaining global experiment budget. Every evaluator call, including failure,
consumes experiment and cost budget. Recovered asks do not.

A finite successful evaluation becomes `COMPLETE` only after structural
verification and identity/provenance/score reconciliation. A valid
constraint-violating trial remains `COMPLETE` but infeasible. Evaluation,
non-finite score or structural verification failures become `FAIL`; no penalty
score is fabricated.

The primary winner is the best feasible complete trial, with direction-aware
comparison and lowest-trial tie breaking. Best overall remains a separate
diagnostic. If no feasible trial exists, primary experiment/evaluation/
verification fields remain empty and explicitly named diagnostic fields hold
the best overall observation.

## Provenance and artefacts

Replay-safe deterministic events record the hypothesis and plan once, then
study start, proposal, experiment, evaluation, verification and report for each
trial, followed by study completion. Existing DIRECT provenance is unchanged.

Atomic aggregate artefacts are:

`study_spec.json`, `study_summary.json`, `trials_summary.json`, and
`selected_trial.json`.

They use stable relative references under
`runs/<run>/studies/<study>/`. Per-experiment evaluator artefacts retain their
existing location. Sensitive policies redact parameter/fixed configuration
payloads.

## Demonstrations

- Synthetic fixed-seed CLI study: 8 asked, 8 complete, 0 failed; best feasible
  trial 5 with score `0.75571`; verified SIMULATED evidence remained
  `INCONCLUSIVE`; all four study artefacts were written.
- Fake iCCA study: 3 asked, 3 complete, 0 failed; best feasible trial 0 with
  imported stability objective `0.8`; the fake reference evaluator and
  objective were each invoked three times. Network, alignment and `r` were
  fixed; alpha `[0.3, 0.9]` and K `[4, 8]` came from the injected v2 bindings.
- The optional installed-v2 contract test passed without patient data and
  confirmed the same imported alpha/K bounds.

## Versions and verification

- Optuna: `4.9.0` (`optuna>=4.9,<5` optional extra; exact transitive lock
  supplied).
- `auto_agent_v2` commit inspected:
  `dab8c47ccdf3d5045ff5d9c76a6961b0dacd97cf`.
- Core/default suite: 84 passed, 1 real-data test skipped, 15 HPO tests
  deselected.
- HPO suite: 15 passed, 85 non-HPO tests deselected.
- Combined suite: 99 passed, 1 skipped, 0 failed.
- The real iCCA patient-data Optuna trial is pending because explicit local
  patient-data environment variables were not supplied. No successful
  real-data run is claimed.

## Known limitations and proposed PR 4

PR 3 is sequential and single-objective. It does not implement pruning,
intermediate reports, parallel/distributed workers, PostgreSQL, cross-run study
reuse, production scheduling, live LLM agents, OpenEvolve, Neo4j or MRI
training.

Recommended PR 4 scope: replace the deterministic hypothesis/planner stand-ins
with bounded live LLM agents while preserving typed outputs, approval,
task-owned search spaces and the graph lifecycle. OpenEvolve and the other
deferred systems should remain separate follow-on work.
