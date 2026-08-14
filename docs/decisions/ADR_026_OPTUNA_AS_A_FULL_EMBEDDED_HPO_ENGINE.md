# ADR 026 — Optuna as a full embedded HPO engine

Status: Accepted for PR 11.7

Date: 2026-08-14

## Context

Auto Researcher owns the outer scientific programme: Research Intelligence and
Research State, hypotheses, approved search envelopes, scientific evaluation and
verification, evidence, budgets, approvals, Managed Secrets, ResourceBroker, and
programme-level continuation. Optuna 4.9.0 is the pinned inner HPO engine. It is
authoritative for suggestions, samplers, pruners, study/trial state, native
constraints, and Pareto semantics.

The earlier integration intentionally established public `Study.ask()` /
`Study.tell()`, sequential SQLite, coordinated PostgreSQL workers, fencing,
heartbeats, ResourceBroker leases, and evaluation crash recovery. It retained an
unnecessary v1 restriction to static single-objective TPE studies.

The executable audit is
`docs/capabilities/optuna-4.9.0-capability-manifest.yaml`. It binds the exact
Optuna distribution RECORD and PostgreSQL HPO lock hash. Pin or public-inventory
changes require manifest and probe review.

## Decision

### Authority and lifecycle

The standard integration remains public ask/tell. Auto Researcher does not use
`Study.optimize()` because it must own the evaluation boundary, and it does not
implement a sampler, Pareto ranking, constrained ranking, pruning algorithm, or
proposal queue. A task-owned projection turns verified metrics into an ordered,
finite objective vector and an ordered constraint vector. Optuna consumes those
vectors through its public APIs.

### Versioned study contract

`OptunaStudySpec` v2 additively defines objectives, typed conditions, sampler,
pruner, constraints, intermediate reporting, and diagnostics. Legacy `direction`
and `objective_metric` deterministically represent one objective. A v1 spec still
constructs explicit TPE with its historical startup count, uses fixed
distributions, and retains the byte-compatible study identity payload.

The v2 identity binds sampler/pruner options, objectives, constraints, search
space semantics, projection, and diagnostics. Resume fails closed if that
immutable envelope changes. Worker/session identity and physical resource
placement remain operational and do not enter scientific identity.

### Samplers and pruners

The approved native sampler registry exposes Optuna's material algorithms:
native default, TPE, Random, CMA-ES, GP, NSGA-II, NSGA-III, QMC, Grid, and brute
force. Compatibility is validated before study creation. Optional CMA-ES, GP,
QMC, and MDI dependencies have bounded extras and are imported lazily. Known
unsupported objective/constraint/shared-worker combinations fail rather than
falling back to TPE.

The pruner registry exposes native default, Nop, Median, Patient, Percentile,
Successive Halving, Hyperband, Threshold, and Wilcoxon. Runtime code may register
an approved `BaseSampler` or `BasePruner` factory under a logical name. YAML can
never contain an arbitrary import path.

### Search spaces and objectives

V1 static spaces continue through `Study.ask(fixed_distributions=...)`. V2 uses
public `Trial.suggest_categorical`, `suggest_int`, and `suggest_float` inside a
typed task-approved envelope. Conditions are equality branches over an earlier
declared parameter; there is no `eval` or configuration-supplied code.

Multiple objectives create one native study with `directions=[...]`. Telling uses
the native ordered vector. Results expose `Study.best_trials` and never invent a
scalar winner. A later outer-programme decision may choose a Pareto point.

### Native constraints

Scientific constraint meaning and sign projection belong to the task. The
adapter converts `<= threshold` and `>= threshold` to Optuna's convention:
positive is violated and zero or negative is feasible. A valid infeasible
evaluation remains `COMPLETE`; invalid scientific results are `FAIL`.

Supported native sampler `constraints_func` callbacks read an immutable record
keyed by study and trial. The implementation uses unique public study user attrs,
works with SQLite and RDBStorage, and does not mirror `TrialState` or mutate
private Optuna tables. The record is written before `tell`, so a crash cannot
silently discard a completed projection required by a later worker.

### Intermediate reports and pruning

A task may optionally implement the intermediate-reporting evaluator protocol.
The narrow reporter calls native `Trial.report()` and `Trial.should_prune()` and
persists each finite step/value plus the prune decision. Evaluation stops only at
a cooperative safe checkpoint. Auto Researcher never forcibly kills arbitrary
in-process Python.

After cooperative acknowledgement, public `Study.tell(state=PRUNED)` records the
terminal state without a fabricated final objective. Worker and resource leases
are released. Optuna 4.9 does not support multi-objective `report`/`should_prune`,
so v2 rejects multi-objective pruning rather than scalarizing.

Optuna 4.9 also has no public API to reconstruct a live reporting `Trial` after
process loss. A persisted prune decision is finalized as `PRUNED`; otherwise a
detached interrupted trial becomes `FAIL` and a later native ask replaces it.
This is explicitly classified as weakened, and sampler RNG state is likewise not
durably stored by upstream. Distributed outcomes are schedule-dependent.

### PostgreSQL and resources

PR 11.5 coordination remains the only shared-worker lifecycle. It atomically
admits native asks under the outer trial budget, fences owner tells, and maintains
worker and ResourceBroker heartbeats. PostgreSQL RDBStorage remains authoritative
for Optuna state. ResourceBroker decides where a whole candidate runs; no sampler
can suggest `gpu:2`, and placement is absent from trial parameters and scientific
identity. Native non-TPE, objective-vector/Pareto, and constraint visibility are
tested with real PostgreSQL and psycopg.

### Scientific identity and reuse

An Optuna trial number is not a scientific experiment identity. V2 derives the
experiment identity from task-normalized scientific configuration and evaluator,
dataset, and code versions. Equivalent trials remain visible to Optuna while the
existing evaluation-reuse-v2 authority revalidates the complete evidence bundle
before avoiding expensive work. Reused objective and constraint values are then
told to the new native trial; no weaker HPO cache is introduced.

### Diagnostics and epistemic boundary

Typed diagnostics include sampler/pruner identity, trial-state counts, a native
single-objective best trial, a native multi-objective Pareto set, intermediate
summaries, and supported default/fANOVA/MDI/PED-ANOVA importance evaluators.
Their fixed status is `OPERATIONAL_SEARCH_DIAGNOSTIC`.

Parameter importance describes association with objective variation inside the
sampled HPO study. It does not establish causality, biological importance,
mechanism, or external validity. Diagnostics and Pareto fronts do not become
EvidenceCards or Research State claims automatically.

## Consequences

- Existing v1 TPE/SQLite/PostgreSQL studies retain their identities and behavior.
- New studies can use native alternate samplers, conditions, constraints,
  pruning, and multiple objectives through the ordinary graph start/resume path.
- Optional algorithms remain absent from core startup and fail with a bounded
  install-extra message when dependencies are missing.
- PartialFixedSampler, arbitrary imports/callbacks, Journal/gRPC storage,
  Optuna-owned scientific artefacts, and visualization/dashboard deployment are
  not claimed as integrated capability; their classifications and reasons remain
  executable in the manifest.
- The checked-in bounded FeTA SegResNet acceptance template exercises v2 without
  paid models, mutable architecture search, or the prospective U-Net campaign.
