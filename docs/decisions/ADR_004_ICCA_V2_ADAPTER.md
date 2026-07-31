# ADR 004: Adapt the existing iCCA v2 evaluator

- Status: accepted
- Date: 2026-07-30
- Reference commit: `dab8c47ccdf3d5045ff5d9c76a6961b0dacd97cf`

## Context

`auto_agent_v2` already owns the validated iCCA scientific implementation.
Recreating its propagation, clustering, PAC, survival, clinical, gate, or
objective logic in v2.1 would create scientific divergence.

The pinned reference implementation resolves consensus resampling to `R=100`.
Its lightweight PAC test uses 10 resamples, while the implementation itself has
no stricter lower-bound validation. A synthetic inspection run also establishes
that `r=1` does not necessarily raise. Auto Researcher therefore owns an
executable minimum of 10 and retains 100 as the recommended production default;
this prevents scientifically meaningless stability claims without changing the
reference evaluator.

## Decision

The `icca_nbs` task lazily imports v2 only inside `bindings.py`. An injectable
`ICCABindings` groups the exact APIs and registered bounds required by the
adapter. The production loader binds to the installed `harness` package; tests
use a behaviorally shaped fake.

For a DIRECT experiment, the adapter loads/reuses the cohort, paths, and
propagation cache; resolves network and alignment through v2 enums; propagates;
evaluates exactly the requested K; and calls v2's imported
`stability_objective`. It maps outputs into generic nested metrics and genuine
eligibility gates into `constraint_results`.

The adapter also records a closed, evidence-safe failure stage and completion
flags. Where the reference API combines consensus and eligibility in one call,
trusted module identities from the in-memory traceback are reduced to a stage;
the traceback and raw exception message are discarded and never persisted.

The reference result model can explicitly return `NaN` for the secondary
`metrics.c_index.apparent`, `metrics.c_index.cv`, and
`metrics.c_index.incremental` readouts when Cox estimation is unavailable. The
adapter owns an exact allowlist for those mapped paths. It encodes an allowed
unavailable readout as JSON `null` and records its schema path under
`metric_availability`; it does not change or impute the reference calculation.
Any other non-finite metric fails closed at `RESULT_NORMALISATION`. A non-finite
stability objective fails earlier at `OBJECTIVE_CALCULATION`, and eligibility
constraints must be explicit booleans.

## Consequences

Scientific scoring has one owner. Synthetic mode remains installable and
executable without v2. iCCA readiness requires an editable/local v2 install and
the two expected data files. A compatibility test exercises real v2 dataclasses
and objective code without patient data; a separate opt-in gate compares real
data execution when explicitly configured.

The resampling policy changes the iCCA configuration schema to `1.1`, task
constraints to `1.0`, Optuna search-space identity to
`auto-agent-v2-icca-v2`, and adapter version to `icca-adapter-v1.2`. Experiment
code identity also includes `scientific-json-v1` and `experiment-bundle-v2`, so
old result encoding and partial-file publication cannot replay as the new
semantics.
