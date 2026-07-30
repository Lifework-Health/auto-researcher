# ADR 004: Adapt the existing iCCA v2 evaluator

- Status: accepted
- Date: 2026-07-30
- Reference commit: `dab8c47ccdf3d5045ff5d9c76a6961b0dacd97cf`

## Context

`auto_agent_v2` already owns the validated iCCA scientific implementation.
Recreating its propagation, clustering, PAC, survival, clinical, gate, or
objective logic in v2.1 would create scientific divergence.

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

## Consequences

Scientific scoring has one owner. Synthetic mode remains installable and
executable without v2. iCCA readiness requires an editable/local v2 install and
the two expected data files. A compatibility test exercises real v2 dataclasses
and objective code without patient data; a separate opt-in gate compares real
data execution when explicitly configured.
