# Research task plugin development

1. Implement the runtime-checkable `ResearchTask` protocol under
   `auto_researcher/tasks/<task_id>/`.
2. Define a strict, frozen Pydantic configuration model with forbidden extras.
3. Return a stable `TaskDescriptor`; choose an evaluator ID, policy ID, schema
   version, and supported search types.
4. Validate `ResearchContract` task/version, evaluator, primary metric, and
   supported search types.
5. Implement readiness without scientific side effects. Report missing packages,
   files, or settings as actionable checks.
6. Produce task-owned `ExperimentMetadata` and a safe `DatasetManifest`. Never
   include credentials, raw records, or absolute paths.
7. Implement the generic `Evaluator` boundary. Pass scientific values through
   the finite JSON normaliser with a task-owned exact allowlist; never apply a
   global non-finite-to-null rule. A primary score must be finite and constraint
   results must be explicit booleans.
8. Publish the four-file experiment bundle through the transactional bundle
   writer. Return intended references only after publication; persistence
   failure results must use an empty reference tuple.
9. Implement `VerificationPolicy`. Declare required metrics and interpret only
   task constraints; structural verification remains in core.
10. Register a factory in a registry. Do not use a global mutable singleton.
11. Test configuration, readiness, evaluator mapping, policy outcomes, JSON
    safety, artefacts, provenance, and the complete shared graph.

## Adding an OpenEvolve surface

Only tasks that implement the runtime-checkable `OpenEvolveCapableTask` protocol
may advertise `OPENEVOLVE`. Return an `EvolvableComponent` whose versioned spec
declares exactly one safe relative mutable file, one entry point, its immutable
signature, input/output schemas, allowed imports/dependencies, source-size cap,
seed source, and bounded mutation context. Convert a validated preparation result
to the task's ordinary `ExperimentSpec`; do not implement a second evaluator or
verifier path.

Keep datasets, split construction, evaluator/verifier code and identities,
objectives, eligibility gates, contracts, budgets, and framework files outside
the mutable surface. Bind the task's finite OpenEvolve configuration to its
actual evaluator and verifier identities. Test seed preparation, invalid source,
candidate conversion, negative scientific results, replay, and artefact tamper
handling. A future MRI plugin can expose one bounded loss, augmentation,
thresholding, or model-block file through this interface, while training and
data handling remain task-owned and immutable.

Budget the immutable seed as a normal candidate evaluation. A serial
population-one run with one evolved generation needs two experiment slots: one
for the generation-zero baseline and one for the evolved candidate. The seed
uses zero mutation-model calls. Keep `ResearchContract.maximum_experiments`,
`SearchRequest.experiment_budget`, and the resulting
`maximum_candidate_evaluations` consistent.

For hardened execution, plugins declare finite CPU, memory, process, timeout,
output, log, workspace-byte, individual-file and concurrent-entry limits. They
must not assume `/tmp`, HOME or a host output mount is writable. Candidate
results return through the fixed supervisor protocol; plugins receive only a
validated `candidate-preparation-v2`. The trusted local fixture runner applies
the same entry-count semantics after execution but is never live-eligible.

Acceptance requires comparing the new task with the synthetic task using the same
`build_graph()`: topology and executed node sequence must remain identical. Do
not add task-ID conditionals to graph code.
