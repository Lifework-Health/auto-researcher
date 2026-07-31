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

Acceptance requires comparing the new task with the synthetic task using the same
`build_graph()`: topology and executed node sequence must remain identical. Do
not add task-ID conditionals to graph code.
