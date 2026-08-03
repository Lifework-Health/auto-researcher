# PR 5.4: Bind evaluation reuse to artefact identity

The final pre-OpenEvolve preflight found two interface defects before any live
work began: evaluation reuse proved only that the current bundle was internally
valid, not that it was the original completed bundle, and START conflict error
names differed from the checkpoint contract.

This corrective PR introduces `evaluation-reuse-v2`. A reusable record is
committed only after a successful evaluator result, complete transactional
publication, strict four-file verification, and reference validation. It stores
the experiment and result hashes, evaluator/dataset/code versions, original
bundle hash, `experiment-bundle-v2` schema, `scientific-json-v1` encoding,
expected references, evaluator-manifest payload hash, and completion timestamp.
Replay requires exact equality. A different but internally valid recomputed
bundle fails with `artefact_bundle_identity_conflict`; missing, partial,
tampered, schema-incompatible, encoding-incompatible, failed, and unpublished
results are non-reusable and never cause evaluator repair.

Verification reuse remains bound to its existing evaluation payload, verifier,
and policy identities while referencing the authoritative evaluation-reuse
record. Legacy `evaluation-reuse-v1` rows are explicitly non-reusable and are
not silently populated from current files. Checkpoint 03 remains unsupported.

The public START rejection vocabulary is versioned as
`run-execution-errors-v1` and uses exactly:

- `conflicting_run_identity`
- `conflicting_contract_identity`
- `conflicting_task_identity`
- `conflicting_initial_input_identity`
- `thread_already_exists_use_resume_or_inspect`

CLI preflight preserves those codes before external dependency construction.
All validation for this PR is offline. Anthropic, Aura, genuine patient data,
and OpenEvolve are not invoked.
