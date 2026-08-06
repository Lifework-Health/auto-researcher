# PR 8.3: Stabilise executor workspace roots and seed evaluation budgets

## Summary

- Materialise and validate configured hardened-executor workspace parents before creating operation children.
- Reject final symlinks, non-directories, and inaccessible parents with `hardened_executor_workspace_root_unavailable`.
- Retain the trusted parent while cleaning every per-operation child.
- Define candidate-evaluation budgets as seed-inclusive and reject mutation-enabled one-evaluation searches before execution.
- Prove the generation-zero baseline followed by one durable fake-production mutation with two evaluator and verifier calls and one model call.

## Root cause

Both hardened isolation verification and candidate preparation passed a missing configured parent directly to `tempfile.mkdtemp`, which raised a raw `FileNotFoundError`. Separately, the search budget treated `maximum_candidate_evaluations=1` as sufficient for mutation even though the deliberate generation-zero baseline consumes that sole evaluation.

## Safety and compatibility

The generation-zero seed remains validated, prepared, evaluated, and verified through the normal task path. It consumes no mutation-model call. The retained executor image, worker protocol, evaluator, verifier, graph topology, and container mount policy are unchanged. No live provider, Aura, patient, iCCA, or MRI work is included.

## Validation

- Focused workspace-root, budget, graph, replay, bridge, provenance, and reuse tests.
- Real retained-image Docker isolation and preparation gate.
- Complete installed suite and isolated dependency-absent suite.
- Focused Ruff, formatting, mypy, diff, and credential-pattern checks.
