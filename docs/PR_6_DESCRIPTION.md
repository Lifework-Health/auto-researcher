# PR 6: Generic, bounded and resumable OpenEvolve search

## Scope and baseline

Base: `main` at `78fe2b4385aa2fecd2b54bda5f4d818fd05113e9` (equal to `origin/main` before branching).

This PR adds `OPENEVOLVE` behind the existing search and LangGraph lifecycle. It contains no live OpenEvolve run, Claude/Anthropic call, Aura query/write, patient data, genuine iCCA evaluation, MRI implementation, dependency installation, or distributed execution.

## Architecture and contracts

1. The generic backend registry now exposes DIRECT, Optuna, and OpenEvolve capabilities without task-ID routing.
2. `openevolve-search-v1` contains search/task/component/seed identities; finite population, generation, candidate, time, model, failure, and artefact limits; mutation, selection, replacement, sandbox, resume, and stopping policies; evaluator/verifier identities; seed; objective direction; and optional threshold.
3. `OpenEvolveCapableTask` supplies the task-owned `EvolvableComponent`. Its spec declares a single mutable Python file, full-file seed/replacements, immutable entry-point contract, schemas, allowed imports/dependencies, size cap, and safe mutation context. The task maps validated output to its normal `ExperimentSpec`.
4. Evaluator/verifier code and identities, data and splits, contract/objective/gates, budgets/policies, graph/runtime/provenance/checkpoint/reuse code, and hidden fixtures remain immutable.
5. `openevolve-candidate-v1` binds normalized source, request, immutable interface, dependency manifest, sandbox, parents, generation/birth, operator/model-call metadata, validation/preparation, evaluation identity, and safe status. Domain-separated canonical hashes make identical source in one immutable context one candidate identity.
6. `openevolve-lineage-v1` records ordered ancestry and outcomes. `openevolve-population-v1` persists active/archive/evaluated/failed identities, outcomes, lineage, diversity, deterministic random state, budgets, reservations, and stopping state.
7. Mutation is either deterministic offline replacement or a structured fake-model full-file replacement. The model may only propose the declared source; deterministic code controls validation, reservations, identity, preparation, scientific execution, ranking, replacement, budgets, and stopping.

## Safety, lifecycle, and evidence

8. Static validation fails closed on malformed/oversize source, paths/interfaces/hashes, forbidden imports, network/process/filesystem primitives, reflection/dynamic execution, recursive entry calls, unbounded `while`, classes, and input/attribute mutation.
9. `openevolve-sandbox-v1` uses a fixed worker, isolated interpreter flags, minimal environment, private read-only input/source plus writable output, parent timeout, sanitized bounded logs, bounded output/file count, cleanup, and platform resource limits. It does not invoke a candidate-selected shell or command.
10. Residual risk: this local runner is not a kernel sandbox and does not provide namespace-enforced network isolation. Live/untrusted mutation requires a hardened no-network container or micro-VM executor before approval.
11. Constrained deterministic ranking orders constraint compliance, verification, objective direction, then candidate ID. Bounded elitist replacement preserves the archive. Diversity uses source-hash uniqueness; duplicates are recorded and not reevaluated.
12. Budgets count reservations/proposals, successful/failed preparations, evaluations, verifier calls, model calls, runtime/wall time, failures, and artefact bytes. Stop rules cover objective, generation/evaluation/model/time/artefact limits, failure limits, and invalid population.
13. Checkpoint boundaries cover initialization, generation/parent selection, mutation reservation/completion, validation, preparation, existing evaluation/verification, population update, and finalization. Resume uses persisted reservation/random state.
14. Candidate preparation reuses only validated output; evaluation and verification continue to use `evaluation-reuse-v2` and `verification-reuse-v1`. Population updates and semantic provenance are idempotent; terminal INSPECT is read-only; duplicate START and terminal RESUME retain `run-execution-v2` guards.
15. `openevolve-provenance-v1` adds initialized, generation started/completed, parent selected, mutation reserved, candidate proposed/rejected/prepared/evaluated/verified, population updated, and search stopped events with deterministic semantic keys.
16. Search publication stages and atomically renames a bundle containing request, manifest, population, candidate index/sources/results, lineage, budget, and stopping JSON. Payload hashes and an aggregate bundle identity fail closed on missing/partial/tampered data; references remain safe and relative.

## Offline validation

17. Synthetic demonstration: linear 0.78 → tree 0.84 → neural 0.88; 3 evaluated, 0 failed, objective reached, feasible candidate present, zero model calls, transactional search bundle published.
18. Fake cell-biology-shaped demonstration: a bounded pathway aggregation over synthetic non-patient signals produced 0.88 and passed the unchanged synthetic evaluator/verifier path.
19. Interrupted execution and uninterrupted execution produced identical population, search result, final evaluation, and final verification hashes.
20. A new process reconstructed typed OpenEvolve state under multiple `PYTHONHASHSEED` values with an identical final-state hash. Checkpoint hash and mtime remained unchanged during INSPECT.
21. Repeated terminal INSPECT returned identical state with zero writes. Duplicate START returned `thread_already_exists_use_resume_or_inspect`; terminal RESUME returned `thread_is_terminal_use_inspect`.
22. Hostile fixtures cover process/shell, socket/HTTP, dynamic import/eval/exec, absolute/traversal/home/repository access, environment reads, framework imports/monkey-patching, mutable input, recursion/infinite loop, excessive logs, file/process creation, and introspection. They are rejected statically or contained with safe codes; timeout cleanup and traceback/path sanitization pass.
23. Candidate/interface/population identities are stable across process hash seeds. Duplicate source is not reevaluated. Artefact tampering is rejected.
24. Full offline suite: `353 passed, 2 skipped` in 9.03 seconds. Skips are the explicitly opted-out paid Anthropic smoke test and genuine iCCA patient-data gate; no tests were deselected.
25. Ruff on all changed/new Python files: pass. Ruff format check on all changed/new Python files: pass.
26. Repository-wide pre-existing findings: three unused imports in `tasks/models.py` and `tests/unit/test_task_framework.py`; 20 unrelated pre-existing files would be reformatted. They are not changed here.

## Known limitations and next scope

27. Population evaluation is sequential; parallel/distributed execution is deferred. The mutable representation is intentionally one Python file and full-source replacement. No production model mutation adapter is enabled. Local resource-limit portability varies by OS, and the local runner is approved only for trusted offline fixtures.
28. Recommended PR 7: introduce an optional adapter to the pinned OpenEvolve library behind these contracts, retain explicit approval and call budgets, and add a hardened independently tested no-network executor before any live-model candidate execution. Keep scientific task/evaluator work out of that infrastructure PR.

## Review guide

Start with `docs/architecture/OPENEVOLVE_SEARCH.md`, the two new ADRs, `search/openevolve/models.py`, `search/openevolve/backend.py`, and the explicit OpenEvolve graph nodes. Then inspect the synthetic component, integration tests, hostile-source tests, and resume/threat-model runbooks.
