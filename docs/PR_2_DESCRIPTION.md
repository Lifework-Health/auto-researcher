# PR 2: General research task plugins and iCCA reference integration

## Summary

This PR separates the Auto Researcher control plane from scientific domain
implementations. It introduces a typed `ResearchTask` plugin contract,
deterministic instance-scoped registry, task-driven runtime assembly, two-layer
verification, safe task artefacts, a generic CLI, and two reference tasks:
offline synthetic and iCCA NBS.

The LangGraph topology is unchanged. Both tasks compile through the same
`build_graph()` and execute the same DIRECT node sequence.

## Core contract and runtime changes

- `ResearchContract` now carries `task_id`, `task_version`, `primary_metric`,
  and `task_constraints_version`.
- Immutable plugin models cover descriptors, readiness checks, experiment
  metadata, dataset manifests, artefact policies, runtime-only context, and
  policy decisions.
- `TaskRegistry` keys factories by task ID/version, rejects duplicates, reports
  unknown IDs/versions clearly, and deterministically lists and selects tasks.
- Generic memory/SQLite factories validate the contract and readiness before
  creating evaluator dependencies.
- Runtime paths and environment settings stay in `TaskRuntimeContext`; they are
  excluded from graph state and scientific provenance.

Both built-ins are advertised by the default registry so `auto-researcher tasks`
can report readiness. The iCCA factory itself has no eager v2 import; absence is
represented as an actionable readiness failure.

## DIRECT and verification

`DirectSearchBackend` now receives task-supplied `ExperimentMetadata` and a
configuration normaliser. It selects one deterministic configuration and
preserves evaluator, code, dataset, and provenance metadata without knowing any
scientific parameter names.

The core verifier performs structural checks first: identities, registrations,
result success/presence, task-required metrics, score reconciliation, and
provenance. Only then does it invoke a task `VerificationPolicy`. A policy cannot
bypass structural failure or promote MOCK/SIMULATED evidence to `SUPPORTED`.

## Synthetic reference task

The synthetic task owns a strict configuration (`model_family`, `complexity`,
`learning_rate`), deterministic landscape, generic metrics (`objective_score`,
`stability`, `runtime`), simulated dataset manifest, artefact policy, and
verification thresholds.

The documented demonstration completes one cycle with:

- primary score: `0.84`;
- final evidence: `INCONCLUSIVE (SIMULATED)`;
- one evaluator call and mandatory verifier call;
- five ordered provenance events;
- exactly four safe JSON artefacts.

The task policy's supported, refuted, and inconclusive paths are each tested.
Core verification intentionally downgrades the supported recommendation because
the evidence is simulated.

## iCCA NBS reference task

The iCCA plugin imports rather than copies `auto_agent_v2`. Direct `harness`
imports are isolated in `tasks/icca_nbs/bindings.py`, and synthetic mode never
imports that package.

Development and installed-contract testing used:

`auto_agent_v2@dab8c47ccdf3d5045ff5d9c76a6961b0dacd97cf`

Configuration mapping is strict and canonical:

- network and alignment resolve through the imported v2 enums;
- alpha uses imported `ALPHA_BOUNDS`;
- K uses imported `K_BOUNDS`;
- r must be positive;
- extra fields and runtime paths are forbidden.

For one DIRECT configuration the adapter reuses the v2 cohort, harness paths,
and propagation cache; propagates with network/alignment/alpha; calls the v2
evaluator with `k_values=[requested_K]`; retrieves exactly that K; and invokes
the imported `stability_objective`.

Generic output mapping preserves primary score, stability, complete nested
scientific metrics, selection inputs, eligibility, per-cluster aggregates,
canonical configuration, r, and objective version. Genuine `logrank_pass`,
`clinical_pass`, and `floors_pass` gates become generic constraint results;
diagnostic fields do not.

## Data custody and artefacts

iCCA manifests contain the two expected filenames, sizes, SHA256 hashes,
combined fingerprint, loader/objective versions, and timestamp. They contain no
patient identifiers, values, credentials, or absolute paths.

Each evaluator atomically writes:

```text
runs/<run_id>/<experiment_id>/
├── experiment_spec.json
├── evaluation_result.json
├── dataset_manifest.json
└── evaluator_manifest.json
```

Artefact references are stable and relative. Path traversal identifiers are
rejected. iCCA failure artefacts contain a structured exception type rather than
raw exception text that could reveal runtime paths or data.

## Proof of generality and tests

Cross-task integration builds the graph once with synthetic dependencies and
once with fake iCCA dependencies. It asserts:

- identical compiled Mermaid topology;
- identical executed node sequence;
- different task configurations, evaluator IDs, and policy IDs;
- no task-specific branch or task ID in `build_graph()`.

Final local result:

```text
60 passed, 1 skipped
```

The installed-v2 test passed using real v2 `EvaluationResult`, NumPy values, and
the real imported objective. It verifies objective, full metric JSON conversion,
and eligibility constraint mapping.

The only skipped test is the optional real-data compatibility comparison because
`AUTO_RESEARCHER_ICCA_DATA_DIR` and
`AUTO_RESEARCHER_ICCA_WORKSPACE_DIR` were not supplied. That gate remains
pending; no successful patient-data run is claimed.

## Provenance sequence

A normal DIRECT experiment records:

1. `HYPOTHESIS_PROPOSED`
2. `SEARCH_PLANNED`
3. `EXPERIMENT_PREPARED`
4. `EVALUATION_OBSERVED`
5. `EVIDENCE_VERIFIED`

The evaluation event includes safe relative artefact references.

## MRI extension path

The MRI documentation shows how a future `mri_segmentation` task would own its
configuration, evaluator, metrics, leakage/reproducibility checks, manifests,
and artefacts. DIRECT, future OPTUNA, and future OPENEVOLVE opportunities are
described without registering the task or adding PyTorch/MONAI. No graph changes
would be required.

## Limitations and deviations

- Real-data iCCA equivalence is pending because no external patient data was
  supplied.
- The default registry advertises iCCA even when v2 is absent so readiness and
  installation guidance remain discoverable. Scientific dependencies are still
  lazy and synthetic execution remains independent of v2.
- PR 2 implements DIRECT only. It does not implement Optuna ask/tell, live LLM
  agents, OpenEvolve, Neo4j, MRI training, distributed execution, patient-level
  artefacts, or any change to the iCCA objective.

## Proposed PR 3

Add a generic resumable Optuna ask/tell search backend behind the existing
`SearchBackend` boundary, including budget accounting, checkpoint-safe trial
state, task-defined search spaces, deterministic offline tests, and iCCA
compatibility without changing scientific scoring or graph topology.
