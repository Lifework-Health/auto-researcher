# Auto Researcher v2.1 architecture

## Core and plugin boundary

The core platform owns research contracts, hypotheses, planning, search
routing, lifecycle, budgets, execution orchestration, structural verification,
approval, checkpointing, provenance, and safe artefact references. It sees only
generic contracts such as `ExperimentSpec`, `EvaluationResult`, and
`VerificationResult`.

A `ResearchTask` plugin owns its identity and version, strict scientific
configuration, readiness checks, evaluator construction, experiment metadata,
dataset manifest, artefact policy, metrics, constraints, verification policy,
and scientific execution. Runtime paths live in `TaskRuntimeContext`; that
object never enters `ResearchState`.

The instance-scoped `TaskRegistry` stores factories keyed by `(task_id,
task_version)`. Duplicate versions fail, explicit version lookup is
deterministic, and an absent task produces a typed error. Both built-ins are
registered without importing `harness`; iCCA reports an actionable readiness
failure if its optional package or data files are absent.

```mermaid
flowchart TD
    graph["LangGraph control plane<br/>unchanged across domains"]
    registry["Task registry<br/>task ID + version"]
    runtime["Generic runtime assembly"]
    synthetic["Synthetic task<br/>implemented"]
    icca["iCCA NBS task<br/>implemented adapter"]
    mri["MRI segmentation task<br/>future"]

    registry --> synthetic
    registry --> icca
    registry -. extension .-> mri
    registry --> runtime
    runtime --> graph
    synthetic --> runtime
    icca --> runtime
    mri -. same protocol .-> runtime
```

## Unchanged LangGraph control plane

`build_graph()` is called identically for every task. Runtime assembly injects
the task evaluator, task verification policy, task metadata, and configuration
normaliser. No graph node branches on `task_id`, and cross-task integration tests
compare both the compiled Mermaid topology and executed node sequence.

The normal DIRECT path is:

`initialise → prepare → hypothesis → plan → approval route → search route →
DIRECT → evaluate → verify → provenance → decide`.

The full edge diagram remains in [graph.mmd](graph.mmd).

## Evaluation and verification

`DirectSearchBackend` selects one deterministic value from each planned search
dimension, delegates validation and canonicalisation to the active task, and
stamps the resulting experiment with task-supplied evaluator, code, dataset,
and provenance metadata. It has no scientific parameter knowledge.

Verification has two ordered layers:

1. The core structural verifier checks identity, evaluator/verifier
   registration, successful result presence, required metrics, score
   reconciliation, and provenance.
2. The task `VerificationPolicy` interprets task-owned constraints and recommends
   an evidence status.

A policy cannot bypass structural failure or promote MOCK/SIMULATED evidence to
`SUPPORTED`.

## Reference tasks

The synthetic task has its own bounded model configuration, deterministic score
landscape, simulated dataset manifest, generic metrics, and policy. It is the
offline CI default.

The iCCA NBS task lazily imports the installed `auto_agent_v2` package through
one bindings module. That repository remains the sole owner of cohort loading,
propagation, consensus clustering, PAC, survival and clinical analysis,
eligibility, numerical guards, and the stability objective. The adapter
requests exactly one configured K and maps nested scientific output into generic
`EvaluationResult.metrics`.

The future MRI segmentation plugin would implement the same task protocol and
supply a different evaluator, configuration, manifests, metrics, constraints,
and policy. It requires no graph change; see
[MRI_SEGMENTATION_EXAMPLE.md](../task_plugins/MRI_SEGMENTATION_EXAMPLE.md).

## Data custody and persistence

iCCA manifests contain only source filenames, sizes, SHA256 hashes, a combined
fingerprint, loader/objective versions, and a timestamp. They contain no patient
identifiers, values, credentials, or absolute paths.

Task evaluators atomically write only:

```text
runs/<run_id>/<experiment_id>/
├── experiment_spec.json
├── evaluation_result.json
├── dataset_manifest.json
└── evaluator_manifest.json
```

The graph checkpointer stores resumable execution state by `thread_id`.
Scientific provenance is a separate append-only store keyed by `run_id`.
Neither store receives `TaskRuntimeContext`.

## Generic Optuna ask/tell search

An optional `OptunaCapableTask` supplies an immutable study specification. The
task owns maximum parameter ranges, fixed scientific context, objective metric,
direction and search-space version. A planner may only narrow that structure;
widening ranges, adding choices or parameters, or mutating distribution
semantics is rejected before execution.

LangGraph owns every lifecycle transition:

`prepare → ask → create experiment → evaluate → verify → tell → record → decide`.

There is no `study.optimize()` call and no task-specific optimisation loop.
Every trial therefore uses the same evaluator, verifier, budget and checkpoint
nodes as DIRECT. A structurally valid finite result is told `COMPLETE`;
constraint violations remain complete but infeasible. Evaluation or structural
verification failures are told `FAIL` without fabricated penalty scores.

The selected research result is the best feasible trial. The best objective
value regardless of feasibility is retained separately as a diagnostic. If no
feasible trial exists, the primary experiment/evaluation/verification slots are
left empty and the diagnostic slots are explicit.

Study identity binds run, task and task version, objective and constraint
versions, evaluator, dataset, code version, request and normalised search-space
hash. Reopening validates every identity attribute. A durable running trial is
recovered by slot instead of asked twice, and repeated tell/provenance writes
must match their original content.

PR 3 local execution is sequential. LangGraph checkpoints, append-only
scientific provenance, and Optuna RDB state use three separate databases. A
future MRI plugin can expose learning rate, architecture or augmentation
parameters through the same capability without changing the graph.

## PR 3 non-goals

PR 3 does not implement live LLM calls, OpenEvolve, Neo4j, MRI training,
PyTorch/MONAI, distributed/parallel execution, patient-level artefacts, or
changes to the iCCA objective.
