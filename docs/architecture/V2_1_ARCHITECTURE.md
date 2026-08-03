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

`initialise → prepare → retrieve knowledge → hypothesis → plan → approval
route → search route → DIRECT → evaluate → verify → provenance → decide`.

The full edge diagram remains in [graph.mmd](graph.mmd).

## Bounded hypothesis and planning agents

`generate_hypothesis` and `plan_search` are the only nodes that may invoke a
live model. In default `mock` mode they remain deterministic and offline. Live
mode replaces their injected implementations, not the nodes or edges. The
supervisor, approval flow, search router, search backends, evaluator, verifier,
provenance writer and stop decisions remain deterministic.

A live agent receives an immutable `HypothesisAgentContext` or
`PlannerAgentContext`, renders a repository-versioned prompt, and asks a
provider-neutral `StructuredModelClient` for a Pydantic proposal. The untrusted
proposal cannot contain platform IDs, evidence status or provenance.
`HypothesisReconciler` and `PlannerReconciler` deterministically validate it and
construct the final `Hypothesis` or `SearchRequest`.

```mermaid
flowchart LR
    state["Compact graph state"] --> assembler["Deterministic context assembler"]
    task["Task-safe AgentContext"] --> assembler
    prior["Verified prior provenance only"] --> assembler
    assembler --> proposal["Structured model proposal"]
    prompts["Versioned prompt files"] --> proposal
    proposal --> reconcile["Deterministic reconciliation"]
    reconcile --> contracts["Hypothesis / SearchRequest"]
    contracts --> graph["Existing graph route"]
    proposal -. reserve + complete .-> calls[("Agent-call store")]
```

The provider boundary accepts an injected LangChain `BaseChatModel` and uses
its Pydantic structured-output interface. Anthropic support is an optional
factory; the core installation and default tests do not require a provider
package or credential.

## Context, grounding and authority

Tasks optionally implement `AgentContextCapableTask`. Synthetic and iCCA both
do so. Context includes safe vocabulary, metric semantics, constraints,
available search capabilities and bounded task-owned DIRECT/Optuna schemas.
iCCA exposes only aggregate metadata and registered parameter affordances; it
does not expose patient identifiers, raw mutation values, clinical rows,
credentials or absolute paths.

Prior memory is reconstructed from verified scientific provenance, sorted and
capped. Context history, artefact references and total characters are bounded,
and canonical JSON yields a stable context hash. The model never sees complete
graph state or raw provenance rows.

PR 5 can emit `KNOWLEDGE_GROUNDED` only when a proposal cites a qualifying
reference in the active, validated bundle. Merely retrieving a bundle does not
change grounding. Unknown, stale, cross-bundle, irrelevant, unverified or
uncited references fail reconciliation or remain ungrounded. Model confidence
is capped by the most conservative cited trust-tier cap and is never scientific
confidence.

## Deterministic knowledge retrieval

`retrieve_knowledge` is a deterministic control node between lifecycle
preparation and hypothesis generation. A task that implements
`KnowledgeGroundingCapableTask` supplies a bounded query plan and evidence
policy. The core validates that plan against a registry of fixed, versioned
templates, then calls a provider-neutral `KnowledgeProvider`. Agents cannot
query a provider, generate Cypher, widen limits or request extra graph content.

`DISABLED` performs no provider call. `OPTIONAL` records a safe failure and
continues honestly. `REQUIRED` stops before any paid model call when task
support, configuration, readiness, schema, retrieval or validation is
unavailable. The control graph never branches on iCCA, genes, Neo4j or a future
MRI domain.

The first external provider uses the official Neo4j driver with an explicit
database, connectivity/read-only checks, fixed parameterised Cypher, a
read-transaction API, timeouts, record caps, update-counter checks, schema
preflight and safe error codes. Credentials come only from environment
variables. Internal Neo4j IDs and raw driver objects cannot cross the provider
boundary. The static provider gives offline tests a deterministic
`SIMULATED` bundle; it is never labelled real external evidence.

Providers return entities, assertions, sources and compact references.
`KnowledgeBundleValidator` checks stable identifiers, source provenance,
provider/schema/content identity, task policy, trust tiers, confidence,
limits, privacy fields and deterministic hashes. Unverified or LLM-asserted
records may be diagnostic but cannot qualify as grounding evidence. Graph
grounding affects only a hypothesis prior; it cannot mark an experiment
`SUPPORTED`. Only evaluation and verification determine experimental evidence.

The exact completed bundle and five safe JSON artefacts are stored outside
checkpoint state. A deterministic retrieval identity binds run/cycle, task and
contract, provider/version, graph alias, schema/content configuration and
query-plan hash. Completed replay uses that bundle and never re-queries a
changing graph. A started reservation without an outcome becomes
`INDETERMINATE`; only an explicit linked CLI retry may issue another read.

## Model-call durability and budgets

Model calls are nondeterministic paid side effects. Their deterministic ID
binds run, cycle, role, prompt version, context hash, response schema, provider
and explicit model ID. A separate append-only agent-call store records
`RESERVED`, `COMPLETED`, `FAILED` and `INDETERMINATE` snapshots. Completed calls
are reused. A reservation found without an outcome is marked indeterminate and
cannot be repeated until the operator creates a linked retry through the CLI.

Retries are bounded and allowed only for transient provider errors, timeouts
and invalid structured output. Authentication, context-size and permanent
provider errors fail closed. Prompt hashes are stored, but rendered prompts,
credentials, provider reasoning and chain of thought are not.

`BudgetState` separately accumulates model calls, input/output/cache tokens,
model cost and evaluator cost. `cost_used` is their combined total and remains
subject to `ResearchContract.maximum_cost`. Live configuration requires
versioned explicit pricing and a maximum cost per call before any provider
request.

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

Before a successful result can enter graph state, its primary score must be
finite, constraints must be explicit booleans, and metrics must be strict JSON.
The generic scientific normaliser preserves finite values and rejects infinity
and undeclared NaN. A task may declare exact secondary diagnostic paths where
NaN means unavailable; those values become JSON `null` with explicit metric
availability metadata. Domain paths remain in the plugin and never enter the
control graph.

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

## Run execution and replay safety

`run-execution-v2` exposes a closed `START`, `RESUME`, and `REPLAY_INSPECT`
vocabulary. `START` accepts initial input only for a thread with no checkpoint.
`RESUME` continues a non-terminal thread with `None`, or with LangGraph's
`Command(resume=...)` for an explicit interrupt value. `REPLAY_INSPECT` reads a
terminal `StateSnapshot` and never invokes the graph.

Each checkpoint stores a stable execution identity containing thread ID, run
ID, contract/task identity, canonical contract hash, graph schema version, and
canonical initial-input hash. A conflicting caller fails before a graph node or
external dependency can run. Supplying a fresh dictionary to an existing
LangGraph thread is a new execution and is never treated as replay.

`provenance-events-v2` gives hypothesis, plan, experiment, evaluation, and
verification events deterministic semantic identities. Identical events return
the original event; conflicting scientific payloads fail closed. The
`evaluation-reuse-v2` additionally reuses an identical successful per-run
result only after the completed four-file artefact bundle passes verification.
The durable record binds reuse to the original bundle hash, schema, scientific
JSON encoding, exact references, evaluator-manifest payload hash, result hash,
experiment hash, evaluator version, dataset version, code version, and
completion timestamp. A different internally valid bundle cannot replace that
identity. Failed, missing, partial, tampered, legacy-v1, schema-incompatible,
or encoding-incompatible evidence is non-reusable and never triggers evaluator
repair. Verification reuse references this authoritative evaluation record and
remains bound to the evaluation hash, verifier version, and task policy version.

START identity errors are the stable `run-execution-errors-v1` public
vocabulary: `conflicting_run_identity`, `conflicting_contract_identity`,
`conflicting_task_identity`, and `conflicting_initial_input_identity`.

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

The four files are one `experiment-bundle-v2` transaction, not four independent
commits. They are fully serialised with `allow_nan=False` before I/O, written to
a temporary sibling directory, flushed and fsynced, checked for completeness,
and published by one directory rename. The evaluator manifest contains the
expected filenames, `scientific-json-v1`, per-payload SHA256 values and a bundle
hash. Identical replay compares bytes and is idempotent; different content at
the same identity is a conflict. A publication failure returns no references,
so graph state and provenance never advertise a partial or missing bundle.

The graph checkpointer stores resumable execution state by `thread_id`.
Scientific provenance is a separate append-only store keyed by `run_id`.
Optuna owns its independent study database, live model calls use an agent-call
database, and knowledge retrievals use a fifth append-only store. Runtime
rejects path collisions between them. None receives `TaskRuntimeContext`.

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

PR 5 local execution is sequential. LangGraph checkpoints, append-only
scientific provenance, Optuna RDB state, agent calls and knowledge retrievals
use separate databases. A future MRI plugin can expose learning rate,
architecture or augmentation parameters through the same capability without
changing the graph.

## PR 5 non-goals

PR 5 does not implement graph writes, ingestion, experiment-result projection,
literature/web search, vector retrieval, embeddings, GraphRAG answers,
text-to-Cypher, OpenEvolve, MRI training, PyTorch/MONAI, critic/report agents,
autonomous tool calling, ReAct loops, model voting, prompt optimisation,
distributed execution, patient-level artefacts, or changes to scientific
evaluators and policies.
