# Auto Researcher

Auto Researcher is a typed, resumable LangGraph control plane for bounded
scientific research. Scientific domains plug into the same graph through a
`ResearchTask` contract; the core owns lifecycle, budgets, orchestration,
structural verification, checkpoints, provenance, and approval.

Repository: [Lifework-Health/auto-researcher](https://github.com/Lifework-Health/auto-researcher)

Issues: [report a problem or propose an enhancement](https://github.com/Lifework-Health/auto-researcher/issues)

PR 5 includes:

- an offline deterministic `synthetic` task;
- an `icca_nbs` adapter that delegates scientific execution and scoring to
  `auto_agent_v2`;
- task-owned configuration, manifests, artefact policies, and verification
  policies;
- a deterministic task registry and task-driven runtime assembly.
- an optional, resumable Optuna 4 ask/tell backend controlled explicitly by
  LangGraph;
- task-owned bounded search spaces for synthetic and iCCA NBS;
- feasible winner selection, diagnostic best-overall reporting, and separate
  checkpoint, provenance and Optuna SQLite stores.
- optional bounded live hypothesis/planner agents using typed structured
  proposals and deterministic reconciliation;
- plugin-owned safe model contexts, honest grounding labels, durable
  replay-safe model-call records, and combined model/evaluator cost accounting;
- optional LangChain/Anthropic integration while mock mode remains the offline
  default.
- a provider-neutral, task-owned knowledge-grounding capability and deterministic
  `retrieve_knowledge` graph node;
- a deterministic static provider plus an optional read-only Neo4j 6.2 adapter
  using fixed, versioned, parameterised Cypher;
- evidence-safe bundle validation, compact cited agent context, conservative
  trust-tier prior caps, durable replay and explicit indeterminate-read retry;
- iCCA query plans compatible with the inspected `knowledge_graph_auto`
  schema, while default grounding remains disabled/offline.

## Quick start

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install -e . --no-deps
.venv/bin/pip install -e '.[hpo]' # only for OPTUNA
.venv/bin/pip install -e '.[agents-anthropic]' # only for live Anthropic
.venv/bin/pip install -e '.[secrets-gcp]' # only for Google Secret Manager
.venv/bin/pip install -e '.[knowledge-neo4j]' # only for Neo4j grounding
.venv/bin/auto-researcher tasks
.venv/bin/auto-researcher run start \
  --task synthetic \
  --contract examples/tasks/synthetic/contract.yaml \
  --task-config examples/tasks/synthetic/task.yaml \
  --run-id demo \
  --thread-id demo-thread
.venv/bin/auto-researcher provenance --run-id demo
.venv/bin/auto-researcher agent-calls list --run-id demo
.venv/bin/auto-researcher knowledge providers
.venv/bin/auto-researcher knowledge retrievals list --run-id demo
```

Run the offline Optuna example with:

```bash
.venv/bin/auto-researcher run start \
  --task synthetic \
  --task-config examples/tasks/synthetic/optuna.yaml \
  --run-id optuna-demo \
  --thread-id optuna-demo-thread \
  --optuna-db .auto-researcher/optuna.sqlite
```

For iCCA, install the reference repository into the same environment:

```bash
.venv/bin/pip install -e ../auto_agent_v2
```

Then edit the external runtime paths in
`examples/tasks/icca_nbs/task.yaml`; paths remain runtime-only and are not
persisted in graph state or scientific provenance.

See [the architecture](docs/architecture/V2_1_ARCHITECTURE.md), the
[task-plugin runbook](docs/runbooks/TASK_PLUGIN_DEVELOPMENT.md), and the
[iCCA runbook](docs/runbooks/ICCA_NBS_RUN.md). Live model setup, pricing,
privacy and indeterminate-call recovery are in
[LIVE_AGENTS.md](docs/runbooks/LIVE_AGENTS.md).
Neo4j least-privilege configuration, readiness, schema preflight and recovery
are documented in
[NEO4J_GROUNDING.md](docs/runbooks/NEO4J_GROUNDING.md).
Safe checkpoint lifecycle operations are documented in
[RUN_EXECUTION.md](docs/runbooks/RUN_EXECUTION.md).
Environment fallback, Google Secret Manager, least-privilege worker identity,
and secret-rotation safety are documented in
[MANAGED_SECRETS.md](docs/runbooks/MANAGED_SECRETS.md).

The real-data, non-patient Iris weighted k-NN benchmark runs fully offline in
DIRECT, Optuna, and OpenEvolve modes. See
[IRIS_KNN_BENCHMARK.md](docs/runbooks/IRIS_KNN_BENCHMARK.md).

The ARC Virtual Cell 2026 foundation freezes Viet's submitted B2 baseline,
official scorer identity and known readiness blockers without copying private
data or large submission payloads. See
[VCC2026_BASELINE_FOUNDATION.md](docs/runbooks/VCC2026_BASELINE_FOUNDATION.md).
