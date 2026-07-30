# Auto Researcher v2.1

Auto Researcher is a typed, resumable LangGraph control plane for bounded
scientific research. Scientific domains plug into the same graph through a
`ResearchTask` contract; the core owns lifecycle, budgets, orchestration,
structural verification, checkpoints, provenance, and approval.

PR 3 includes:

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

## Quick start

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install -e . --no-deps
.venv/bin/pip install -e '.[hpo]' # only for OPTUNA
.venv/bin/auto-researcher tasks
.venv/bin/auto-researcher run \
  --task synthetic \
  --contract examples/tasks/synthetic/contract.yaml \
  --task-config examples/tasks/synthetic/task.yaml \
  --run-id demo \
  --thread-id demo-thread
.venv/bin/auto-researcher provenance --run-id demo
```

Run the offline Optuna example with:

```bash
.venv/bin/auto-researcher run \
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
[iCCA runbook](docs/runbooks/ICCA_NBS_RUN.md).
