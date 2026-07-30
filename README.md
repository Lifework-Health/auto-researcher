# Auto Researcher v2.1

Auto Researcher is a typed, resumable LangGraph control plane for bounded
scientific research. Scientific domains plug into the same graph through a
`ResearchTask` contract; the core owns lifecycle, budgets, orchestration,
structural verification, checkpoints, provenance, and approval.

PR 2 includes:

- an offline deterministic `synthetic` task;
- an `icca_nbs` adapter that delegates scientific execution and scoring to
  `auto_agent_v2`;
- task-owned configuration, manifests, artefact policies, and verification
  policies;
- a deterministic task registry and task-driven runtime assembly.

## Quick start

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install -e . --no-deps
.venv/bin/auto-researcher tasks
.venv/bin/auto-researcher run \
  --task synthetic \
  --contract examples/tasks/synthetic/contract.yaml \
  --task-config examples/tasks/synthetic/task.yaml \
  --run-id demo \
  --thread-id demo-thread
.venv/bin/auto-researcher provenance --run-id demo
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
