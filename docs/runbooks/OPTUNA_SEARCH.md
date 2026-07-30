# Optuna search runbook

## Install and verify capability

```bash
python -m pip install -e '.[hpo]'
auto-researcher tasks
```

OPTUNA is available only when Optuna is installed, the selected task implements
the optional capability, the contract allows OPTUNA, its study specification is
valid, and storage is ready. Router errors identify the missing condition.

## Configure and run

Use `examples/tasks/synthetic/optuna.yaml` or
`examples/tasks/icca_nbs/optuna.yaml`. The `search.type` must be `OPTUNA`.
Ranges may narrow task-registered limits but cannot widen or change types,
steps, logarithmic semantics or categorical order.

```bash
auto-researcher run \
  --task synthetic \
  --task-config examples/tasks/synthetic/optuna.yaml \
  --run-id study-001 \
  --thread-id study-001-thread \
  --checkpoint-db .auto-researcher/checkpoints.sqlite \
  --provenance-db .auto-researcher/provenance.sqlite \
  --optuna-db .auto-researcher/optuna.sqlite
```

All three database paths must be distinct. The output reports trial counts,
best feasible and best overall diagnostic scores, finish reason, evidence and
stable relative artefact references.

## Resume

Repeat the command with the same run ID, thread ID, configuration and database
paths. The graph resumes from its checkpoint. Study identity rejects changes to
dataset, objective, evaluator/code version or search space. A tagged running
trial is recovered by slot; a completed tell and deterministic provenance event
are idempotent.

## Trial and result interpretation

- `COMPLETE`, feasible: valid candidate for the research winner.
- `COMPLETE`, infeasible: valid measurement retained diagnostically.
- `FAIL`: evaluation failed, score was non-finite, or structural verification
  failed. No penalty value is invented.
- Best feasible is the selected research result.
- Best overall is diagnostic and may violate constraints.
- No feasible result leaves primary result slots empty.

## Stale trials and limitations

An untagged, foreign, duplicate-slot or multiple running trial is not guessed
away; inspect the Optuna database and repair it manually or start a new run.
PR 3 is single-objective and sequential. It does not provide pruning,
PostgreSQL, distributed workers, OpenEvolve, live LLM agents or MRI training.
