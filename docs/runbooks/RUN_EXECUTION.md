# Safe run start, resume, and inspection

Auto Researcher uses `run-execution-v2`. A LangGraph checkpoint is keyed by
thread ID, while the persisted execution identity also binds run ID, contract,
task, graph version, and initial input.

## Commands

Start a new thread once:

```bash
auto-researcher run start \
  --task synthetic \
  --contract examples/tasks/synthetic/contract.yaml \
  --task-config examples/tasks/synthetic/task.yaml \
  --run-id demo \
  --thread-id demo-thread
```

Continue a non-terminal checkpoint:

```bash
auto-researcher run resume \
  --thread-id demo-thread \
  --task-config examples/tasks/synthetic/task.yaml
```

For a human-approval interrupt, add exactly one of `--approve` or `--reject`;
the runtime resumes it with `Command(resume={"approved": ...})`.

Inspect a terminal checkpoint without executing a node:

```bash
auto-researcher run inspect \
  --thread-id demo-thread \
  --checkpoint-db .auto-researcher/checkpoints.sqlite
```

`START` rejects every existing thread. `RESUME` rejects unknown and terminal
threads. `INSPECT` requires a terminal thread and performs no provider, model,
evaluator, verifier, artefact, checkpoint, or provenance write.

START rejections use the stable public vocabulary `run-execution-errors-v1`:

| Conflict | Error code |
| --- | --- |
| run identity | `conflicting_run_identity` |
| contract identity | `conflicting_contract_identity` |
| task identity | `conflicting_task_identity` |
| canonical initial input | `conflicting_initial_input_identity` |
| exact duplicate thread | `thread_already_exists_use_resume_or_inspect` |

The CLI prints these codes unchanged. Identity preflight occurs before live
agents, knowledge providers, task evaluators, or other external dependencies
are created.

## Python API

```python
from auto_researcher.runtime import inspect_terminal_run, resume_run, start_run

initial = start_run(graph, input, config)
resumed = resume_run(graph, config)
final = inspect_terminal_run(graph, config)
```

`graph.invoke(None, config)` is a continuation mechanism used internally by
`resume_run`. Terminal inspection uses `graph.get_state(config)`. Never use
`graph.invoke(initial_payload, same_terminal_thread_config)` as replay: a fresh
dictionary asks LangGraph to begin another execution on that thread.

## Defence in depth

The execution guard is the primary protection. Semantic provenance uniqueness
and per-run evaluator/verifier reuse also protect direct graph callers. An
identical completed evaluation is reused only through `evaluation-reuse-v2`.
The record binds the result to the exact original `experiment-bundle-v2` hash,
schema, scientific JSON encoding, expected references, evaluator-manifest
payload hash, and completion time. A complete but recomputed replacement bundle
is a conflict even when it is internally valid. Missing, partial, tampered,
schema-incompatible, encoding-incompatible, changed, or conflicting scientific
state fails closed without rerunning the evaluator.

`evaluation-reuse-v1` rows and checkpoint 03 are legacy inputs. They are not
silently upgraded from whatever artefacts happen to exist and cannot be used as
v2 reusable evidence. Start a fresh run under the current protocol.
