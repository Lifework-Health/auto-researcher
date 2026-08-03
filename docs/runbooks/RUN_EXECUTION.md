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
identical completed evaluation is reused only after its existing transactional
artefact bundle is complete and untampered. Missing, changed, or conflicting
scientific state fails closed.
