# Live agents runbook

Live agents are optional. Mock mode is the default for examples, CI and local
offline work. PR 4 supports Anthropic through LangChain structured output; it
does not implement Neo4j grounding, web/literature retrieval, tools or
OpenEvolve.

## Install

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install -r requirements-agents.lock
.venv/bin/pip install -e . --no-deps
```

The resolved integration versions are `langchain-core==1.5.3`,
`langchain-anthropic==1.5.3` and `anthropic==0.120.2`. The core package does not
import `langchain-anthropic` unless live Anthropic mode is requested.

## Credentials and configuration

Export the key only in the process environment:

```bash
export ANTHROPIC_API_KEY='...'
```

Never put the key in YAML, shell history, task options, provenance or
artefacts. Copy the `agents` section from
`examples/agents/anthropic-live.yaml` into the selected task YAML, replace the
explicit model ID if required, and replace every pricing placeholder with
current rates that you have reviewed. Pricing needs a version and currency.
Floating `latest` model aliases and zero input/output prices are rejected.

Hypothesis and planner prompts are both version `1.0.0`. Their files and SHA256
hashes are recorded with each call. Rendered prompts are not stored.

| Prompt | Template-set SHA256 |
|---|---|
| `hypothesis@1.0.0` | `27bd3f206761e3e1469ec81c7a1fb1a8a7651653cff92c385abdd5bebf95daf8` |
| `planner@1.0.0` | `dcf4dbc29d358a6da79852cd9e46a64cda38559e224266e45ae8d44439c1a06d` |

Set:

- a strict maximum output token count and timeout for each role;
- at most two attempts unless a reviewed policy requires less;
- a maximum cost per logical call no greater than remaining contract cost;
- a maximum total model-call count;
- `ResearchContract.maximum_cost` high enough for combined model and evaluator
  estimates.

## Run and inspect

```bash
.venv/bin/auto-researcher run \
  --task synthetic \
  --contract examples/tasks/synthetic/contract.yaml \
  --task-config /path/to/synthetic-with-live-agents.yaml \
  --run-id live-demo \
  --thread-id live-demo-thread \
  --checkpoint-db .auto-researcher/checkpoints.sqlite \
  --provenance-db .auto-researcher/provenance.sqlite \
  --agent-calls-db .auto-researcher/agent-calls.sqlite

.venv/bin/auto-researcher agent-calls list --run-id live-demo
.venv/bin/auto-researcher agent-calls show --call-id model-call-...
```

The run summary prints mode, provider/model, calls, input/output/cache tokens,
model/evaluator/total cost, prompt versions and hypothesis grounding. It never
prints the API key.

## Indeterminate call recovery

If a process stopped after a reservation began, resuming the thread marks the
call `INDETERMINATE` and exits non-zero without a second provider request.
Inspect the snapshots and provider account, then explicitly authorise one
linked attempt:

```bash
.venv/bin/auto-researcher agent-calls show \
  --call-id model-call-original \
  --agent-calls-db .auto-researcher/agent-calls.sqlite

.venv/bin/auto-researcher agent-calls retry \
  --call-id model-call-original \
  --agent-calls-db .auto-researcher/agent-calls.sqlite
```

Resume the same LangGraph thread. The retry has a new call ID and
`retry_of_call_id`; the original reservation is never overwritten. A second
crash requires inspecting and authorising the new indeterminate child.

## Troubleshooting

- Missing optional package: install `.[agents-anthropic]` or the agent lock.
- Missing key: export `ANTHROPIC_API_KEY`; do not add it to config.
- Missing/zero pricing: provide reviewed positive rates before any call.
- Context too large: lower prior/context limits; do not remove privacy filters.
- Authentication, invalid model and permanent provider errors are not retried.
- Timeout, transient provider and invalid structured-output failures may retry
  only within the configured attempt ceiling, and all returned usage is charged.
- Unknown evidence IDs, widened Optuna ranges, excessive budgets and invalid
  DIRECT fields fail reconciliation and never launch an experiment.

## Privacy and grounding

Task contexts must contain summaries and schemas only. Do not add patient IDs,
raw records, matrices, absolute paths, credentials or evaluator internals.
iCCA context is mutation-only at the conceptual level and contains aggregate
manifest information only. Until a later, separately approved knowledge
integration exists, `KNOWLEDGE_GROUNDED` cannot be produced and model
pretraining is not a citable source.
