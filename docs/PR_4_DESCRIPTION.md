# PR 4: Bounded live LLM hypothesis and planning agents

## Summary

This PR adds optional live hypothesis and planner implementations without
changing the LangGraph node or edge topology. Mock mode remains the offline
default. Live agents produce typed untrusted proposals through a
provider-neutral model client; deterministic reconcilers remain responsible for
platform IDs, grounding, search validation, budgets and approval.

No scientific evaluator or verification policy changed. The model cannot
execute tools, alter `ResearchContract`, write graph state, invent evidence
references, set measured scores or mark evidence supported.

## Live agent and provider architecture

`StructuredModelClient.generate_structured` is the only provider boundary.
`LangChainStructuredModelClient` accepts an injected `BaseChatModel` and calls
the current Pydantic `with_structured_output(..., include_raw=True)` interface.
The optional Anthropic factory uses an explicit model ID, per-role temperature,
token limit and timeout with provider retries disabled; platform retry policy
remains bounded and auditable.

Provider dependencies are optional. Core, task listing, DIRECT, OPTUNA and
default tests work without `langchain-anthropic`. The API key is read only from
`ANTHROPIC_API_KEY` and never enters contracts, call records, provenance,
artefacts or output.

## Structured proposals and deterministic reconciliation

`HypothesisProposal` contains only a testable statement, concise rationale,
predicted task subspace, measurable expected observation, explicit
falsification condition, supplied evidence references and bounded confidence.
It cannot supply a hypothesis ID, status or provenance.

`PlannerProposal` contains only DIRECT/OPTUNA choice, target, proposed bounded
space, experiment budget, rationale and an approval recommendation. It cannot
supply a request/hypothesis/evaluator ID, measured score or provenance.

`HypothesisReconciler` validates task compatibility and permitted references,
derives deterministic identity and honest grounding, fixes status to `OPEN`,
sets source `MODEL_GENERATED` and caps prior weight at 0.3/0.6/0.8 for
ungrounded/contract/prior-result grounding. Model confidence is not scientific
confidence.

`PlannerReconciler` verifies installed, contract-allowed and task-supported
search, rejects budget clipping and fallback, delegates DIRECT normalisation to
the task, and delegates OPTUNA narrowing validation to the task-owned study
specification. It derives deterministic request identity and applies contract
or model-recommended approval.

## Graph topology

`src/auto_researcher/graph/builder.py` is byte-for-byte unchanged from merged
PR 3. Live and mock compiled Mermaid graphs compare equal in integration tests.
Only the implementations injected into `generate_hypothesis` and `plan_search`
differ. Supervisor, routers, approval, search, evaluator, verifier, provenance
and lifecycle decisions remain deterministic.

## Safe task context and prior memory

`AgentContextCapableTask` supplies bounded `TaskAgentContext`. Synthetic exposes
its safe configuration schema and registered Optuna space. iCCA exposes
mutation-only NBS concepts, available enum labels, alpha/K semantics,
eligibility descriptions and aggregate manifest metadata. It excludes patient
IDs, raw mutation/clinical values, runtime paths and evaluator internals.

`AgentContextAssembler` reads only the active contract, task-safe context,
installed/allowed capabilities, compact budget state and prior verified
scientific provenance. Prior hypotheses/results, artefact references and total
characters are capped and deterministically sorted. Canonical JSON yields a
stable context hash. Unverified model claims never become established prior
findings.

Before a future knowledge integration, producible grounding states are only
`UNGROUNDED`, `CONTRACT_GROUNDED` and `PRIOR_RESULTS_GROUNDED`.
`KNOWLEDGE_GROUNDED` is reserved and cannot be emitted.

## Prompts

Repository prompt files:

- `hypothesis@1.0.0`, template-set SHA256
  `27bd3f206761e3e1469ec81c7a1fb1a8a7651653cff92c385abdd5bebf95daf8`;
- `planner@1.0.0`, template-set SHA256
  `dcf4dbc29d358a6da79852cd9e46a64cda38559e224266e45ae8d44439c1a06d`.

They define allowed decisions, prohibited actions, honesty, grounding,
budget-awareness, untrusted context delimiters and structured-only output.
They neither request nor store chain of thought.

## Model-call identity, replay and retry

A deterministic logical call ID binds run, cycle, role, prompt version, context
hash, schema fingerprint, provider and explicit model ID. An append-only
`AgentCallStore`, separate from the checkpointer, records immutable
`RESERVED`, `COMPLETED`, `FAILED` and `INDETERMINATE` snapshots.

A completed call is reused and conflicting completions fail closed. A started
reservation without an outcome becomes indeterminate and is never
automatically repeated. `agent-calls retry` creates a new authorised child
attempt linked to the original; retry chains preserve every original snapshot.
Model-call provenance IDs derive from call-record IDs and are replay
idempotent.

The platform retries only transient provider errors, timeouts and invalid
structured output within the configured ceiling, using bounded exponential
backoff. Authentication, rate limiting, context-size and permanent provider
errors are not automatically retried. Reconciliation correction messages are
concise machine codes, not private reasoning.

## Budgets and cost accounting

`AgentBudgetPolicy` bounds per-role calls per cycle, attempts, context
characters, output tokens, cost per logical call and total provider attempts.
Live mode requires explicit positive, versioned pricing before a provider
request.

`BudgetState` now tracks provider calls, input/output/cache creation/cache read
tokens, model cost and evaluator cost. `cost_used` is the combined total and is
still governed by `ResearchContract.maximum_cost`. Returned usage from invalid
output and provider retries is accumulated and charged.

## Provenance and CLI

Generic provenance adds `MODEL_CALL_RESERVED`, `MODEL_CALL_COMPLETED` and
`MODEL_CALL_FAILED`. Hypothesis and plan events reference their model-call IDs
and record proposal source, grounding and prompt version. Call events contain
safe provider/model, hashes, usage, currency and estimated cost—never rendered
prompts, API keys or hidden reasoning.

The task YAML supports explicit `agents.mode: mock|live`, provider/model,
per-role configuration, pricing and budget policy. Run output prints mode,
provider/model, prompt versions, grounding, calls, tokens and separate/combined
cost. New commands list, show and explicitly retry agent calls.

## Demonstrations

Deterministic fake pricing used one input and two output currency units per
million tokens. Each successful fake run made exactly two provider requests and
recorded model cost `0.0004`.

- Synthetic DIRECT: completed, one verified experiment, score `0.84`.
- Synthetic OPTUNA: completed two verified trials; selected score `0.737639`.
- Fake iCCA DIRECT: completed through the imported fake v2 evaluator, score
  `0.8`.
- Fake iCCA OPTUNA: completed two verified trials through the same graph,
  selected score `0.8`.

Invalid live planning was retried twice, charged as three total provider calls
including hypothesis generation, failed the run, and launched no experiment.
Crash simulation proved reservation → indeterminate → explicit linked retry;
completed replay made no second provider request.

## Versions and verification

- LangGraph `1.2.10`
- LangChain Core `1.5.3`
- LangChain Anthropic `1.5.3`
- Anthropic SDK `0.120.2`
- Optuna `4.9.0`
- 132 tests passed, 2 skipped, 0 failed
- skipped: explicitly paid live Anthropic end-to-end test (credentials, model
  and pricing were not supplied)
- skipped: optional real iCCA patient-data gate

The four-way fake-live matrix, existing mock DIRECT/OPTUNA regressions,
provider/error tests, context/reconciliation tests, cost tests and replay tests
all passed. No successful real provider call is claimed.

## Known limitations and proposed PR 5

PR 4 is local, sequential and single-provider-per-call. It does not implement
Neo4j, literature/web search, RAG, OpenEvolve, MRI training, critic/report
agents, autonomous tools, ReAct, model voting, prompt optimisation, parallel
calls or production deployment.

Recommended PR 5 scope is a separately approved, evidence-safe knowledge
grounding layer that can make `KNOWLEDGE_GROUNDED` real while preserving the
same agent context, reference validation, call durability and deterministic
graph control. No PR 5 work is included here.
