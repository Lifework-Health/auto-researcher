# Auto Researcher v2.1 architecture

## Scope

PR 1 establishes a minimal, offline research workflow and the stable interfaces
through which the v2 scientific evaluator and future search systems can connect.
It does not implement live model calls, Optuna ask/tell, OpenEvolve, Neo4j, patient
data ingestion, or the v2 scientific methods.

## Component types

The hypothesis generator and planner are the only LLM-shaped nodes. They make
scientific proposals through protocols; PR 1 injects deterministic mock
implementations and makes no external calls.

Initialisation, the supervisor, approval routing, human approval, search routing,
evaluation invocation, verification, provenance writing, and stop decisions are
deterministic control nodes. They own lifecycle, budgets, routing, and truth.
Agents cannot access the provenance store, change the contract, spend budgets, or
skip verification.

DIRECT is the only installed search backend. It selects a single deterministic
configuration and emits an `ExperimentSpec`; it never measures that experiment.
OPTUNA and OPENEVOLVE are closed enum values with protocol boundaries only. A
request for either produces a structured unavailable result and stops without
substitution.

## Why LangGraph is the control plane

Major execution steps are explicit graph nodes and all branches are deterministic
functions over typed state. LangGraph provides executable checkpoints, interrupts,
and replay semantics. The compact `ResearchState` stores current domain objects,
identifiers, budgets, errors, and stop state; it does not accumulate prompt or
report history.

The graph invokes the evaluator and verifier on the DIRECT path in fixed order.
No agent receives either component. Human approval uses `interrupt()` before any
node side effect, allowing the node to safely restart when a `Command(resume=...)`
is supplied.

## Scientific truth

Agents propose; they do not measure or adjudicate. The evaluator is the only
source of measured scores and constraint outcomes. The verifier reconciles
experiment identity, evaluator registration, required metrics, constraints, and
claimed versus measured scores. Its schema and code make `SUPPORTED` structurally
invalid for MOCK or SIMULATED evidence.

The `V2EvaluatorAdapter` is intentionally a skeleton. Its documentation defines
how v2 selected metrics and eligibility outputs will map to v2.1 contracts, while
leaving propagation, PAC, survival, clinical gates, and selection rules owned by
the existing v2 repository.

## Checkpoints are not provenance

The LangGraph checkpointer persists executable state by `thread_id`; tests use
`InMemorySaver`, while local runs use `langgraph-checkpoint-sqlite`. A runtime can
be destroyed, rebuilt around the same checkpoint file, and resumed with the same
thread.

Scientific provenance is a separate append-only SQLite store keyed by `run_id`.
It records typed `DecisionEvent` objects for the hypothesis, plan, experiment,
evaluation, and verification. Its protocol exposes append and read operations
only. Runtime construction rejects a shared checkpoint/provenance file. This
separation lets execution-state retention and scientific audit retention evolve
independently.

## Implemented flow

The source of truth for edges is [`graph.mmd`](graph.mmd). A normal one-cycle mock
run visits every non-approval node on the DIRECT path, records five scientific
events, and ends because its cycle budget is reached. MOCK evidence remains
`INCONCLUSIVE`, even when score reconciliation and constraints pass.
