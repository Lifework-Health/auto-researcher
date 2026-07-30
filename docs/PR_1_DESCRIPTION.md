# PR 1: LangGraph control plane, contracts, and mock end-to-end loop

## Built

- Immutable, versioned Pydantic contracts and compact typed graph state
- Explicit LangGraph nodes for proposal, planning, approval, routing, DIRECT
  search, evaluation, verification, provenance, and lifecycle decisions
- Deterministic offline agents, search backend, evaluator landscape, and verifier
- Replay-safe human approval interrupts
- In-memory and SQLite checkpointers behind runtime factories
- A separate append-only SQLite scientific provenance store
- CLI commands for mock runs and provenance inspection
- Unit and integration coverage, including process-style SQLite reconstruction
- Architecture documentation, graph diagram, and ADRs

## Deliberately not implemented

Live LLM calls, v2 scientific code, Optuna ask/tell, OpenEvolve, Neo4j, Postgres,
patient data, distributed execution, reporting, and a web UI. `V2EvaluatorAdapter`
documents the future boundary without presenting an incomplete integration as
working.

## Verification

The default suite runs offline. A normal mock run completes one cycle, automatically
invokes verification, records hypothesis through verification events, and leaves
MOCK evidence `INCONCLUSIVE`.
