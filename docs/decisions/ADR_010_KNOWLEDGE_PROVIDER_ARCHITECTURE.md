# ADR 010: Provider-neutral task-owned knowledge retrieval

Status: accepted for PR 5.

## Context

Scientific tasks need different schemas and relevance rules, while lifecycle,
budgets, approval and model calls must remain domain-neutral. Letting an agent
choose a database query would mix untrusted generation with infrastructure and
scientific authority.

## Decision

Knowledge retrieval uses a runtime-checkable `KnowledgeProvider` boundary.
Tasks may separately implement `KnowledgeGroundingCapableTask` to return an
immutable bounded query plan and grounding policy. The core validates plan
template identity, version, parameters and limits, runs the provider through
the generic `retrieve_knowledge` node, validates its bundle, and exposes only
compact cited references to agents.

Task plugins contain scientific vocabulary and relevance. Providers contain
connection and result-projection logic. Fixed query templates are registered
outside both the graph and the model. Selecting static or Neo4j changes injected
dependencies, not LangGraph topology.

## Alternatives considered

- Neo4j calls inside agents: rejected because models would control side effects.
- Generic model-generated Cypher: rejected because validation cannot make
  arbitrary queries a stable evidence boundary.
- Task-specific graph nodes: rejected because every domain would fork lifecycle.
- Loading all graph content into prompts: rejected for privacy, cost, authority
  and reproducibility.
- Neo4j GraphRAG: deferred; PR 5 needs references and provenance, not generated
  database answers.

## Consequences

Future MRI or relational providers can reuse the lifecycle. A task without the
capability still runs when grounding is disabled or optional; a required
contract fails before model calls.
