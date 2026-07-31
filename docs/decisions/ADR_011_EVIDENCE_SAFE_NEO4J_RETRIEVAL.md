# ADR 011: Evidence-safe Neo4j retrieval

Status: accepted for PR 5.

## Context

Neo4j is external, mutable infrastructure. Database connectivity alone does not
make graph assertions citable or prove that a query is read-only.

## Decision

The Neo4j provider uses the official driver, an explicit database, least-
privilege credentials, connectivity and privilege readiness checks,
`execute_read`, fixed versioned parameterised templates, per-query timeout and
row caps, a schema preflight, and result-summary update-counter checks. The
template linter rejects write clauses and unrestricted procedure calls. Schema
preflight uses only the allowlisted read-mode `db.labels()` and
`db.relationshipTypes()` metadata procedures; it does not scan graph data.

Projected rows contain allowlisted fields only. Raw nodes, relationships,
paths, connection exceptions and internal IDs never leave the provider.
Stable CURIE-like identifiers, versioned sources and deterministic assertion
IDs are required. Bundle validation applies task trust/source/predicate policy
before any reference reaches a prompt.

`knowledge_graph_auto` commit
`759090a220148fbe360f4fc519561fa41cb0bfdc` was inspected. The compatibility
profile follows the loaded `Signature-[:INCLUDES]->Gene` direction and treats
`Network` as metadata-only; it does not assume PPI edges are stored in Neo4j.
Configured content version is recorded honestly and is not described as a
transactionally immutable graph snapshot.

## Alternatives considered

- Arbitrary or text-to-Cypher queries: rejected.
- Read routing alone: insufficient; credentials, template lint, transactions,
  counters and validation are independent barriers.
- Broad graph dumps: rejected for scope, privacy and context size.
- Neo4j internal IDs: rejected because they are not stable evidence identity.

## Consequences

The adapter is intentionally narrower than a general graph client. Aura
read-only privilege introspection may be unavailable; when verification is
required, readiness fails without attempting a test write.
