# PR 5: Evidence-safe Neo4j knowledge grounding

## Summary

This PR adds a provider-neutral knowledge boundary and one deterministic
`retrieve_knowledge` node. Tasks own bounded scientific query plans and
grounding policy; providers own external reads and safe row projection; the
core owns lifecycle, policy enforcement, bundle validation, durability,
artefacts, provenance and compact agent context.

Grounding is `DISABLED`, `OPTIONAL` or `REQUIRED`. Disabled mode makes no
provider call. Optional failures continue explicitly ungrounded. Required
failures stop before a live hypothesis model call. No graph node branches on
Neo4j, iCCA, genes, pathways or MRI.

This PR does not add graph writes, ingestion, text-to-Cypher, GraphRAG,
literature/web retrieval, embeddings, OpenEvolve, MRI training, distributed
execution or patient-level graph queries.

## Graph topology

Before:

```text
initialise → supervisor_prepare → generate_hypothesis → plan_search → ...
supervisor_decide → generate_hypothesis | END
```

After:

```text
initialise → supervisor_prepare → retrieve_knowledge → generate_hypothesis → ...
supervisor_decide → retrieve_knowledge | END
```

`retrieve_knowledge` remains the same node for static, Neo4j and future
providers. It performs no model call. The compiled topology is identical across
tasks and provider selections.

## Provider and task architecture

`KnowledgeProvider` is a runtime-checkable protocol with provider identity,
execution-template hashes, readiness, retrieval and close operations.
`KnowledgeProviderRegistry` rejects duplicates, unknown IDs and implementations
that do not satisfy the protocol.

`KnowledgeGroundingCapableTask` is optional and supplies:

- a task/version/schema-bound `KnowledgeQueryPlan`;
- fixed template IDs, typed parameters and aggregate caps;
- a `KnowledgeGroundingPolicy` over entity types, predicates, sources,
  asserted-by identities, trust tiers, confidence and size;
- task-specific relevance between references and scientific parameters.

Synthetic implements a deterministic offline fixture. iCCA implements network
catalogue, configured gene→signature/pathway, signature identity, disease
context and optional immune-bridge plans. Seeds are stable identifiers; no
patient IDs, clinical rows or mutation values are accepted.

## Neo4j provider and schema preflight

`Neo4jKnowledgeProvider` uses the official driver lazily. It always selects the
configured database and uses connectivity/privilege readiness checks,
`execute_read`, fixed parameterised Cypher, bounded timeouts, returned-row
caps, result-summary update counters and safe error codes. It never performs a
test write. Raw nodes/relationships/paths, database exceptions, URIs,
credentials and internal IDs cannot cross the provider boundary.

Immediately before task reads, the fixed schema preflight checks required
labels and relationship types. Stable identifiers are then checked in every
projected entity and source. The provider records configured content identity
without claiming a transactionally immutable graph snapshot.

The inspected external repository was `Lifework-Health/knowledge_graph_auto`
at exact commit:

```text
759090a220148fbe360f4fc519561fa41cb0bfdc
```

The compatibility profile follows the loaded schema rather than the older
design sketch:

- `Signature-[:INCLUDES]->Gene` (not `Gene-[:MEMBER_OF]->Signature`);
- Network nodes are metadata-only; PPI edges remain in the harness data layer;
- backbone labels are Gene, CellState, Signature, Pathway, Disease and Network;
- Subtype, ClinicalCovariate, Cohort and their assertion relations are
  cohort-gated.

## Query template registry

All templates are versioned, parameter-schema checked, task/schema compatible,
bounded in rows and graph hops, deterministically ordered before `$limit`, and
statically linted against write clauses and unrestricted `CALL`. Least-
privilege credentials and result update counters remain independent barriers.

| Template | Version | SHA256 |
|---|---|---|
| `generic.entity_lookup` | `1.0.0` | `b5bbf530bbda97ff32f94df002fb666ef9734937fe70ae9a56622dff8aad808e` |
| `generic.schema_preflight` | `1.0.0` | `2194cae2f6386a7e006fbc7c603b75e6a1518aafc01b2e624f6f538f30a65413` |
| `icca_nbs.disease_context` | `1.0.0` | `31d766af6f05298485c598ce1513bf107d2c31501756a304267c6ef4817647d1` |
| `icca_nbs.gene_signature_pathway` | `1.0.0` | `016c03e760a4160489fe21ed7876505249d727ffda71beb93bb39488f6a5d09a` |
| `icca_nbs.immune_bridge` | `1.0.0` | `3441a5a91f5006ce3e47dff751c8ea05887e227ace51e933b61bda2ee2ef7e42` |
| `icca_nbs.network_catalog` | `1.0.0` | `bbbc536f73b7d7cda9550aff5c697e3eb0943f43f2063569d0ceff6e4f022062` |

Template hashes are snapshotted in each request, included in retrieval identity,
checked again by the provider and emitted in provenance. Changing Cypher
without changing its semantic version therefore changes the retrieval ID and
cannot silently reuse an old bundle.

## Bundle validation, trust and evidence

The immutable model covers graph snapshot metadata, sources, entities,
assertions, compact references, validation summary and artefact references.
The validator checks:

- provider, schema, content and query-plan identity;
- canonical inbound and accepted bundle hashes;
- source and endpoint resolution;
- stable CURIE-like identities and deterministic entity/assertion/reference IDs;
- template, result, assertion, entity, reference, hop and byte caps;
- task-allowed entity/predicate/source/asserted-by/trust policy;
- finite confidence and source/version/publication requirements;
- nested internal IDs, credential fields/values, absolute paths, patient-like
  identifiers, clinical/mutation fields and prohibited properties.

Rejected assertions are counted and cannot produce references. Platform prior
ceilings are CURATED `0.90`, CORPUS `0.70`, LIVE `0.30` and UNVERIFIED `0.30`;
LIVE, UNVERIFIED and LLM-asserted records do not qualify for
`KNOWLEDGE_GROUNDED`. A task may be stricter. Multiple citations use the
minimum qualifying cap.

Knowledge grounding is a proposal prior, not experimental proof. Only evaluator
output and deterministic verification can produce experimental
`SUPPORTED`/`REFUTED`/`INCONCLUSIVE` status.

## Context and reconciliation

Only references from the active completed bundle with a matching bundle hash
enter `HypothesisAgentContext` and `PlannerAgentContext`. Context is sorted,
capped, compact and included in the context hash. It excludes raw graph rows,
Cypher, Neo4j IDs, credentials, paths and patient data.

Prompt sets `hypothesis@2.0.0` and `planner@2.0.0` describe supplied knowledge
as evidence records, forbid invented identifiers/query claims and do not allow
grounding to override task bounds, budget, approval or verification.

`KNOWLEDGE_GROUNDED` is derived only when a proposal cites an active qualifying
reference relevant to its predicted/planned parameters. A retrieved but
uncited bundle leaves the proposal ungrounded. Unknown and cross-bundle
references fail. A planner does not inherit the hypothesis grounding label
unless it cites evidence itself.

## Retrieval identity, durability and replay

Retrieval identity binds run/cycle, task/version, contract, provider/adapter
version, graph alias, schema/content configuration, query-plan version, typed
parameters and all execution-template hashes.

The separate append-only store records `RESERVED`, `COMPLETED`, `FAILED` and
`INDETERMINATE`. A completed bundle is reused exactly and the provider is not
called on replay. A started read without a durable outcome becomes
indeterminate and cannot repeat automatically. The CLI creates an explicitly
authorised child; repeated crash/retry lineages remain traversable and preserve
all ancestors.

Five atomic safe artefacts are produced:

```text
runs/<run-id>/knowledge/<retrieval-id>/
├── retrieval_request.json
├── query_plan.json
├── graph_snapshot.json
├── knowledge_bundle.json
└── validation_summary.json
```

Checkpoint state contains only `KnowledgeBundleReference`.

## Provenance and CLI

Replay-idempotent event order is:

```text
KNOWLEDGE_RETRIEVAL_RESERVED
→ KNOWLEDGE_RETRIEVAL_COMPLETED | KNOWLEDGE_RETRIEVAL_FAILED
→ KNOWLEDGE_BUNDLE_VALIDATED
→ MODEL_CALL_RESERVED
→ MODEL_CALL_COMPLETED | MODEL_CALL_FAILED
→ HYPOTHESIS_PROPOSED
→ SEARCH_PLANNED
→ existing experiment/evidence events
```

Knowledge events reference retrieval/bundle identity and hashes, provider,
safe graph alias, schema/content configuration, plan/template hashes, accepted
and rejected counts, trust summary and artefacts. Hypothesis/plan events link
the bundle, cited reference IDs, grounding and deterministic prior. They contain
no claims, rows, Cypher, URI, credentials or patient data.

New CLI surfaces list provider installation, check readiness, list/show safe
retrieval metadata and explicitly retry indeterminate reads. Run output reports
grounding mode, provider, graph/schema/content identity, status, bundle ID/hash,
reference/trust counts, cited IDs and artefacts without printing secrets or raw
queries.

## Demonstrations

- Static synthetic + fake live agents: two source-backed curated references
  were accepted; the unverified LLM assertion was rejected; hypothesis and plan
  cited the active reference and became `KNOWLEDGE_GROUNDED`; prior was capped
  at `0.90`; score was `0.84`; simulated experimental evidence remained
  `INCONCLUSIVE`.
- Static synthetic uncited path: the bundle existed but both hypothesis and
  plan stayed `UNGROUNDED`.
- Required provider unavailable: run stopped before any fake paid model call.
- Fake iCCA + Neo4j-shaped records: schema preflight, fixed gene/pathway query,
  bundle validation, cited context and the unchanged fake v2 evaluator
  completed; the hypothesis became `KNOWLEDGE_GROUNDED`; score was `0.8`.
- Replay: a completed bundle caused zero additional provider calls.
- Crash recovery: reservation became indeterminate, explicit child and
  grandchild retries executed once, and all records remained append-only.

## Dependencies and verification

- Neo4j Python driver: `6.2.0`
- pytz resolved lock: `2026.3.post1`
- core install remains independent of Neo4j
- no neo4j-graphrag, APOC, n10s, graph chain, embedding or vector dependency
- 168 tests passed
- 2 tests skipped
- 0 tests failed

The skips are the opt-in paid Anthropic end-to-end test and the optional real
iCCA patient-data gate. All mock DIRECT/OPTUNA, fake live DIRECT/OPTUNA, iCCA
adapter, model-call replay, budget and provenance regressions passed.

The official Neo4j driver is installed locally, but no disposable/local Neo4j
endpoint or credentials were supplied, so a real local graph test is pending.
No Aura variables were supplied; live Aura validation is also pending. No
successful live graph test is simulated or claimed.

## Security controls and known limitations

Credentials are environment-only. Provider configuration rejects connection-
like graph aliases and credential fields recursively. Runtime database,
schema/content and contract policy cannot be weakened. Query templates,
read-only accounts, read routing, explicit database selection, timeouts, row
caps, schema checks, update counters, safe projection, bundle validation and
prompt/reference reconciliation form independent barriers.

Known limitations: execution is local/sequential; configured content labels are
operator-reviewed rather than transactional graph snapshots; Aura privilege
introspection may be restricted; PR 5 has no ingestion, cross-run curation,
literature retrieval, graph writes or automatic promotion of experiment
results.

## Proposed PR 6

After explicit approval, PR 6 should add bounded program-evolution search
through the same task/search contracts and lifecycle, without adding graph
writes or MRI training. PR 6 is not started here.
