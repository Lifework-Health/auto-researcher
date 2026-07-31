# Neo4j grounding runbook

## Install and configure

Core Auto Researcher does not require Neo4j. Install the pinned optional
adapter only where grounding is needed:

```bash
.venv/bin/pip install -e '.[knowledge-neo4j]'
```

PR 5 pins `neo4j==6.2.0`. Create a dedicated database account with only the
minimum traversal privileges needed by the registered templates. Do not reuse
loader/admin credentials. Supply connection values only at runtime:

```bash
export NEO4J_URI='neo4j+s://...'
export NEO4J_USERNAME='auto-researcher-reader'
export NEO4J_PASSWORD='...'
export NEO4J_DATABASE='neo4j'
```

Never put those values in YAML, contracts, artefacts, logs or provenance.
`graph_alias` is a safe operator name, not a URI. `schema_version` describes
the compatibility profile; `content_version` is the reviewed deployment label
and is not a claim of transactionally immutable graph state.

## Task configuration

The research contract is authoritative:

```yaml
grounding:
  mode: OPTIONAL
  permitted_providers: [neo4j]
  permitted_trust_tiers: [CURATED, CORPUS]
  minimum_assertion_confidence: 0.6
  maximum_knowledge_references: 20
  maximum_query_records: 100
  maximum_graph_hops: 3
  maximum_retrieval_duration: 20
  knowledge_schema_version: knowledge-graph-auto-v0.1
  knowledge_content_version: backbone-2026-06
```

The task YAML selects a configuration no weaker than that contract:

```yaml
grounding:
  mode: OPTIONAL
  provider: neo4j
  graph_alias: cell-biology-grounding
  database: neo4j
  schema_version: knowledge-graph-auto-v0.1
  content_version: backbone-2026-06
  require_verified_read_only: true
  query_timeout_seconds: 20
  maximum_records: 100
  maximum_attempts: 2
  minimum_assertion_confidence: 0.6
  allowed_trust_tiers: [CURATED, CORPUS]
  disease_curies: [MONDO:0004992]
  gene_curies: [HGNC:11998]
  gene_seed_provenance: CURATED
  signature_ids: [HALLMARK_P53_PATHWAY]
  include_network_catalog: true
  include_pathways: true
  include_immune_bridge: false
```

Seeds must be stable identifiers. Patient-derived lists, patient IDs, mutation
matrix values and clinical rows are prohibited.

## Readiness and schema preflight

Run:

```bash
auto-researcher knowledge providers
auto-researcher knowledge readiness \
  --task icca_nbs \
  --contract examples/knowledge/icca_nbs-contract.yaml \
  --task-config examples/knowledge/icca_nbs-neo4j.yaml
```

Readiness verifies configuration, connectivity and the account privilege view.
It never performs a test write. Immediately before task templates, a fixed read
preflight uses the allowlisted read-mode `db.labels()` and
`db.relationshipTypes()` metadata procedures to compare required labels and
relationship types without scanning graph data. The read-only account must be
able to execute those two procedures. Missing `INCLUDES`, required labels or
stable projected identifiers fail closed. The inspected profile uses
`Signature-[:INCLUDES]->Gene`; Network nodes are metadata-only.

To independently confirm least privilege, review `SHOW CURRENT USER
PRIVILEGES` as an administrator and verify no `WRITE`, `CREATE`, `DELETE` or
property-setting grant exists for this account. Also monitor server query logs
for the registered template hashes. Do not grant writes merely to make the
readiness check pass.

## Templates and bundle inspection

Templates are fixed files under `auto_researcher/knowledge/queries` and are
identified by ID, version and SHA256. Runtime configuration supplies typed
parameters only; models never see Cypher or a database tool.

Run records are stored separately:

```bash
auto-researcher knowledge retrievals list --run-id <run-id>
auto-researcher knowledge retrievals show --retrieval-id <retrieval-id>
```

Safe artefacts are under:

```text
runs/<run-id>/knowledge/<retrieval-id>/
├── retrieval_request.json
├── query_plan.json
├── graph_snapshot.json
├── knowledge_bundle.json
└── validation_summary.json
```

They must contain no credentials, URI, raw Cypher, internal Neo4j IDs, patient
data or unrestricted properties.

## Replay and explicit retry

A `COMPLETED` retrieval reuses its exact bundle and does not query Neo4j. A
started reservation discovered without an outcome becomes `INDETERMINATE`.
Inspect external logs, then explicitly authorise a linked attempt:

```bash
auto-researcher knowledge retrievals retry \
  --retrieval-id <retrieval-id>
```

Resume the same LangGraph thread. Never delete the original record or manually
change its status.

## Troubleshooting and Aura limitations

- `PROVIDER_NOT_INSTALLED`: install `.[knowledge-neo4j]`.
- `PROVIDER_NOT_CONFIGURED`: check environment variables without printing them.
- `AUTHENTICATION_FAILED` or `CONNECTIVITY_FAILED`: review account/database and
  network policy; raw driver messages are intentionally suppressed.
- `READ_ONLY_NOT_VERIFIED`: confirm the privilege view is available to the
  account or use an operator-reviewed setting; no test write is attempted.
- `SCHEMA_MISMATCH`: compare the deployment with the registered schema profile.
- `CONTENT_VERSION_MISMATCH`: align reviewed contract and runtime labels.
- `QUERY_TIMEOUT` or `RESULT_LIMIT_EXCEEDED`: narrow task-owned seeds; do not
  weaken the contract.
- `BUNDLE_VALIDATION_FAILED`: inspect the safe validation summary.

Aura tiers and roles may restrict privilege introspection, query logging or
test-database creation. Live Aura validation is optional and must be explicitly
enabled outside ordinary CI. No successful live test should be reported unless
it actually ran.
