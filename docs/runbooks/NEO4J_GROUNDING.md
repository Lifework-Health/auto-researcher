# Neo4j grounding runbook

## Install and configure

Core Auto Researcher does not require Neo4j. Install the pinned optional
adapter only where grounding is needed:

```bash
.venv/bin/pip install -e '.[knowledge-neo4j]'
```

PR 5 pins `neo4j==6.2.0`. Prefer a dedicated database account with only the
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
  permitted_read_safety_modes: [PRIVILEGE_VERIFIED]
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
  read_safety:
    mode: PRIVILEGE_VERIFIED
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

`permitted_read_safety_modes` is a contract ceiling. Runtime configuration may
select only a listed mode; it cannot silently downgrade or reinterpret the
contract. The closed modes in `knowledge-read-safety-v2` are:

- `PRIVILEGE_VERIFIED`: the effective database privileges are visible and no
  write, schema or administrative privilege is present.
- `OPERATOR_ATTESTED`: a time-bounded operator attestation authorises the
  compensating query-level controls described below. This is weaker than
  database-enforced read-only access.
- `UNVERIFIED`: no read-safety assurance. It cannot satisfy required grounding.

The removed `require_verified_read_only: false` setting is rejected because it
is ambiguous. A legacy value of `true` continues to mean
`PRIVILEGE_VERIFIED` only; new configurations should use the explicit mode.

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

To independently confirm least privilege, review `SHOW USER PRIVILEGES AS
COMMANDS` as an administrator and verify no graph write, schema write or
administrative grant exists for this account. Also monitor server query logs
for the registered template hashes. Do not grant writes merely to make the
readiness check pass.

## AuraDB Professional operator attestation

Aura project `Viewer` and Aura Tool authentication are control-plane and
hosted-tool identities; they are not external Python-driver database
credentials. AuraDB Professional external driver access therefore uses a
native instance credential. When its effective privilege set cannot be
inspected or constrained to read-only, classify it honestly as
`MANAGED_INSTANCE_PRIMARY`. Never describe that credential as a read-only
identity.

`OPERATOR_ATTESTED` is available only when the research contract explicitly
permits it. Copy the credential-free example attestation to a protected
operator-controlled location outside source control, replace every placeholder,
bind it to the exact graph, schema, content version, caps and registered
template hashes, and obtain the named review. The attestation must:

- identify AuraDB Professional, the graph alias and the native credential
  class without containing a URI, username, password or token;
- be immutable, deterministically hashed, unexpired and versioned;
- list every permitted versioned template and all prohibited capabilities;
- state `DATABASE_CREDENTIAL_NOT_ENFORCED_READ_ONLY` as residual risk;
- be renewed whenever its expiry, graph identity, schema/content version,
  query caps, database name or template registry changes.

Configure it explicitly:

```yaml
grounding:
  mode: REQUIRED
  provider: neo4j
  graph_alias: cell-biology-grounding
  database: neo4j
  schema_version: knowledge-graph-auto-v0.1
  content_version: backbone-2026-06
  read_safety:
    mode: OPERATOR_ATTESTED
    attestation_file: /protected/operator/read-safety-attestation.yaml
  query_timeout_seconds: 20
  maximum_records: 100
  maximum_attempts: 2
```

Validate and inspect the credential-free record before readiness:

```bash
auto-researcher knowledge attestation validate \
  --file /protected/operator/read-safety-attestation.yaml
auto-researcher knowledge attestation inspect \
  --file /protected/operator/read-safety-attestation.yaml
```

These commands validate shape, expiry and the attestation content hash. Runtime
readiness additionally validates the configuration hash and registered template
hashes. They do not approve, sign or renew an attestation.

Both commands report:

```text
Attestation hash algorithm: canonical-json-sha256-v1
Configuration hash algorithm: canonical-json-sha256-v1
```

Generate both hashes through the current library implementation; never copy or
invent a digest. The canonical creation order is: validate the credential-free
logical payload, bind its exact provider configuration and registered template
hashes, calculate the configuration hash, seal the attestation hash, then write
stable human-readable YAML. Reordered YAML keys, alternate quoting, final
newlines and reordered values in the three unordered fields do not change the
identity.

The unordered fields are `evidence_references`,
`permitted_query_template_ids` and `prohibited_capabilities`; write them in
lexical order for readability. Other YAML lists remain ordered when their order
has procedural or scientific meaning. Duplicate unordered values, duplicate
YAML mapping keys, naive timestamps and non-finite numbers fail closed.

Attestation and configuration hashes have distinct versioned domain envelopes.
The attestation hash includes review and expiry but excludes its own hash and the
local file path. The configuration hash binds the explicit database,
graph/schema/content identity, caps, policy and exact template hashes. See ADR
014 for the complete included/excluded field boundary.

Any pre-fix attestation without both algorithm fields must be regenerated. The
safe error is `LEGACY_ATTESTATION_REGENERATION_REQUIRED`; do not add the fields
to an old file and reuse its stored digest. This affects no credential or graph
state.

In operator-attested mode the provider accepts only registered, statically
linted templates through explicit read transactions. It enforces timeout, row
and hop caps, runs the registered schema preflight, requires both data and system
update counters to be present and zero, validates the final bundle, and records
the safe execution audit. Any missing counter, unattested template, hash drift,
expired attestation or update signal fails closed. These barriers reduce risk;
they do not change the native credential's database permissions.

Use `PRIVILEGE_VERIFIED` instead whenever database-enforced least privilege is
available. Move to an Aura tier or deployment supporting dedicated database
RBAC when database-enforced read-only access is a hard requirement.

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
  account. Do not perform a test write.
- `READ_SAFETY_MODE_NOT_PERMITTED`: the runtime selected a weaker mode than the
  research contract allows.
- `ATTESTATION_INVALID`: inspect only the safe validation codes; renew or
  replace the attestation rather than bypassing the check.
- `OPERATOR_ATTESTED_WRITE_BARRIER_VIOLATION`: an update counter was missing or
  non-zero. Stop; do not retry with the same retrieval identity.
- `SCHEMA_MISMATCH`: compare the deployment with the registered schema profile.
- `CONTENT_VERSION_MISMATCH`: align reviewed contract and runtime labels.
- `QUERY_TIMEOUT` or `RESULT_LIMIT_EXCEEDED`: narrow task-owned seeds; do not
  weaken the contract.
- `BUNDLE_VALIDATION_FAILED`: inspect the safe validation summary.

Aura tiers and roles may restrict privilege introspection, query logging or
test-database creation. Project Viewer access does not establish a Python-driver
read-only identity. Live Aura validation is optional and must be explicitly
enabled outside ordinary CI. No successful live test should be reported unless
it actually ran.
