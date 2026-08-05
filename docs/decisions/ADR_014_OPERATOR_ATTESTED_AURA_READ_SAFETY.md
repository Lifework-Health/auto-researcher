# ADR 014: Operator-attested Aura read safety

Status: accepted for corrective PR 5.5.

## Context

Auto Researcher previously had a binary `require_verified_read_only` setting.
That setting could not accurately represent AuraDB Professional deployments in
which project Viewer access is available for the Aura console but external
Python-driver access still uses a native instance credential. The project role
does not become the driver identity, and the native primary credential must not
be claimed to be database-enforced read-only.

The platform still needs a controlled way to use an approved managed instance
when database privilege introspection is unavailable. The assurance must remain
explicitly weaker than database RBAC, must be bounded by the research contract,
and must fail closed if any compensating control is absent.

## Decision

`knowledge-read-safety-v2` defines three closed modes:

- `PRIVILEGE_VERIFIED` proves the effective database identity has no visible
  graph-write, schema-write or administrative capability.
- `OPERATOR_ATTESTED` accepts a reviewed, immutable and expiring attestation for
  an AuraDB Professional `NATIVE_INSTANCE_CREDENTIAL`, classified as
  `MANAGED_INSTANCE_PRIMARY`.
- `UNVERIFIED` provides no read-safety assurance and cannot satisfy required
  grounding.

The research contract lists its permitted modes. Runtime configuration may
select only a listed mode and cannot silently downgrade. Operator-attested mode
requires an attestation bound by hashes to the exact provider, graph alias,
database, schema/content versions, query caps and registered template versions.
It lists evidence references and prohibited capabilities while excluding all
credentials and connection details. Its required residual-risk code is
`DATABASE_CREDENTIAL_NOT_ENFORCED_READ_ONLY`.

Operator-attested execution is accepted only through registered, versioned,
statically linted templates; typed parameters; explicit read transactions;
timeout, row and hop caps; the registered schema preflight; mandatory zero data
and system update counters; bounded result projection; and final bundle
validation. Retrieval provenance records the mode, safe attestation identity and
hash, platform/tier/credential class, residual-risk code, executed template
identities and hashes, and zero update-counter outcomes.

Readiness and every retrieval revalidate the attestation and its configuration
binding. Expiry, hash drift, an unattested template, a missing update counter or
any update fails closed. No test write is used to establish safety.

## Alternatives considered

- Treating an Aura project Viewer as the Python-driver identity: rejected; it is
  not a database authentication identity for this path.
- Labelling the native primary credential read-only: rejected as a false
  security claim.
- Retaining `require_verified_read_only: false`: rejected because it conflates
  reviewed compensating controls with no verification.
- Relying only on `execute_read`: rejected because routing intent is not a
  database permission boundary.
- Automatically drafting, approving or renewing attestations: rejected; review
  remains an explicit operator responsibility.

## Consequences

Operator-attested mode enables a narrowly controlled AuraDB Professional path,
but it remains weaker than `PRIVILEGE_VERIFIED`. Query-level barriers can detect
and prevent application-path violations; they cannot remove write capability
from the underlying native credential. Contracts that require database-enforced
least privilege must allow only `PRIVILEGE_VERIFIED` and use an Aura tier or
deployment with suitable database RBAC.

Attestations must be securely stored, reviewed, renewed and re-bound whenever
their configuration or template registry changes. Offline tests use synthetic
drivers and records only; live credentials, patient data and provider calls are
outside corrective PR 5.5.

## Canonical identity correction

Corrective PR 5.6 retains `knowledge-read-safety-v2` semantics and identifies
both attestation and configuration hashes as `canonical-json-sha256-v1`.
Pydantic JSON-mode serialization was not a valid identity boundary because it
converted `frozenset` values into arrays using process-dependent hash order.

One recursive canonical encoder now normalises Pydantic models, dataclasses,
mappings, sequences, sets, enums, paths, dates, datetimes and JSON scalars.
Mappings are string-keyed and lexically sorted; duplicate normalised keys fail.
Lists and tuples preserve order. Sets and frozensets become arrays sorted by the
canonical JSON of each normalised element, and ambiguous duplicate canonical
elements fail. Datetimes must be timezone-aware and are encoded in UTC ISO 8601.
Floats must be finite. JSON uses UTF-8, unescaped Unicode, sorted keys, compact
separators and disallows NaN.

The attestation fields `evidence_references`,
`permitted_query_template_ids`, and `prohibited_capabilities` are semantically
unordered sets. Runtime `allowed_trust_tiers`, contract permitted modes and
template-hash mappings are also unordered for hashing. Query-plan request order,
scientific lists, procedural event order, and ordinary tuples/lists remain
ordered and identity-bearing.

Attestation hashes use the domain
`auto-researcher-read-safety-attestation`; configuration hashes use
`auto-researcher-read-safety-configuration`. Both envelopes carry hash version
`1` and schema version `knowledge-read-safety-v2`. The attestation identity
includes all immutable model fields, including review and expiry timestamps,
the configuration hash and both algorithm identifiers. It excludes only the
self-referential `attestation_hash`. The configuration identity includes the
provider, platform/tier and credential classification, explicit database,
graph/schema/content identity, execution caps, trust/confidence policy,
enabled state and exact registered template ID/hash mapping. It excludes the
attestation self-hash, validation outcomes, CLI fields and attestation file path.

No successful live attestation existed before this correction. Files lacking
either algorithm identifier fail with
`LEGACY_ATTESTATION_REGENERATION_REQUIRED`; stored pre-fix hashes are never
silently trusted or migrated. Operators must regenerate them from reviewed
logical content and the current registered template hashes.
