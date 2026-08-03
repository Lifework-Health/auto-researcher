# Corrective PR 5.5: operator-attested read safety for AuraDB Professional

## Summary

This corrective PR replaces the ambiguous Neo4j read-only boolean with the
closed `knowledge-read-safety-v2` contract:

- `PRIVILEGE_VERIFIED`
- `OPERATOR_ATTESTED`
- `UNVERIFIED`

It adds an immutable, credential-free operator attestation for the specific
AuraDB Professional limitation where external Python-driver access uses a native
instance credential and database-enforced read-only status cannot be claimed.
The identity is classified honestly as `MANAGED_INSTANCE_PRIMARY`, and the
attestation always records the residual risk that the database credential is not
enforced read-only.

The evidence reviewed was the stopped checkpoint 04 readiness result and the
bounded Aura identity probe: Aura project Viewer with Tool authentication
worked only inside Aura-hosted tools, did not authenticate through the official
Python driver, and external-driver examples continued to require a native
instance credential. No patient query or write probe was used.

## Enforcement

- Research contracts declare permitted modes; runtime configuration cannot
  silently weaken them.
- Required grounding rejects `UNVERIFIED`.
- Operator attestations are schema-validated, expiring and hash-bound to graph,
  database, schema/content versions, query caps and registered template hashes.
- Operator-attested execution accepts only registered and attested templates in
  explicit read transactions, with static linting, timeouts, row/hop caps,
  schema preflight, mandatory zero data/system update counters and final bundle
  validation.
- Readiness and retrieval both revalidate the attestation. Missing counters,
  drift, expiry, unregistered templates and any update signal fail closed.
- Safe provenance records mode and attestation identities, template identities
  and hashes, residual-risk classification and zero-counter results without
  credentials, connection details, raw Cypher or graph rows.
- CLI commands validate and safely inspect an existing attestation; they do not
  approve or renew it.

## Documentation

- ADR 014 records the assurance boundary and upgrade path to database RBAC.
- The Neo4j grounding runbook documents configuration, validation, renewal,
  failure handling and the Aura project Viewer/Python-driver distinction.
- Credential-free example configurations cover both privilege-verified and
  operator-attested modes.

## Validation

All validation for this PR is offline. The complete suite passed with 271 tests
passed, 2 explicitly optional live/real-data tests skipped, and 0 failures. Unit
and integration coverage includes
valid, missing, expired, drifted and malformed attestations; closed vocabulary;
contract ceilings; template allowlisting; query-level write barriers; safe
provenance; replay; optional ungrounded continuation; and existing regression
suites. No live Aura credentials, Anthropic calls, patient data, test writes or
OpenEvolve work are used.

## Checkpoint 04 after merge

Prepare a fresh, securely stored and operator-reviewed attestation bound to the
approved graph/configuration and the four registered iCCA templates. Configure
the checkpoint contract with required grounding and an explicit ceiling:

```yaml
grounding:
  mode: REQUIRED
  permitted_read_safety_modes:
    - PRIVILEGE_VERIFIED
    - OPERATOR_ATTESTED
```

Select `OPERATOR_ATTESTED` in the task configuration and reference the protected
attestation file. Run attestation validation and Neo4j readiness before starting
the fresh checkpoint identities. Readiness must report
`privilege_verified=false`, `attestation_valid=true` and residual risk
`DATABASE_CREDENTIAL_NOT_ENFORCED_READ_ONLY`. If any check fails, checkpoint 04
must stop without model or evaluator calls and without a workaround.

No credential in this PR is claimed to be database-enforced read-only.
