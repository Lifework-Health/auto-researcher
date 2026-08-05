# Corrective PR 5.7: checkpoint reconstruction for operator-attested runs

## Root cause

PRs 5.5 and 5.6 added `ReadSafetyMode` to the checkpointed
`ResearchContract.grounding.permitted_read_safety_modes` field, but the enum was
not added to the explicit LangGraph serializer allowlist. A fresh process could
therefore decode `OPERATOR_ATTESTED` as a loose string. That changed the
canonical contract identity and caused terminal INSPECT to fail with
`checkpoint_execution_identity_conflict`.

## Change

- Allowlist exactly
  `auto_researcher.contracts.enums.ReadSafetyMode`.
- Require exact stored `RunExecutionIdentity`, `ResearchContract`, and
  `ReadSafetyMode` types during reconstruction; do not accept untyped mappings
  or string substitutions.
- Canonicalize model-owned frozensets before Pydantic can flatten them in
  process-dependent hash iteration order. Closed enum sets use declaration
  order and primitive sets use canonical lexical order, preserving checkpoint
  04c's stored contract hash across fresh processes.
- Add a checkpoint-04c-shaped, REQUIRED-grounding, operator-attested,
  REAL/REFUTED terminal fixture.
- Serialize the fixture, close it, and reconstruct it in a fresh process.
- Verify read-only INSPECT, stable hashes and identities, duplicate/conflicting
  START codes, terminal RESUME behavior, and zero dependency or persistence
  side effects.
- Add negative coverage for unknown enums, arbitrary classes/callables/modules,
  malformed qualified types, invalid enum values, and unallowlisted subclasses.

No other PR 5.5/5.6 type is allowlisted. `ReadSafetyAttestation` and its
platform, tier, credential, capability, and residual-risk enums are provider
configuration persisted as validated JSON in the separate retrieval store, not
as LangGraph checkpoint objects.

## Compatibility and safety

The change does not modify run-execution-v2, knowledge-read-safety-v2,
canonical-json-sha256-v1, provenance-events-v2, evaluation-reuse-v2,
scientific-json-v1, experiment-bundle-v2, iCCA logic, prompts, or verification
policy. Existing operator-attested checkpoints retain their hashes and can be
inspected after deployment; they do not require a fresh run identity.

Validation is entirely offline. Aura, Anthropic, patient data, OpenEvolve, and
the live checkpoint 04c stores are not accessed or mutated by the test suite.
