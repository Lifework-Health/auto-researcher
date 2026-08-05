# Corrective PR 5.6: canonical operator-attestation identity

## Root cause

Checkpoint 04 correctly stopped before live work because logically identical
operator attestations produced different hashes in fresh Python processes.
`ReadSafetyAttestation.model_dump(mode="json")` converted its
`prohibited_capabilities` frozenset to an array before the shared JSON helper
could canonicalise it. Array order therefore depended on Python hash
randomisation. The same risk applied to other model sets and frozensets.

## Correction

- One generic recursive encoder canonicalises models, dataclasses, mappings,
  ordered sequences, unordered sets, enums, paths, UTC datetimes and finite JSON
  scalars before hashing.
- Canonical JSON uses sorted keys, compact separators, UTF-8 Unicode and strict
  non-finite rejection.
- Duplicate stringified mapping keys, duplicate unordered inputs, naive
  datetimes and unsupported objects fail closed.
- Evidence references, permitted template IDs and prohibited capabilities are
  explicitly unordered `frozenset` fields. Lists and tuples remain ordered.
- Attestation and configuration hashes use separate versioned domain envelopes
  and identify `canonical-json-sha256-v1` in the immutable attestation.
- The attestation self-hash remains excluded from its own payload. Review/expiry
  and the configuration hash remain included intentionally.
- CLI output reports both algorithms. YAML key order, unordered-list order,
  quoting style and final newline do not affect validation.
- Pre-fix files lacking algorithm identifiers fail with
  `LEGACY_ATTESTATION_REGENERATION_REQUIRED`; no old nondeterministic hash is
  migrated or trusted.
- Readiness, retrieval identity, safe provenance and persisted round trips all
  retain the corrected hash.

## Validation

The regression suite uses six fresh Python subprocesses with different mapping,
frozenset and YAML orders and `PYTHONHASHSEED` values `1`, `2`, `7`, `41`,
`999` and `random`. Each subprocess independently recomputes both hashes and
must return the same pair. Separate-process CLI validate/inspect, readiness
drift, retrieval replay, provenance and persistence tests cover downstream use.

All work is offline. No live checkpoint attestation, credentials, Aura query,
Anthropic call, patient data or OpenEvolve execution is used.

Validation result: 299 tests passed, 2 explicitly opt-in live/genuine-data
tests skipped, and 0 failed. Ruff check and format validation pass for every
changed Python file. A credential-free, in-memory checkpoint-04-shaped fixture
with the exact three permitted template identities passed canonical attestation
and configuration validation; no live attestation file was created.
