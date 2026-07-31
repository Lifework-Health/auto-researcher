# Corrective PR 5.2: Finite scientific results and transactional artefact bundles

## Why

The second Aura + Claude + genuine iCCA checkpoint completed dataset loading,
network propagation, consensus clustering with `r=100`, eligibility evaluation,
objective calculation and result mapping, then failed safely at
`ARTEFACT_WRITING` with `ValueError`. Only `experiment_spec.json` had been
published, while the returned failure still advertised all four intended
artefact references.

The exact non-finite field in that live run was not retained and therefore is
not claimed as identified. Inspection of the installed `auto_agent_v2` commit
`dab8c47ccdf3d5045ff5d9c76a6961b0dacd97cf` confirms that its three C-index
readouts explicitly return `NaN` when Cox estimation is unavailable. A
patient-free, genuine-shaped fixture containing those unavailable C-index
values reproduces the former strict-JSON failure surface. The strongest
supported diagnosis is therefore that a legitimate unavailable aggregate
readout reached the first `allow_nan=False` validation point during sequential
file writing. The sequential writer committed the first file before reaching
the invalid later payload, and references were computed before publication.

## Scientific JSON policy

- Added a generic recursive normaliser for Python and NumPy numbers, NumPy
  arrays, mappings, lists/tuples, Pydantic models, dataclasses and already
  accepted pandas-like conversion protocols.
- Finite values are preserved. Positive and negative infinity always fail
  closed. Unknown NaN values fail closed.
- A successful `EvaluationResult` now requires a finite primary score, strict
  standards-compliant metrics and explicit boolean constraints before it can
  enter graph state.
- Non-finite primary objective values fail at `OBJECTIVE_CALCULATION`; secondary
  mapping failures use the new safe `RESULT_NORMALISATION` stage.
- Optuna persists already validated evaluation and verification contracts; it
  no longer applies a blanket non-finite-to-null conversion.

The iCCA plugin owns the only permitted unavailable paths:

```text
scientific.c_index.apparent
scientific.c_index.cv
scientific.c_index.incremental
```

An allowed NaN becomes JSON `null`, never zero, a string or an imputed value.
`metric_availability` records the exact safe schema paths, count and
`null_for_unavailable_non_finite_v1` encoding. No confidence limits, clinical
p-values, per-cluster values, identifiers, configuration values, counts,
eligibility inputs or unknown paths are allowlisted because they are not
present as legitimate non-finite values in the mapped reference result.

## Transactional artefact publication

The experiment writer now:

1. validates and serialises all four payloads in memory with `allow_nan=False`;
2. creates a temporary sibling directory;
3. writes, flushes and fsyncs every staged file;
4. verifies the exact four-file set;
5. fsyncs the staged directory;
6. publishes the complete directory with one same-filesystem rename; and
7. removes staging content on every handled failure.

The evaluator manifest contains the schema version, expected filenames,
result-encoding version, SHA256 for each logical payload, bundle SHA256 and a
completed flag. The verifier can deterministically detect missing files or
payload tampering. Identical byte-for-byte replay is idempotent. A partial or
different existing bundle is a conflict and is never overwritten.

Intended references are returned only after successful publication. If
success-bundle or failure-bundle persistence fails, the returned failure uses
an empty reference tuple and a safe `bundle_publication_failed` code. Provenance
therefore never receives a path to a missing or partial bundle.

## Versioning

- iCCA adapter: `icca-adapter-v1.1` → `icca-adapter-v1.2`
- result encoding: `scientific-json-v1`
- experiment bundle: `experiment-bundle-v2`
- iCCA task version: unchanged at `1.0`
- iCCA configuration schema: unchanged at `1.1`
- task constraints: unchanged at `1.0`
- scientific objective and eligibility policy: unchanged

The result and bundle versions are included in experiment code identity, so old
unsanitised results and old sequential publication semantics cannot replay as
the corrected format. The synthetic reference task uses the same bundle
identity and bumps its evaluator implementation identity to
`synthetic-landscape-v3`.

## Validation

- Modified-file Ruff checks and formatting: passed.
- Full offline suite: **213 passed, 2 skipped, 0 failed**.
- Installed-v2 dataclass/objective and iCCA Optuna compatibility tests: passed.
- DIRECT, Optuna, fake live-agent, knowledge-grounding, replay, security,
  transactional fault-injection and provenance regressions: passed.
- Paid Anthropic smoke test: skipped by explicit opt-in guard.
- Genuine patient-data test: skipped by explicit opt-in guard.
- Anthropic was not called, Aura was not queried, the genuine checkpoint was
  not run, and OpenEvolve work was not started.

## Next live checkpoint

Use a fresh run, thread and persistence identity with the same pinned v2 commit,
`r=100`, explicit Claude model/prompt versions and one DIRECT experiment. First
confirm the result bundle is complete and passes hash verification, then inspect
only aggregate `metric_availability` paths. Do not reuse or overwrite the
partial checkpoint directory.
