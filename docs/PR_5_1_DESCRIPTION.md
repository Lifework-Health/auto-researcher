# PR 5.1: Enforce valid iCCA consensus resampling and stage safe failures

## Why

The first genuine Aura + Claude + iCCA checkpoint reached the scientific
evaluator with `r=1` and returned the deliberately sanitised
`icca_evaluation_failed: ValueError`. The plugin allowed any positive `r`, even
though `r` counts repeated consensus resamples and a single resample cannot
meaningfully estimate stability.

## Authoritative evidence

This correction was checked against the installed `auto_agent_v2` commit
`dab8c47ccdf3d5045ff5d9c76a6961b0dacd97cf`:

- `harness/evaluator/pac.py` consumes `r` as the number of resampled NMF fits and
  defines `PAC_R = 100`.
- `harness/evaluator/evaluator.py` and `harness/v2/search.py` also default to
  `r=100`.
- `docs/design/objective_full.md` resolves the production parameter to `R=100`.
- the lightweight PAC test uses `r=10`.
- there is no stricter explicit lower bound in the reference implementation.
- a local synthetic `consensus_pac(..., r=1)` completed, proving that `r=1`
  does not necessarily produce the original `ValueError`.

The task therefore enforces 10 as an executable testing floor and retains 100
as the recommended production default. Ten is not claimed to be statistically
adequate for final scientific work; such runs may require a higher value.

## What changed

- Added one authoritative iCCA resampling policy: minimum 10, default 100.
- Rejected lower values during DIRECT normalisation and Optuna fixed-context
  validation, before experiment construction or patient-data access.
- Advertised minimum, default and recommended values in the model-safe task
  context and added an explicit task limitation.
- Kept `r` fixed in Optuna; only alpha and K remain optimised.
- Added a closed evaluator failure-stage vocabulary and safe completion flags.
  Diagnostics retain only the exception class, stage, evaluator identity,
  experiment identity, canonical configuration, dataset fingerprint and four
  completion booleans. Raw messages, tracebacks, paths, credentials, patient IDs
  and patient values are discarded.
- Retained production examples at `r=100`; lightweight deterministic tests use
  `r=10`.
- Carried forward the approved live Aura compatibility corrections: Neo4j 6
  managed-transaction timeout handling and collision-free pathway/signature
  source identities. These were required by the separate Aura checkpoint and do
  not broaden the knowledge-query surface.

## Versioning

- Task version: unchanged at `1.0`.
- Task constraints version: `0.9` → `1.0`.
- Configuration schema: `1.0` → `1.1`.
- Optuna search-space identity: `auto-agent-v2-icca-v1` →
  `auto-agent-v2-icca-v2`.
- Evaluator adapter: `icca-adapter-v1` → `icca-adapter-v1.1`.
- Scientific objective and eligibility policy: unchanged.

## Diagnosis status

The original `ValueError` remains partially diagnosed. `r=1` was an invalid
scientific choice and is now impossible, but it was not proven to be the sole
runtime cause. The next checkpoint's safe stage will distinguish consensus,
eligibility, objective and surrounding adapter failures without weakening data
protection.

## Validation

- Modified-file Ruff checks: passed.
- Focused iCCA, DIRECT, Optuna, live-agent, knowledge-grounding and installed-v2
  regression set: 76 passed, 1 explicitly opt-in real-data test skipped.
- Full offline suite: 185 passed, 2 skipped. The skipped tests require explicit
  paid-Anthropic and genuine-data opt-in environment flags respectively.
- Paid Anthropic calls, Aura queries and the genuine live checkpoint were not
  run.

## Next checkpoint

Use a fresh run/thread identity and databases, `r=100`, the same explicit Claude
model and prompt versions, one DIRECT experiment, and no more than two paid model
calls. Do not reuse the failed checkpoint.
