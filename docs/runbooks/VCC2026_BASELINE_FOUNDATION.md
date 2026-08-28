# ARC Virtual Cell 2026 baseline foundation

This fixture freezes the first submitted baseline from Viet's `vcc2026`
repository without copying challenge data, predictions or private leaderboard
material into Auto Researcher.

## Frozen identities

- Source: `https://github.com/v-iettran/vcc2026`
- Source commit: `dfb906d135f7b962350004b179107ef1101be353`
- Submitted method: B2 shared-delta transfer
- Initial leaderboard rank: 82, user-reported and not a tuning signal
- Official scorer: `cell-eval2==0.16.0`, profile `vcc2026`
- Scorer source commit: `5e64833518a6603a0301cbe28185d49c30f4a986`

## Archival verification

```bash
PYTHONPATH=src python -m auto_researcher.benchmarks.vcc2026 \
  --manifest examples/benchmarks/vcc2026/viet-b2-rank82-baseline.json \
  --checkout /absolute/path/to/vcc2026 \
  --mode archival
```

This succeeds only when the checkout and every bound small artefact match the
submitted source exactly, the B2 report has the frozen dimensions, and the
known blocker set has not drifted.

## Campaign gate

Replace `--mode archival` with `--mode campaign`. The checked-in baseline is
expected to return `PRE-RUN BLOCKED` until all of the following are true:

1. `requirements.txt` pins `cell-eval2==0.16.0` and removes `cell-eval` v1.
2. README and result notes no longer describe the VCC-2025 scorer as current.
3. D1-D7 decisions are computed from the official local harness.
4. Every required scientific guardrail is true.
5. The challenge dataset manifest and both large submission payloads are bound
   by SHA-256 outside Git.

The leaderboard receipt is advisory evidence and may remain user-reported for
internal reproduction. It must be bound before a publication claim. No
leaderboard result is consumed as an optimisation objective.
