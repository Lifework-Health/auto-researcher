# Upstream OpenEvolve integration

PR 7 optionally adapts OpenEvolve v0.3.2 at commit `411fb59c886c18704caaffb611e17cf9e7d824d2`. The adapter uses upstream `Program` and in-memory `ProgramDatabase` mechanics only. Auto Researcher owns requests, model reservations, source reconciliation, canonical candidate identity, safety validation, execution, scientific evaluation, verification, budgets, stopping, persistence, resume, artefacts and provenance.

Upstream controllers, evaluators, provider clients, embeddings, subprocess execution, persistence, checkpoints, telemetry prompts, network access and parallelism are never constructed. An upstream parent/ranking result is a recorded recommendation; the PR 6 constrained deterministic selection remains final. The dependency is optional and the internal backend remains the default.
