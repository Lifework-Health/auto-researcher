# OpenEvolve resume and idempotency runbook

Use fresh run/thread IDs and stores. Start with a task configuration whose search type is `OPENEVOLVE`; use the normal `run start`, `run resume`, and `run inspect` commands. Human approval remains mandatory when the research contract requires it.

Durable boundaries include search initialization, mutation reservation, mutation completion, candidate validation, candidate preparation, existing evaluation and verification, and population update. Resume reconstructs typed contracts, candidates, population, and lineage from the checkpoint and continues with the persisted reservation/random state. It does not infer progress from a working directory.

On interruption, issue `resume` once with the same thread and configuration. If the thread is terminal, use `inspect`; terminal resume returns the stable instruction to inspect. Duplicate or conflicting starts fail through `run-execution-v2`. `inspect` is read-only: it must not add model calls, preparation/evaluator/verifier work, artefacts, checkpoints, or provenance.

Do not delete partial evidence to force progress. Candidate preparation is reusable only when its output validates. Scientific evaluation and verification reuse only their existing identity-bound stores. A duplicate candidate source is recognized by identity and is not evaluated again. Tampered search or experiment artefacts require a fresh identity after diagnosis, not an in-place retry.
