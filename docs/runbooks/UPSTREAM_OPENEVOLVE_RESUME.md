# Upstream adapter persistence and resume

Only `upstream-openevolve-state-v1` is checkpointed: pinned identity hash, bounded counters/cursor, program IDs, recommendations, final core decisions and validated JSON metadata. No upstream object, client, callback, file, process or database is serialized. Completed bridge reservations reuse their response, while candidate preparation, evaluation and verification retain their existing identity-bound reuse rules. Terminal INSPECT remains read-only and run-execution-v2 guards remain unchanged.
