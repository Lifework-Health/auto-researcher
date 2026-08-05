# OpenEvolve search contract

`openevolve-search-v1` binds a search request to task/component versions, a seed candidate, population size, generation and evaluation limits, wall time, model calls, failed and consecutive-failure limits, artefact bytes, mutation/selection/replacement policies, sandbox policy, evaluator and verifier identities, random seed, resume policy, stopping policy, and objective direction/threshold. Every numeric limit is finite; missing, non-finite, incoherent, unsupported, or identity-mismatched configurations fail before execution.

The candidate budget is also bounded by `ResearchContract.maximum_experiments`. Model calls are reserved before mutation. Candidate evaluation, runtime, failure, and artefact-byte counters are persisted in the population budget. Stop reasons include objective reached, generation/evaluation/model/wall/artefact budgets, failure limits, and no valid population.

The versioned records are `openevolve-candidate-v1`, `openevolve-population-v1`, `openevolve-lineage-v1`, `openevolve-sandbox-v1`, and `openevolve-provenance-v1`. Candidate identity binds the search request, immutable component interface hash, and canonical normalized source hash. Ordered fields such as parent IDs, active population, archive, lineage, and event sequence remain ordered. Set-like policy fields are sorted by the canonical JSON encoder before hashing.

Source replacement is the only PR 6 mutation format: a structured response supplies the entire content of the one task-declared mutable file and a bounded description. Patches, arbitrary file paths, dependencies, shell commands, and repository mutations are not accepted.
