# Bounded OpenEvolve search architecture

OpenEvolve is a search backend inside the existing `run-execution-v2` graph. It is not a second execution engine. The planner emits a generic `SearchRequest(search_type=OPENEVOLVE)`, the registry resolves the backend, and explicit graph nodes advance a versioned population. Candidate experiments still pass through the existing evaluator, transactional experiment artefact publication, verifier, reuse stores, budget accounting, checkpointing, and provenance.

The task plugin owns the scientific surface: one declared mutable Python file, its seed source, immutable function signature, schemas, permitted imports, candidate-to-`ExperimentSpec` conversion, and optional mutation context. Core owns reservations, candidate and lineage identities, validation, preparation isolation, selection, replacement, stopping, and replay behavior. Evaluator code, verifier code, datasets and splits, contracts, objectives, budgets, policies, framework code, and hidden fixtures are immutable.

The lifecycle deliberately establishes a measured baseline before mutation:

1. validate the finite search contract and initialize the generation-zero seed;
2. validate, prepare, evaluate, verify, and record that immutable seed;
3. reserve a deterministic generation-one mutation birth index;
4. obtain a complete source replacement from a deterministic or structured fake-model operator;
5. validate the source and immutable interface;
6. prepare the candidate in the configured bounded runner; untrusted/live-eligible execution requires hardened executor v2;
7. derive an `ExperimentSpec` through the task plugin;
8. use the existing evaluation and verification nodes;
9. record the outcome and update population/lineage exactly once;
10. stop deterministically or select the next parent;
11. publish and verify the search bundle transactionally.

`maximum_candidate_evaluations` includes every evaluated candidate, including the generation-zero seed. Consequently, a one-generation search that evaluates one evolved candidate requires two candidate evaluations: one seed baseline plus one evolved candidate. `SearchRequest.experiment_budget` and `ResearchContract.maximum_experiments` must fund the same total. A mutation-enabled budget of one is rejected before graph or executor construction rather than being silently increased or reinterpreted.

Each lifecycle boundary is a LangGraph checkpoint boundary. Parallel population evaluation is intentionally deferred. A future MRI task may supply a loss, augmentation, thresholding, or model-block component and its own evaluator, but it must not require graph changes or grant candidates access to MRI data, training orchestration, or verifier internals.

Hypotheses and search requests are stable lifecycle facts across every candidate
cycle. Their generic provenance event IDs are derived from their semantic keys, so
repeated recording of unchanged content is idempotent even when a caller's fallback
ID generator is prefix-stable. A changed scientific payload under the same semantic
identity still fails closed. Nested hypothesis search-space content is recursively
immutable, and candidate-specific state is recorded only through the distinct
OpenEvolve candidate and generation event identities.

Model judgement is permitted only inside a configured mutation operator that proposes the declared source replacement. The graph, identities, validation, budgets, eligibility, ranking, replacement, stopping, evaluation, and verification remain deterministic and authoritative. PR 6 uses deterministic and fake-model operators only.

Hardened executor v2 does not change graph topology. It replaces only candidate preparation: immutable host input plus one private inode-limited tmpfs, a fixed supervisor/child boundary, and a strict pipe-framed result. Preparation evidence binds executor, workspace, image and request identities so v1 results cannot be reconstructed as v2.
