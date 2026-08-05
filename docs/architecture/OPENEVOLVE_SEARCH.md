# Bounded OpenEvolve search architecture

OpenEvolve is a search backend inside the existing `run-execution-v2` graph. It is not a second execution engine. The planner emits a generic `SearchRequest(search_type=OPENEVOLVE)`, the registry resolves the backend, and explicit graph nodes advance a versioned population. Candidate experiments still pass through the existing evaluator, transactional experiment artefact publication, verifier, reuse stores, budget accounting, checkpointing, and provenance.

The task plugin owns the scientific surface: one declared mutable Python file, its seed source, immutable function signature, schemas, permitted imports, candidate-to-`ExperimentSpec` conversion, and optional mutation context. Core owns reservations, candidate and lineage identities, validation, preparation isolation, selection, replacement, stopping, and replay behavior. Evaluator code, verifier code, datasets and splits, contracts, objectives, budgets, policies, framework code, and hidden fixtures are immutable.

The lifecycle is:

1. validate the finite search contract and initialize the seed population;
2. reserve a deterministic mutation birth index;
3. obtain a complete source replacement from a deterministic or structured fake-model operator;
4. validate the source and immutable interface;
5. prepare the candidate in the bounded subprocess;
6. derive an `ExperimentSpec` through the task plugin;
7. use the existing evaluation and verification nodes;
8. record the outcome and update population/lineage exactly once;
9. stop deterministically or select the next parent;
10. publish and verify the search bundle transactionally.

Each lifecycle boundary is a LangGraph checkpoint boundary. Parallel population evaluation is intentionally deferred. A future MRI task may supply a loss, augmentation, thresholding, or model-block component and its own evaluator, but it must not require graph changes or grant candidates access to MRI data, training orchestration, or verifier internals.

Model judgement is permitted only inside a configured mutation operator that proposes the declared source replacement. The graph, identities, validation, budgets, eligibility, ranking, replacement, stopping, evaluation, and verification remain deterministic and authoritative. PR 6 uses deterministic and fake-model operators only.
