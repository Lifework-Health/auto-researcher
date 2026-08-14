# ADR 025 — OpenEvolve as a full embedded evolutionary search engine

Status: Accepted for v2.2 PR 11.6

## Context

The earlier OpenEvolve integration deliberately reduced the dependency to a
one-parent mutation proposal seam while Auto Researcher owned population,
selection, evaluation ordering, and stopping. That boundary was useful for the
A1–A3 security work, but it removed the search behavior for which OpenEvolve was
selected. A3 also exposed a second problem: four distinct generated sources
represented only two scientific TrainingPolicy configurations, yet the old path
could evaluate all four and did not reliably feed verified outcomes into later
mutation context.

The exact dependency is OpenEvolve 0.3.2, upstream commit
`411fb59c886c18704caaffb611e17cf9e7d824d2`, installed from the wheel and lock
identities recorded in
`docs/capabilities/openevolve-0.3.2-capability-manifest.yaml`. The versioned
manifest is executable: a pin change, missing classification, unsupported
adapter claim, or reintroduction of a preserved capability into the disabled
set fails tests.

## Decision

Auto Researcher owns the outer research programme. It remains authoritative for
Research State and Research Intelligence, scientific evaluator and verifier
selection, evidence, data boundaries, approvals, provider policy, Managed
Secrets, programme budgets, and final conclusions. OpenEvolve owns the bounded
inner evolutionary search: its native controller, population, archive,
best-program tracking, parent/top/diverse selection, exploration/exploitation,
MAP-Elites feature maps, islands, migration, prompt sampling, model selection,
generation history, traces, stopping, and checkpoint state remain active.

`EmbeddedOpenEvolveSearch` constructs the pinned upstream `OpenEvolve`
controller and invokes its native run lifecycle. It substitutes only boundary
services. `ResourceBrokerParallelController` retains upstream sampling,
admission, migration, database updates, and checkpoint callbacks while running
candidate evaluations through a thread executor so process-local
ResourceBroker and durable evaluator adapters remain usable. Auto Researcher
observes upstream parent, inspiration, island, feature-map, archive, and champion
decisions; it does not recompute and overwrite them.

## Scientific evaluator and identity boundary

`AutoResearcherEvaluatorAdapter` normalises generated source through the
task-owned candidate normaliser before evaluation. Source identity and canonical
scientific identity are intentionally separate. The scientific identity hashes
the task/component identity plus canonical configuration; the compatible
evaluation identity additionally includes evaluator, dataset, and code versions.
FeTA uses validated canonical `TrainingPolicy` JSON, so formatting-only or
source-level differences do not create new scientific experiments.

The adapter checks an in-flight and durable safe reuse index before expensive
execution. Concurrent duplicates wait for the first compatible result, then
return that result with `evaluation_status=REUSED`. The index contains only safe
metrics and the bounded feedback schema. It is Auto Researcher evaluation
evidence, not a competing evolutionary population.

The native database receives scalar fitness, multiple metrics, verifier and
constraint flags, resource placement, and a bounded
`auto_researcher_safe_feedback.json` artifact. Later native prompt construction
therefore sees verified scores, parent/champion deltas, safe aggregate artifact
summaries, failure classification, and canonical scientific summary. The schema
rejects paths and protected-data, patient, holdout, credential, and secret
tokens. Raw data, evaluator internals, credentials, and unrestricted artifact
paths never cross the prompt boundary.

## Models, resources, and security adaptations

Native weighted ensemble/model selection is retained by injecting approved
`NativeLLMAdapter` clients through OpenEvolve's `init_client` seam. The production
adapter invokes Auto Researcher's durable model bridge, so credentials resolve
outside OpenEvolve, calls remain at-most-once, and upstream checkpoints contain
no secret. Native prompt stochasticity and full/diff rewrite modes remain
configurable. Optional algorithmic embeddings are preserved through a bounded
adapter that accepts permitted candidate source only.

Parallel scientific evaluations acquire whole-candidate ResourceBroker leases.
Equivalent NVIDIA/FeTA GPUs remain operational placements and do not enter
scientific identity. Leases are released after evaluation; the underlying broker
and shared PostgreSQL lease store retain their PR 11.5 fencing and recovery
semantics.

Direct provider credential access, arbitrary network access, arbitrary package
installation, and unrestricted host-filesystem access remain disabled. Their
useful semantics are retained where applicable through the approved provider,
hardened executor, and scoped output adapters. The manifest records every exact
classification and justification. Per-evaluation retry/timeout behavior is
explicitly weakened: the outer wall ceiling remains, but upstream retries are
zero to preserve durable at-most-once boundaries, and an already-running Python
evaluator is not preemptively killed.

## Checkpoint and provenance ownership

OpenEvolve checkpoints are authoritative for population, archive, islands,
feature maps, evolutionary counters, programs, lineage, artifacts, and native
history. Auto Researcher does not maintain a shadow population. A versioned
search-envelope sidecar binds the outer search identity, approved immutable
configuration and limits, upstream pin, and capability-manifest version to the
output/checkpoint. Resume fails closed if that envelope changes.

Resume reads the native `last_iteration` and runs only the remaining minimum of
the OpenEvolve configuration and Auto Researcher ceiling. The safe scientific
reuse index restores compatible evaluator evidence and counts prior expensive
executions against the total evaluation ceiling. Provenance reports source and
scientific identities, parent, generation, inspirations, island, evaluator
execution/reuse, safe feedback, resource placement, search/checkpoint identity,
and the native final champion. OpenEvolve 0.3.2 does not persist Python/NumPy RNG
state; that limitation is classified as not present in the pinned upstream,
rather than claimed as restored behavior.

## Consequences and acceptance

The permanent A3 regression generates one seed plus four distinct child sources
that canonicalise to two child scientific identities. It requires exactly two
expensive child evaluations, two pre-execution reuses, valid lineage/outcomes for
all sources, and safe feedback in a later prompt.

The offline A4-like suite exercises a population larger than one, multiple
islands and generations, archive/MAP-Elites evolution, migration, safe feedback,
semantic reuse, checkpoint/resume, approved ensembles and prompt variation, and
parallel placement across three simulated equivalent GPUs. The bounded FeTA A4
template is ready for the explicit post-merge acceptance campaign. PR 11.6 does
not perform paid calls, use live secrets, require GPU/data, alter the active L4
runtime, or implement Planner v2.
