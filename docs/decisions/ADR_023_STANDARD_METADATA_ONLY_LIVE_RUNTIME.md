# ADR 023: Standard metadata-only live OpenEvolve runtime

## Status

Accepted for the v2.2 development line.

## Context

ADR 020 established a two-axis approval for MRI-backed evaluators whose mutation-model exposure is independently attested as metadata/code-only. The A2/A3 proof assembled the durable model bridge, pinned upstream adapter and hardened executor in bespoke Python. That proved the boundary but was not an operational runtime interface.

## Decision

The standard `run start` and `run resume` commands accept an optional top-level `openevolve_live_mutation` section. It is a value-free manifest containing absolute paths to six immutable or reviewed artifacts and an optional non-sensitive `SecretReference` for the provider credential:

- metadata-only v2 approval;
- model bridge contract;
- pinned upstream dependency lock;
- mutation prompt;
- hardened executor policy;
- executor isolation evidence.

The existing task runtime constructs the task-owned evolvable component, then assembles the approved live operator and hardened executor against the same durable SQLite model-call store used by the rest of the run. There is no alternate graph or OpenEvolve lifecycle.

Before graph execution, assembly verifies the run, contract, task, component, component interface, model exposure, boundary, adapter, prompt, provider/model, pricing identity, executor policy, image digest, mutation budgets and expiry. `DurableOpenEvolveModelBridge` repeats approval validation immediately before dispatch. The authoritative `SearchRequest` identity is bound from the mutation reservation before computing model-call identity.

Provider construction remains lazy. `anthropic` is the only production factory. An omitted reference uses the backwards-compatible `ANTHROPIC_API_KEY` environment reference; configured references use either an explicit environment variable or fully-qualified Google Secret Manager identity. The runtime retains only the non-sensitive reference until durable `DISPATCHING` ownership is acquired. The first genuine provider construction resolves one runtime-only `ResolvedSecret`; later calls in that assembled runtime reuse it while constructing fresh clients. A reconstructed runtime resolves again only for a new dispatch, naturally observing rotation, while completed-call replay performs no resolution. `fake-production` has no CLI factory and can be supplied only through the existing dependency-injection seam for offline tests. A provider or credential failure is persisted as `FAILED_BEFORE_DISPATCH`; no deterministic operator, local sandbox, direct upstream provider, retry or network/filesystem expansion is selected.

All live preparation uses `HardenedDockerExecutor`. The task YAML must name `openevolve-hardened-executor-v2`, and its exact policy and immutable image digest must match approval and isolation evidence. Assembly also inspects the current Docker server version and image configuration/digest before graph execution; environment drift therefore fails before a provider call.

## Provenance

The adapter records `LIVE_MODEL` for an approved real provider and `FAKE_MODEL` for the structured fake-production provider. The seed remains `SEED`. This distinction is derived from the immutable bridge contract, not from mutable provider output.

## Durability and restart

The durable checkpoint, provenance, model-call and knowledge stores remain separate. Metadata-only live runtime is rejected by the in-memory dependency factory. Completed calls are reused by exact identity after process restart. A dispatching or outcome-unknown call is never automatically redispatched. Resume requires the same task configuration and therefore revalidates all runtime artifacts before continuing.

## Security and scientific invariants

This change does not broaden the mutable file, model-facing schema, evaluator, verifier, FeTA dataset/split/holdout, metrics, architecture or TrainingPolicy. FeTA remains MRI-backed. Its data directory and evaluator runtime context remain host-side and do not enter the mutation request. The existing v1 synthetic/public-benchmark approval path is unchanged.

The credential reference and resolved value are operational only. Neither contributes to research-contract, search-request, candidate, experiment, evaluator/verifier, model-call semantic/replay, Research Intelligence or Research State identity. The resolved value is never persisted or serialised, and there is no disk credential cache. Application code does not enable cloud APIs or change IAM.

## Embedded search-engine capability rule

This decision productionises the current approved metadata-only runtime and existing adapter through the standard Auto Researcher lifecycle. It does not establish the current reduced mutation adapter as the long-term OpenEvolve algorithmic boundary and does not claim to provide “full OpenEvolve”. The v2.2 architectural rule is to preserve native embedded search-engine capability by default. Auto Researcher's security, provider custody, scientific evaluation, resource governance and provenance are outer-layer adapters and controls, not justification for disabling native evolutionary-search capability. Follow-on PR 11.6 / issue #53 audits and restores full-strength OpenEvolve capability parity, including native population, archive, quality-diversity and island semantics.

## Consequences

Operators must generate and review the approval, bridge contract, executor policy and isolation evidence before launch, and use absolute paths. A configuration mismatch or unavailable artifact stops setup. The command can run the full seed/evaluate/verify/select/mutate lifecycle without an A3 driver, but a real campaign still requires valid operator-issued artifacts, the pinned OpenEvolve extra, the attested Docker image/runtime, a resolvable provider-secret reference and task resources.
