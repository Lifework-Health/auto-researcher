# ADR 020: Two-axis live-mutation security model

## Status

Accepted for PR 10.7.

## Context

`live-mutation-approval-v1` classifies the task's evaluator dataset and permits only `synthetic` and `public_benchmark`. MRI and patient data are prohibited. That is the correct policy for v1 and cannot represent a task whose host evaluator uses MRI even when the mutation provider receives only code and bounded metadata.

Treating FeTA as `public_benchmark` would erase a material fact: its underlying evaluator is MRI-backed. Reinterpreting v1's dataset class as “content sent to the model” would also allow an old approval to acquire new authority without review.

## Decision

Live mutation has two independent axes:

1. **Underlying evaluator dataset class** records what the host-side scientific evaluator uses. The closed vocabulary includes `synthetic`, `public_benchmark`, `aura`, `genuine_icca`, `mri`, and `patient_data`.
2. **Model exposure class** records what can cross the provider boundary. PR 10.7 supports only `metadata_only`.

The existing `live-mutation-approval-v1`, its hash domain, context and task capability remain unchanged. Synthetic and public-benchmark runs continue through that path.

Sensitive evaluators require the distinct `live-mutation-approval-v2-metadata-only` contract and `MetadataOnlyOpenEvolveModelCallContext`. The approval binds:

- run, research contract, task and component identities;
- underlying dataset class and `metadata_only` exposure;
- component interface hash and exact model-facing schema/context identity;
- one mutable filename;
- pinned adapter, hardened executor policy and image digest;
- provider, exact model, prompt version and prompt-content hash;
- pricing, call/token/cost budgets, creation and expiry;
- explicit false values for underlying-data, MRI, patient-data, filesystem, network and evaluator-runtime-context access.

The model-exposure identity is a domain-separated hash of the exact `MutationConstraints` transported to the provider. A task-owned mutation context must be embedded in the parameter schema, so it is part of that hash. Runtime assembly independently recomputes the component interface and exposure identities from the trusted component specification.

Only the following dynamic content is added to the attested schema/context: the current parent candidate source and its non-sensitive lineage identifiers. Metadata-only requests use a closed, extra-forbidden model. Dynamic parent content and provider outputs are rejected if they contain paths, data-directory references, MRI/voxel/mask/patient/subject/case records, predictions, checkpoints or holdout content. Errors crossing the bridge remain fixed safe codes.

Candidate execution remains in the existing pinned hardened executor with network isolation, read-only root, sanitised environment, one mutable file and no direct provider access.

## Backwards compatibility and non-upgrade rule

V1 models and hashes are unchanged. The v2 protocol has a different schema, context type, bridge policy value and approval hash domain. A v1 approval cannot validate a metadata-only context; extra v2 fields are rejected by v1; a v2 approval cannot validate a v1 bridge contract. No default maps an old approval into v2.

## Consequences

FeTA remains classified as `mri` and may opt into `metadata_only` through a task-owned boundary declaration. A real run still requires a fresh external approval matching every identity and the verified hardened executor evidence. Any schema, context, component, prompt, executor, provider, budget or expiry change requires a new approval.

This ADR does not authorise model access to sensitive data and does not make other MRI/patient tasks eligible automatically. Each task must explicitly implement the metadata-only capability and pass runtime recomputation.
