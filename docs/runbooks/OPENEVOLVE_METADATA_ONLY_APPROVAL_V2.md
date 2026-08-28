# Metadata-only live-mutation approval v2

Use this path only for a trusted task that implements `MetadataOnlyLiveMutationCapableTask`. The task's underlying evaluator class remains truthful; FeTA is `mri`. The only supported provider exposure is `metadata_only`.

## Standard v2.2 runtime path

Use `examples/tasks/feta_seg_evolve/openevolve-live-metadata-only-template.yaml` with the standard CLI. Replace every absolute artifact placeholder with the exact reviewed file for the run. The bridge contract template is only a starting point: its explicit provider/model, pricing and finite limits must match the approval.

```bash
auto-researcher run start \
  --task feta_seg_evolve \
  --contract examples/tasks/feta_seg_evolve/contract.yaml \
  --task-config /protected/config/openevolve-live-metadata-only.yaml \
  --run-id approved-run-id \
  --thread-id approved-thread-id \
  --checkpoint-db /protected/state/checkpoints.sqlite \
  --provenance-db /protected/state/provenance.sqlite \
  --agent-calls-db /protected/state/agent-calls.sqlite \
  --knowledge-retrievals-db /protected/state/knowledge.sqlite
```

Resume uses the same configuration and store paths. The runtime rejects relative/missing artifacts, raw credential values, non-metadata-only approval, identity drift, expired approval, a non-durable model-call store, zero live-call budget, local sandbox policy or executor/image mismatch before provider dispatch. Omitting `credential` uses the backwards-compatible `ANTHROPIC_API_KEY` environment reference. Production deployments may instead provide a value-free Google Secret Manager reference with a fully-qualified `projects/<project>/secrets/<secret>` identifier and ADC/attached workload identity. Application code never enables APIs or changes IAM; protected environment loading remains the fallback.

Resolution remains after durable dispatch ownership. Approval or executor preflight failure, completed-call replay, and failure to obtain `DISPATCHING` ownership perform zero secret accesses. The first new dispatch in an assembled runtime resolves once; subsequent calls reuse the same runtime-only `ResolvedSecret` while constructing fresh zero-retry clients. A reconstructed runtime starts with only the `SecretReference`: replay remains credential-free, while its first new dispatch resolves again and observes rotation. The value is never persisted, serialised, written to a disk cache or incorporated into scientific/model-call identity. Fake production is available only through dependency injection in offline tests, never as a CLI fallback.

## Required identities

Create the approval outside the repository after constructing the exact research contract, component specification, pinned adapter and hardened executor policy. Bind:

- `protocol_version: live-mutation-approval-v2-metadata-only`;
- run, contract, task, component and one mutable filename;
- `component_interface_hash` from the trusted component;
- `model_exposure_identity` from the exact model-facing `MutationConstraints`;
- `underlying_dataset_class` and `exposure_class: metadata_only`;
- adapter identity, executor policy hash and immutable image digest;
- provider, dated model ID, prompt ID/version and exact prompt-content hash;
- token, call, cost and expiry limits;
- reviewer identity and residual-risk acknowledgement;
- false values for underlying-data, MRI, patient-data, filesystem, network, evaluator-runtime-context, direct-provider, subprocess, retry, package, multi-file and evaluator/verifier access.

Calculate the approval with `metadata_only_approval_content_hash`. The v2 hash domain is distinct from v1. Never copy a v1 hash or approval into this path.

## Validation and preflight

```text
auto-researcher openevolve approval validate --file /protected/metadata-only-approval.yaml
auto-researcher openevolve approval inspect --file /protected/metadata-only-approval.yaml
```

Both commands are credential-free and print only allowlisted identities. Stop if the reported underlying class, exposure, expiry or hash differs from the intended run.

Before provider construction:

1. Recompute the component interface and model-exposure identities from the deployed task and compare them with the approval.
2. Verify the pinned adapter, retained executor image and network/mount/environment isolation evidence.
3. Assemble with `build_approved_live_upstream_runtime`, passing the trusted task and exact component specification.
4. Confirm the returned candidate runner is `openevolve-hardened-executor-v2`.
5. Use fresh run, thread, budget and durable model-call stores.

The provider request may contain only the current candidate source/lineage and the attested mutation constraints. Any extra runtime context, path, data directory, voxel/mask, subject/case record, prediction, checkpoint or holdout content is a stop condition. Provider output is subject to the same dynamic-content guard before candidate validation and hardened execution.

Approval does not grant access to the evaluator dataset and does not permit a local sandbox for a real live campaign.
