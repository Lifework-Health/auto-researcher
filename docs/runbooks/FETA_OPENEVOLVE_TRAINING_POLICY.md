# FeTA bounded TrainingPolicy OpenEvolve runbook

`feta_seg_evolve@1.0` is a sibling of the active `feta_seg_search@1.0` campaign. It evolves one metadata-only Python function whose complete output must validate as `TrainingPolicy@feta-training-policy-v1`. The host retains the FeTA dataset, split, SegResNet, preprocessing, training loop, evaluator, verifier, GPU admission and holdout boundary.

## Pure and hybrid runs

Pure OpenEvolve uses the registered base configuration and seed policy in `openevolve-deterministic-smoke.yaml`. A hybrid run uses `openevolve-hybrid-template.yaml`; replace only `runtime.options.base_configuration` with the selected bounded Optuna configuration. This inherits configuration values, not checkpoints. The recorded `seeding_mode`, base identity, policy identity, candidate source hash and OpenEvolve lineage distinguish pure and hybrid runs.

The budget includes generation zero. Three candidate evaluations mean one seed plus two evolved candidates. `maximum_candidate_evaluations`, the `SearchRequest` experiment budget and the contract's `maximum_experiments` must agree.

## GPU and resource expectations

Candidate source preparation is CPU-only and sandboxed. Each scientific evaluation is a full fold-0 SegResNet training run. The production template uses courteous primary admission on physical GPU 0 and permits only 25/50/100 epoch fidelities. Set `CUDA_VISIBLE_DEVICES=0`; training uses logical `cuda:0`. Use separate output/checkpoint databases and do not point this task at an active Optuna run's output directory. The deterministic preprocessing cache may safely share a workspace because population is protected by `flock` and the cache identity is unchanged.

## Live-mutation security status

Deterministic and injected fake-production mutation remain supported. PR 10.7 adds a distinct approved live path: FeTA declares its underlying evaluator class as `mri` and its model exposure as `metadata_only`. It does not implement or inherit the v1 `live_mutation_dataset_class`, and must never be relabelled as `public_benchmark`.

A live run requires a fresh external `live-mutation-approval-v2-metadata-only` binding the exact run, contract, component interface, model-facing schema/context identity, mutable file, adapter, hardened executor and image, provider/model/prompt hash, finite budget and expiry. MRI, patient data, underlying data, filesystem, network and evaluator runtime context access must all be false. An old v1 approval cannot be used.

The provider receives the current candidate source plus the exact attested `MutationConstraints`: interface, parameter/output schemas, bounded context, imports/dependencies and source cap. It receives no FeTA directory, MRI, masks, subject/case rows, predictions, checkpoints, holdout information or evaluator runtime context. Nested input tampering and unsafe candidate output fail before evaluation.

## Preflight

1. Confirm the deployed commit and `feta_seg_search@1.0` regression tests.
2. Verify the exact FeTA manifest, development split and sealed holdout.
3. Select pure or hybrid configuration; for hybrid, copy and independently verify the Optuna configuration only.
4. Confirm `maximum_candidate_evaluations` includes the seed and matches the contract.
5. Use fresh run/thread IDs, SQLite stores and output directory.
6. Set the one-device CUDA binding and verify the scheduler's physical GPU index.
7. Run static configuration/component tests on the server before any training.
8. Start with 25 epochs and deterministic mutation; inspect seed/evolved lineage and aggregate metrics before increasing the budget.
9. For deterministic runs, do not configure a provider or approval. For live mutation, validate a fresh v2 metadata-only approval and independently match its component interface, exposure, prompt, adapter and executor identities before constructing the provider.
10. Confirm the runtime uses `openevolve-hardened-executor-v2`; local sandbox preparation is not approved for a real live-model campaign.

Example command after replacing the data path:

```shell
CUDA_VISIBLE_DEVICES=0 auto-researcher run start \
  --task feta_seg_evolve \
  --contract examples/tasks/feta_seg_evolve/contract.yaml \
  --task-config examples/tasks/feta_seg_evolve/openevolve-production-template.yaml \
  --run-id feta-evolve-001 \
  --thread-id feta-evolve-thread-001 \
  --checkpoint-db /protected/feta-evolve-checkpoints.sqlite \
  --provenance-db /protected/feta-evolve-provenance.sqlite \
  --agent-calls-db /protected/feta-evolve-agent-calls.sqlite \
  --knowledge-retrievals-db /protected/feta-evolve-knowledge.sqlite
```
