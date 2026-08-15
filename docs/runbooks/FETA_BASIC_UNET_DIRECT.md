# Frozen FeTA BasicUNet DIRECT baseline

This built-in is `feta_unet_direct@1.0`. It does not change or reinterpret
`feta_seg@1.0`.

## Frozen identities

- Scientific: `feta-development-oof-subject-macro-dice-v1`
- Architecture: `monai-basic-unet-3d-v1`
- Evaluator: `feta-basic-unet-direct-evaluator-v1`
- Smoke runner: `feta-basic-unet-engineering-smoke-runner-v1`
- Baseline runner: `feta-basic-unet-five-fold-oof-runner-v1`
- Result: `feta-basic-unet-direct-result-v1`
- Loss: `dice-ce-softmax-onehot-no-background-equal-v1`
- Optimiser: `adamw-lr1e-4-wd1e-5-v1`
- Inference: `sliding-window-128-overlap0.5-gaussian-native-restore-v2`

The model is MONAI `BasicUNet` with 5,749,608 trainable parameters. The
recorded acceptance probe is `1x1x128x128x128` to `1x8x128x128x128`, with
2,338 MiB peak CUDA allocated and 3,076 MiB peak CUDA reserved. A full
forward/loss/backward/AdamW step passed under a 20,638 MiB allocator ceiling.

## Runtime preparation

Copy the applicable example YAML outside the repository and replace every
placeholder with an absolute path. `data_dir` and `workspace_dir` must be
access-controlled. `output_dir` must be a distinct protected destination for
the identifier-free result bundle. The runner creates cache, checkpoints and
fold state only below `workspace_dir`.

The smoke uses one locked fold-0 training subject and one locked fold-0
validation subject for one epoch. It exercises training, validation, a hashed
checkpoint and the complete metric panel but is marked non-baseline. The
baseline uses all five fixed development folds and reports aggregate OOF Dice;
the holdout remains sealed in both profiles.

## Standard runtime commands

Engineering smoke:

```bash
auto-researcher run start \
  --task feta_unet_direct \
  --contract examples/tasks/feta_unet_direct/contract.yaml \
  --task-config /absolute/protected/config/engineering-smoke.yaml \
  --run-id feta-unet-smoke-001 \
  --thread-id feta-unet-smoke-001 \
  --checkpoint-db /absolute/protected/control/smoke-checkpoints.sqlite \
  --provenance-db /absolute/protected/control/smoke-provenance.sqlite \
  --agent-calls-db /absolute/protected/control/smoke-agent-calls.sqlite \
  --knowledge-retrievals-db /absolute/protected/control/smoke-knowledge.sqlite
```

Frozen baseline:

```bash
auto-researcher run start \
  --task feta_unet_direct \
  --contract examples/tasks/feta_unet_direct/contract.yaml \
  --task-config /absolute/protected/config/frozen-baseline.yaml \
  --run-id feta-unet-baseline-001 \
  --thread-id feta-unet-baseline-001 \
  --checkpoint-db /absolute/protected/control/baseline-checkpoints.sqlite \
  --provenance-db /absolute/protected/control/baseline-provenance.sqlite \
  --agent-calls-db /absolute/protected/control/baseline-agent-calls.sqlite \
  --knowledge-retrievals-db /absolute/protected/control/baseline-knowledge.sqlite
```

Completed folds are identity-bound to the manifest, split, fold assignments,
configuration, architecture, seed and validation membership. Reusing them in a
new run requires `runtime.options.resume_root` to point to the prior protected
experiment root. Checkpoint size and SHA-256 are verified before import.

## Storage planning

Allow approximately 70 MiB per active AdamW checkpoint (about 350 MiB for five
folds), plus 10–25 GiB for the MONAI deterministic preprocessing cache,
temporary validation tensors and fold-state headroom. The smoke typically needs
under 1 GiB beyond the source dataset. Site-specific volume geometry and cache
serialization can change these estimates; provision at least 30 GiB protected
workspace capacity for the baseline. Identifier-free JSON output and control
databases are normally under 100 MiB.
