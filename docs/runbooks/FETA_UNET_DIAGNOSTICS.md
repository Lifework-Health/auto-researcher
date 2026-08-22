# FeTA U-Net diagnostic sidecar

## Purpose

This sidecar compares completed development-only FeTA U-Net experiments without
changing training, evaluation, search identity, or the sealed holdout boundary.
Diagnostic observations are evidence for a later scientific decision; they are
not optimisation objectives and do not constitute causal explanations.

The first delivery slice is intentionally metric-only and can be built and
tested while a GPU campaign is running. It provides:

- a deterministic hard-case panel balanced across MIAL and IRTK;
- checkpoint, dataset, split, fold, and configuration identity checks;
- per-class and reconstruction-subgroup Dice deltas;
- per-class HD95, volume-similarity and topology deltas, when present in the
  completed fold evidence, separating boundary, extent and topology signals;
- counts of material improvements, regressions, and displaced errors;
- pairwise candidate complementarity summaries;
- learning-curve summaries;
- a protected panel manifest and a separate public-safe report.

## Data boundary

The panel manifest contains development subject identifiers and belongs only in
protected runtime storage. It is created with mode `0600`. The public report
contains the panel hash and aggregate observations but no subject identifiers,
MRI paths, label paths, predictions, or checkpoint paths.

The command inspects the audited FeTA inventory without decoding label voxels,
reconstructs the locked development/holdout partition, and refuses any panel
whose cases are not development subjects.

## Metric-sidecar command

Run from a checkout containing this feature after the baseline and candidate
fold results are durable:

```bash
PYTHONPATH=src .venv/bin/python -m \
  auto_researcher.tasks.feta_unet_diagnostics.runner \
  --diagnostic-id feta-unet-v7-parent-diagnostics \
  --data-dir /absolute/path/to/feta \
  --baseline-root /protected/workspace/baseline-experiment \
  --candidate-root /protected/workspace/candidate-one \
  --candidate-root /protected/workspace/candidate-two \
  --report-dir /protected/runtime/public-diagnostic-report \
  --protected-panel /protected/runtime/private/diagnostic-panel.json \
  --panel-size 12
```

Every experiment root must contain identity-bound `fold-results/fold-*.json`
and the referenced `checkpoints/fold-*/best.pt` files. Existing non-empty report
directories and existing protected panel paths are rejected.

## Captum decision

Captum is an optional attribution backend, not a core Auto Researcher or FeTA
dependency. The current code exposes capability detection and an attribution
backend protocol but does not yet add Captum to the locked environment.

After the active campaign finishes, use one real-data CUDA smoke to validate a
pinned Captum version against the exact MONAI model/checkpoint loader. The first
attribution slice should implement a tissue-targeted scalar forward wrapper and
compare one layer-gradient method with one perturbation method:

1. Layer Grad-CAM or Guided Grad-CAM at a declared convolutional layer;
2. occlusion sensitivity using declared 3-D window and stride values;
3. a matched random-occlusion control;
4. exact checkpoint, target tissue, target region, layer, method, parameters,
   software version, and artefact hashes in the diagnostic result.

Integrated Gradients and Captum infidelity/sensitivity metrics are useful
follow-ons once the target and baseline semantics are validated. Captum must
remain absent from campaign readiness, training, evaluator, and score paths.

## Completion boundary for the first GPU smoke

The attribution slice is not ready until:

- the same checkpoint and case produce repeatable attribution summaries;
- the output is finite and aligned to native case geometry;
- gradient and occlusion evidence can be compared on the same tissue target;
- randomisation or matched perturbation checks prevent automatic causal claims;
- all subject-bearing images and manifests stay in protected runtime storage;
- the public result contains only aggregate observations and artefact hashes.
