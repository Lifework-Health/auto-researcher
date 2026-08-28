# FeTA U-Net V9 preparation

V9 is a 36-hour, fold-0 development campaign with an eight-hour protected
finalisation reserve. The scientific objective, dataset manifest, split, fold,
preprocessing and sealed-holdout policy remain unchanged.

## Why V9 is different

- Seventy percent of the provisional search budget exploits the V8 DynUNet
  lineage. Two attention-gated U-Net roots receive twenty percent, while fixed
  UNETR and SwinUNETR feasibility pilots share the remaining ten percent.
- The Research Director owns strategic intent and course correction. It receives
  the verified V8 campaign and ensemble record plus a frozen, cited knowledge
  library and literature brief. It cannot directly create experiments.
- The deterministic Planner compiles Director intent into registered parameters,
  exact operator allocations and typed search requests. The Supervisor remains
  the hard identity, evidence, memory, time and cost gate.
- The Director has sixteen successfully validated decision slots. Invalid
  structured calls remain bounded by attempts, cost and the global model-call
  limit but do not consume the valid-decision cap.
- Finalist selection is intended to consider macro Dice, external-CSF and
  grey-matter Dice, reconstruction gap, lineage diversity and ensemble marginal
  gain. Macro Dice remains the scientific optimisation objective.

## Frozen evidence

`examples/tasks/feta_unet_search/v9-bound-evidence.json` records the verified V8
standalone champion (0.8260411), the primary four-model ensemble (0.8303180),
the weak tissue classes and the reconstruction gap. It also binds the exact
knowledge-library and literature-brief identities. No sealed-holdout evaluation
was used.

The reviewed literature corpus contains four primary sources: Attention U-Net,
UNETR, Swin UNETR and BOHB. Each card states its transfer limitations. Literature
is an advisory prior, not evidence that a method works on FeTA and never an
instruction channel.

## Hard launch boundary

The checked-in template is deliberately not launchable. A production action
preflight must remain blocked until all of the following are true:

1. The V9 adaptive portfolio controller is implemented and deterministically
   replayed across every frozen stage and recovery boundary.
2. The 15-to-30-to-50-to-100-to-150 continuation ladder has tests that restore
   model, optimiser, scheduler, scaler and RNG state.
3. The V8 champion and alternate checkpoints are imported read-only and bound
   to their source manifests and scientific identities.
4. Runtime paths, production contract, deadline, GPU and credential reference
   are frozen; the control directory is fresh; every source hash matches.

Run the zero-model-call static gate with:

```bash
python -m auto_researcher.tasks.feta_unet_search.v9_preflight \
  --config examples/tasks/feta_unet_search/campaign-36h-v9-template.yaml \
  --evidence examples/tasks/feta_unet_search/v9-bound-evidence.json
```

Expected state today: `launch_ready: false`. This is a successful preparation
result, not permission to train.

Real-CUDA calibration on an otherwise idle production A6000 completed one AMP
forward/backward/optimizer step for all four new pilots without touching the
holdout or making model calls. Peak allocated memory was 3.75 and 4.42 GiB for
the two AttentionUnet roots, 2.75 GiB for UNETR and 8.55 GiB for SwinUNETR;
all are well within the 44 GiB campaign ceiling. The measured evidence and its
hash are bound in `v9-cuda-calibration.json` and the campaign template.

The transformer pilots require the pinned `einops==0.8.1` dependency included
in the `feta` optional environment. Dependency presence is part of the CUDA
preflight; the runtime must not silently fall back when it is absent.
