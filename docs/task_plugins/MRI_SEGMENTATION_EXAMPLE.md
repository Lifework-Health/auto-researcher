# Future MRI segmentation task

MRI segmentation is an extension example, not an implemented PR 3 task. It would
implement the same `ResearchTask` protocol and be registered as
`mri_segmentation@1.0`; the LangGraph control plane would not change.

An experiment configuration could contain:

```yaml
architecture: unet
encoder: resnet34
learning_rate: 0.0003
batch_size: 8
loss: dice_bce
augmentation_policy: strong
```

Task-owned metrics could include validation Dice, IoU, Hausdorff distance,
inference time, calibration, and robustness across seeds. Its verification
policy could check train/validation/test separation, patient leakage, reproducible
seeds, foreground coverage, required prediction artefacts, and a performance
threshold.

DIRECT could execute predefined configurations. The generic OPTUNA backend can
optimise task-registered hyperparameters through a future MRI plugin, while a
later OPENEVOLVE backend could evolve losses,
augmentation policies, model blocks, or post-processing. Those backends are
independent of knowledge retrieval. The MRI task, training implementation, and
OPENEVOLVE support remain outside PR 5.

The same future task may implement `KnowledgeGroundingCapableTask` and register
an MRI-specific profile. Its entities could include architecture, loss
function, augmentation method, imaging modality, anatomical target,
evaluation metric, dataset and implementation paper. The task would own fixed
templates and relevance rules for those concepts and could use Neo4j or another
provider. No MRI vocabulary or branches belong in the generic knowledge
contracts or LangGraph control plane.

The plugin would also define data custody and artefact rules suitable for medical
images. No PyTorch, MONAI, training loop, dataset loader, model implementation
or MRI knowledge profile is included here.
