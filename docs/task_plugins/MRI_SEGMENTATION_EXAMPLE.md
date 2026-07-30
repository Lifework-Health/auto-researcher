# Future MRI segmentation task

MRI segmentation is an extension example, not an implemented PR 2 task. It would
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

DIRECT could execute predefined configurations. A later OPTUNA backend could
optimise hyperparameters, and a later OPENEVOLVE backend could evolve losses,
augmentation policies, model blocks, or post-processing. Those backends are
explicitly outside PR 2.

The plugin would also define data custody and artefact rules suitable for medical
images. No PyTorch, MONAI, training loop, dataset loader, or model implementation
is included here.
