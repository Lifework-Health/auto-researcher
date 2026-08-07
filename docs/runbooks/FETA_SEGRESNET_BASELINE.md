# FeTA SegResNet baseline

This runbook describes the locked `feta_seg@1.0` development-only baseline. It does not contain HPO, OpenEvolve, live model calls, or hold-out evaluation.

## Local data and identity

Provide the FeTA root at runtime with `--data-dir`; no absolute path enters the scientific identity. The audited local FeTA 2.1/2022 export is a flat `mri_gz/` directory with 80 usable T2/segmentation pairs: 40 MIAL and 40 IRTK. Subject 080 has equivalent compressed and uncompressed image containers; the loader selects `.nii.gz`. JSON clinical metadata are absent, so gestational age and pathology cannot be stratified or reported.

The task hashes every selected image and mask and records shape, spacing, reconstruction method, labels, loader version and a path-free canonical manifest hash. Raw NIfTI files, MONAI caches, workspaces and checkpoints are ignored by git.

The audited export manifest hash is `6d6f375fda99512a93bbaaa715d6edb5031c4d4f2356584b578f2ebd9631eacf`.

## Locked partitions

- Split: `feta-development-holdout-v1`, seed `20260807`, hash `3ee6e9f02b4d35f7611bb70cdf19aea3ebc12f81ef89b57291eac9983df66561`.
- Hold-out: 12 subjects, 6 MIAL and 6 IRTK. Labels are sealed and never evaluated by this task.
- Development: 68 subjects, 34 MIAL and 34 IRTK.
- Folds: `feta-dev-5fold-v1`, hash `45e70dc010448d124b978a8becdc5866264b457c3b5ffddc802916f30ec28f6e`.
- Fold validation sizes/method counts: `14 (7/7), 14 (7/7), 14 (7/7), 14 (7/7), 12 (6/6)`.

## Fixed science

MONAI SegResNet uses 3D input, one image channel, eight output channels, 32 initial filters, down blocks `(1,2,2,4)`, up blocks `(1,1,1)`, group normalisation, ReLU, explicit `deconv` upsampling and dropout 0.2. Preprocessing is RAS, 0.5-mm isotropic spacing, linear image/nearest-label resampling, non-zero foreground crop and foreground z-score normalisation. Training uses two 128³ 1:1 positive/negative patches per volume, axis flips at probability 0.2, intensity scale/shift ±0.1 at probability 0.2, batch size one and CUDA AMP.

Loss is equal-weight MONAI DiceCE (`softmax`, one-hot target, no background). Optimisation is AdamW, learning rate `1e-4`, weight decay `1e-5`, 300 epochs, validation every five epochs and best-fold checkpoint selection. Seeds are `20260807 + fold`. Whole-volume validation uses 128³ Gaussian sliding windows, 0.5 overlap and `sw_batch_size=1`.

The primary metric is the mean of 68 subject-level macro Dice values over labels 1–7. Background is excluded. Every audited subject contains all seven tissues, so absent reference labels are an integrity failure. Secondary outputs include per-tissue Dice/HD95, fold results, best epochs and MIAL/IRTK aggregate scores and gap.

## Commands and outputs

Install the optional environment with `pip install -e '.[feta]'`. Generate/inspect the manifest through the task using the local data directory. Run the non-scientific generated-data smoke with `examples/tasks/feta_seg/smoke.yaml`; its identity cannot be reused as full evidence. Run the five-fold configuration with `examples/tasks/feta_seg/contract.yaml` and `baseline.yaml`, passing the local data root and a CUDA-capable runtime.

Outputs live below the configured `.auto-researcher` directory. Best `.pt` checkpoints remain outside git and are referenced by relative path, size and SHA-256. Generic artefacts contain aggregate evidence only, never MRI voxels or segmentation volumes.

## Current execution status

The optional mechanics were tested with Python 3.12.0, PyTorch 2.8.0, MONAI 1.5.1, NumPy 2.0.2 and nibabel 5.3.3. The development machine is macOS/arm64 with an Apple M5 GPU and no CUDA. Therefore the genuine five-fold baseline is **FULL BASELINE BLOCKED** with `feta_cuda_unavailable_for_full_baseline`. A smoke result is not scientific evidence. PR 11 must not begin until all five locked folds have completed on CUDA, verified, and recorded with zero hold-out calls.
