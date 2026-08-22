# FeTA U-Net deterministic ensemble sidecar

## Purpose

This sidecar evaluates whether verified, scientifically compatible U-Net
checkpoints have complementary development-fold errors. It does not change
training, candidate promotion, the best single-model result, or the sealed
holdout boundary.

The primary ensemble is a pre-specified equal-weight mean of aligned per-class
probabilities from two to four models. Models execute sequentially so ensemble
evaluation does not require their combined GPU memory. Majority voting,
checkpoint-weight averaging, unconstrained class-specific weights and a learned
stacking model are outside the primary method.

## First implementation slice

The first slice provides fail-closed contracts for:

- identical dataset, split, fold, preprocessing, label mapping, inference and
  output-class identities;
- unique verified checkpoints and configurations;
- two-to-four-member non-negative weights summing to one;
- deterministic per-class probability aggregation;
- protected, atomic `0600` probability caches bound to member and content
  hashes;
- rejection of invalid shapes, non-finite probabilities, incompatible members,
  invalid weights, overwrite attempts and corrupted caches.

The next slice will load the V4-V7 verified checkpoints, perform sequential
native-geometry inference on the locked fold-0 development subjects, calculate
individual and ensemble subject/class metrics, and emit a public-safe aggregate
report. Subject identifiers and probability tensors remain in protected runtime
storage and never enter model context.

## Reporting boundary

Every report must state the best single-model score separately from the ensemble
score. An ensemble improvement is an inference result, not evidence that any one
architecture achieved that score. The sealed holdout remains unused until an
explicitly authorised evaluation under a pre-registered selection rule.
