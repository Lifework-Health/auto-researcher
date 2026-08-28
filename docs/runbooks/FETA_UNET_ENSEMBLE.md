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

## Implemented evaluation

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

The evaluator loads two to four verified checkpoints, performs sequential
inference on the locked fold-0 development subjects, caches aligned softmax
probabilities in protected storage, verifies that every single-model score is
reproduced, evaluates the pre-specified all-member equal-weight ensemble and
labels every smaller equal-weight subset as exploratory. It emits a public-safe
aggregate report plus a separate protected subject-level report. Subject
identifiers and probability tensors remain in protected runtime storage and
never enter model context.

The manifest schema is `feta-unet-ensemble-run-manifest-v1`. Each member names
its experiment id and absolute checkpoint, experiment-spec and evaluation-result
paths. Evaluation fails closed on incompatible dataset, split, fold,
preprocessing, inference or label identities; altered checkpoints; holdout
access; incomplete fold-0 evidence; and failure to reproduce the recorded
single-model score within `1e-6`.

## Reporting boundary

Every report must state the best single-model score separately from the ensemble
score. An ensemble improvement is an inference result, not evidence that any one
architecture achieved that score. The sealed holdout remains unused until an
explicitly authorised evaluation under a pre-registered selection rule.
