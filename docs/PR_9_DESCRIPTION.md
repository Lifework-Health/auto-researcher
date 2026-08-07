# PR 9: real-data Iris search benchmark

This PR adds `iris_knn@1.0`, the first real-data, non-patient benchmark for the
shared research control plane. It vendors UCI's unmodified `bezdekIris.data`
under CC BY 4.0 with its source and SHA-256, plus one immutable five-fold
stratified assignment. Runtime evaluation is fully offline.

The task-owned standard-library evaluator applies training-fold-only z-score
standardisation, weighted Minkowski distance, deterministic k-NN voting, and
mean five-fold balanced accuracy. Its output contains only bounded aggregate
metrics. Dataset rows, fold rows, and row-level predictions are prohibited
artefacts.

DIRECT, Optuna, and OpenEvolve use the same task version, configuration bounds,
evaluator identity, dataset/fold identity, objective, and verification policy.
The neutral DIRECT baseline scores `0.94`. A deterministic 20-trial Optuna run
scores `0.96`; the three-evaluation offline OpenEvolve fixture progresses from
`0.94` to `0.953333333333` to `0.96` with zero model calls. Generation zero is
the exact DIRECT baseline.

The OpenEvolve component evolves only `candidate.py`, has no imports or
dependencies, and receives a schema, feature names, objective, baseline, and
bounded aggregate history—never raw observations, labels, fold rows,
predictions, or confusion matrices. The existing hardened executor image is
unchanged; an optional Iris-specific smoke verifies the seed in the retained
approved image.

The plugin is registered through the existing task registry. No graph node,
generic search backend, evaluator-reuse mechanism, hardened executor, provider
bridge, knowledge integration, or iCCA implementation is changed.
