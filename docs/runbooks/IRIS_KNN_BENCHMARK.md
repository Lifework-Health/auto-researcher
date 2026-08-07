# Run the offline Iris weighted k-NN benchmark

## Scientific identity

The `iris_knn@1.0` task uses the UCI Iris `bezdekIris.data` measurements,
vendored byte-for-byte under CC BY 4.0. The immutable dataset identity is
`uci-iris-bezdek-2023-05-22-v1`; its SHA-256 is
`0fed2a99db77ec533a62dc66894d3ec6df3b58b6a8f3cf4a6b47e4086b7f97dc`.
The fixed `iris-stratified-5fold-v1` assignment has SHA-256
`8f31b0bcb1cadea5599ceb389142acfff41c9fbb057215bf861c5af37fe3a831`.

There are 150 rows, four centimetre-valued features, and 50 observations for
each of `Iris-setosa`, `Iris-versicolor`, and `Iris-virginica`. Every fold has
30 observations: 10 from each species. Fold assignment is deterministic: rows
within each canonical 50-row class block are assigned `row-within-class mod 5`.

Each evaluation standardises a validation fold using means and population
standard deviations fitted only on the other four folds. It then applies
weighted Minkowski k-NN and reports mean five-fold balanced accuracy. Voting
ties resolve by vote count, summed neighbour distance, then canonical class
order. Only aggregate confusion counts, recalls, fold scores, and identities
are published; row-level inputs and predictions are prohibited artefacts.

## Configuration

Feature weights are four finite values in `[0.1, 4.0]`, `k` is one of
`1, 3, 5, 7, 9`, and distance power is `1` or `2`. The neutral baseline is
`feature_weights=[1, 1, 1, 1]`, `k=3`, and `distance_power=2`. The DIRECT YAML
uses the equivalent four canonical scalar weight fields because the generic
DIRECT planner treats lists as choice sets. Optuna and OpenEvolve use the same
bounds and evaluator; persisted scientific results render the four weights as
one ordered vector.

## Run all three modes

No provider credentials, network connection, or external dataset path is
needed. Start each example with fresh identities and stores:

```bash
.venv/bin/auto-researcher run start \
  --task iris_knn \
  --contract examples/tasks/iris_knn/contract.yaml \
  --task-config examples/tasks/iris_knn/direct.yaml \
  --run-id iris-direct \
  --thread-id iris-direct-thread \
  --checkpoint-db .auto-researcher/iris-direct-checkpoints.sqlite \
  --provenance-db .auto-researcher/iris-direct-provenance.sqlite
```

```bash
.venv/bin/auto-researcher run start \
  --task iris_knn \
  --contract examples/tasks/iris_knn/contract.yaml \
  --task-config examples/tasks/iris_knn/optuna.yaml \
  --run-id iris-optuna \
  --thread-id iris-optuna-thread \
  --checkpoint-db .auto-researcher/iris-optuna-checkpoints.sqlite \
  --provenance-db .auto-researcher/iris-optuna-provenance.sqlite \
  --optuna-db .auto-researcher/iris-optuna-study.sqlite
```

```bash
.venv/bin/auto-researcher run start \
  --task iris_knn \
  --contract examples/tasks/iris_knn/contract.yaml \
  --task-config examples/tasks/iris_knn/openevolve.yaml \
  --run-id iris-openevolve \
  --thread-id iris-openevolve-thread \
  --checkpoint-db .auto-researcher/iris-openevolve-checkpoints.sqlite \
  --provenance-db .auto-researcher/iris-openevolve-provenance.sqlite
```

The OpenEvolve example uses deterministic local mutation fixtures and makes
zero model calls. Candidate code receives the parameter schema, feature names,
objective, baseline, and bounded aggregate history—never measurements, labels,
fold rows, predictions, or confusion matrices.

The optional retained-image isolation smoke requires the pre-approved hardened
executor image and digest. It does not rebuild or retag Docker. Keep this gate
opt-in and run it only with the explicit retained image environment variables.
