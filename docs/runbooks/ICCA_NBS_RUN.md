# Run the iCCA NBS task

## Prepare the environment

Keep `auto_researcherv2.1` and `auto_agent_v2` as sibling repositories. Install
the scientific reference package into the Auto Researcher environment:

```bash
.venv/bin/pip install -e ../auto_agent_v2
```

PR 3 compatibility was verified against v2 commit
`dab8c47ccdf3d5045ff5d9c76a6961b0dacd97cf`.

## Consensus resampling policy

`r` is the number of repeated consensus-clustering resamples. The task enforces
`r >= 10` before planner reconciliation can create an experiment. Ten is the
minimum executable setting for lightweight deterministic compatibility tests;
it is not a claim of final statistical adequacy. The reference implementation's
resolved production default is `r=100`, and production examples and live
checkpoints must use that recommended setting. Optuna keeps `r` fixed and never
optimises it.

The external data directory must contain:

```text
Combined_binary_matrix.csv
Combined_clinical.csv
```

Do not copy these files into this repository.

## Configure and run

Copy `examples/tasks/icca_nbs/task.yaml` outside source control or edit its
placeholder runtime paths locally. Then run:

```bash
.venv/bin/auto-researcher tasks
.venv/bin/auto-researcher run \
  --task icca_nbs \
  --contract examples/tasks/icca_nbs/contract.yaml \
  --task-config examples/tasks/icca_nbs/task.yaml \
  --run-id icca-demo \
  --thread-id icca-demo-thread
```

An unavailable package or missing data fails during readiness before evaluator
creation. Experiment fields are persisted; runtime paths are not. The adapter
writes only aggregate JSON artefacts listed in the architecture document.

## Finite scientific results

All persisted results are standards-compliant JSON and retain
`allow_nan=False`. A finite primary stability objective is mandatory. The task
permits only three secondary readouts to be unavailable:

```text
scientific.c_index.apparent
scientific.c_index.cv
scientific.c_index.incremental
```

The installed evaluator explicitly emits `NaN` at those fields when Cox
estimation cannot be completed. The adapter stores those values as JSON `null`
and lists their schema paths in `metric_availability`; `null` means unavailable
for that evaluated configuration, never zero. Infinity and non-finite values at
all other paths fail closed. Eligibility gates remain explicit booleans and a
missing or non-finite gate input cannot become a pass.

Each experiment publishes its four JSON files as a transactional sibling
directory. All payloads are validated in memory first, each staged file is
flushed and fsynced, and one directory rename publishes the complete bundle.
The evaluator manifest records deterministic payload and bundle SHA256 values.
Identical replay is idempotent; conflicting replay is rejected. If publication
fails, the returned failure has no artefact references, so provenance cannot
point to nonexistent files.

If the evaluator fails, its external error and `failure_diagnostics` persist only
a safe exception class, a closed failure-stage value, canonical configuration,
dataset fingerprint and completion flags. Raw exception messages, tracebacks,
paths and patient-level values are never persisted. The allowed stages are
`CONFIGURATION_VALIDATION`, `DATASET_LOADING`, `NETWORK_PROPAGATION`,
`CONSENSUS_CLUSTERING`, `ELIGIBILITY_EVALUATION`, `OBJECTIVE_CALCULATION`,
`RESULT_NORMALISATION`, `ARTEFACT_WRITING` and `UNKNOWN`.

## Compatibility gates

The installed-v2 contract test uses real v2 dataclasses and objective code:

```bash
.venv/bin/pytest tests/integration/test_v2_adapter_optional.py -v
.venv/bin/pytest tests/integration/test_icca_optuna_installed_optional.py -v
```

The real-data comparison is opt-in:

```bash
AUTO_RESEARCHER_ICCA_DATA_DIR=/external/data \
AUTO_RESEARCHER_ICCA_WORKSPACE_DIR=/external/workspace \
.venv/bin/pytest tests/integration/test_icca_real_data_optional.py -v
```

Optional environment variables select network, alignment, alpha, K, and r using
the `AUTO_RESEARCHER_ICCA_*` prefix. Never report this gate as passed unless it
actually ran with external data.
