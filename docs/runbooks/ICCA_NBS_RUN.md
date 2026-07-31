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

If the evaluator fails, its external error and `failure_diagnostics` persist only
a safe exception class, a closed failure-stage value, canonical configuration,
dataset fingerprint and completion flags. Raw exception messages, tracebacks,
paths and patient-level values are never persisted. The allowed stages are
`CONFIGURATION_VALIDATION`, `DATASET_LOADING`, `NETWORK_PROPAGATION`,
`CONSENSUS_CLUSTERING`, `ELIGIBILITY_EVALUATION`, `OBJECTIVE_CALCULATION`,
`ARTEFACT_WRITING` and `UNKNOWN`.

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
