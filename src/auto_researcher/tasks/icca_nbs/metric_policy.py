"""Finite-result policy owned by the iCCA NBS task plugin."""

from auto_researcher.tasks.scientific_json import ScientificJsonPolicy

# The reference evaluator at dab8c47 explicitly returns NaN from these three
# readouts when a Cox model cannot be estimated. They are secondary diagnostics,
# not the stability objective or an eligibility gate.
ICCA_SCIENTIFIC_JSON_POLICY = ScientificJsonPolicy(
    permitted_nan_paths=frozenset(
        {
            "scientific.c_index.apparent",
            "scientific.c_index.cv",
            "scientific.c_index.incremental",
        }
    )
)
