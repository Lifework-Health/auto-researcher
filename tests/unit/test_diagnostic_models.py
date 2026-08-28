from __future__ import annotations

import pytest
from pydantic import ValidationError

from auto_researcher.diagnostics.models import (
    DiagnosticCheckpoint,
    DiagnosticExperiment,
    DiagnosticMethodSpec,
    DiagnosticPanelReference,
    DiagnosticResult,
)


def _checkpoint(experiment_id: str) -> DiagnosticCheckpoint:
    return DiagnosticCheckpoint(
        experiment_id=experiment_id,
        checkpoint_sha256s=("a" * 64,),
        architecture_identity="architecture",
        configuration_identity="b" * 64,
        best_epochs=(25,),
    )


def _panel() -> DiagnosticPanelReference:
    return DiagnosticPanelReference(
        panel_identity="c" * 64,
        dataset_manifest_hash="manifest",
        split_hash="split",
        fold_hash="fold",
        case_count=4,
        subgroup_counts={"IRTK": 2, "MIAL": 2},
        contains_case_identifiers=False,
    )


def test_diagnostic_experiment_keeps_checkpoint_identities_unique():
    with pytest.raises(
        ValidationError, match="diagnostic_checkpoint_identity_duplicate"
    ):
        DiagnosticExperiment(
            diagnostic_id="diagnostic",
            task_id="feta_unet_search",
            task_version="1.0",
            baseline=_checkpoint("same"),
            candidates=(_checkpoint("same"),),
            panel=_panel(),
            methods=(DiagnosticMethodSpec(method="error", version="v1"),),
            target_labels=(1, 2),
        )


def test_public_panel_reference_rejects_case_identifiers():
    with pytest.raises(
        ValidationError, match="diagnostic_public_panel_contains_case_identifiers"
    ):
        DiagnosticPanelReference(
            panel_identity="c" * 64,
            dataset_manifest_hash="manifest",
            split_hash="split",
            fold_hash="fold",
            case_count=4,
            subgroup_counts={"IRTK": 2, "MIAL": 2},
            contains_case_identifiers=True,
        )


def test_diagnostic_result_separates_success_from_errors():
    with pytest.raises(ValidationError, match="successful_diagnostic_result_has_error"):
        DiagnosticResult(
            diagnostic_id="diagnostic",
            success=True,
            error="unexpected",
        )
    with pytest.raises(ValidationError, match="failed_diagnostic_result_missing_error"):
        DiagnosticResult(diagnostic_id="diagnostic", success=False)
