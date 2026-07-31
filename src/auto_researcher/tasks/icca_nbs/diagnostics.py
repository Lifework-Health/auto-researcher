"""Evidence-safe failure stages for the iCCA evaluator boundary."""

from __future__ import annotations

from enum import Enum


class ICCAEvaluationFailureStage(str, Enum):
    CONFIGURATION_VALIDATION = "CONFIGURATION_VALIDATION"
    DATASET_LOADING = "DATASET_LOADING"
    NETWORK_PROPAGATION = "NETWORK_PROPAGATION"
    CONSENSUS_CLUSTERING = "CONSENSUS_CLUSTERING"
    ELIGIBILITY_EVALUATION = "ELIGIBILITY_EVALUATION"
    OBJECTIVE_CALCULATION = "OBJECTIVE_CALCULATION"
    ARTEFACT_WRITING = "ARTEFACT_WRITING"
    UNKNOWN = "UNKNOWN"


def classify_scientific_failure(exc: Exception) -> ICCAEvaluationFailureStage:
    """Classify trusted v2 traceback frames without retaining traceback content.

    The installed v2 API combines consensus and eligibility in one call. Inspecting
    only module names lets the adapter distinguish those stages while ensuring raw
    exception messages, frames, paths and patient values never enter persistence.
    Unknown implementations fail closed to ``UNKNOWN``.
    """

    saw_eligibility_frame = False
    traceback = exc.__traceback__
    while traceback is not None:
        module = str(traceback.tb_frame.f_globals.get("__name__", ""))
        if module == "harness.evaluator.pac":
            return ICCAEvaluationFailureStage.CONSENSUS_CLUSTERING
        if module == "harness.evaluator.evaluator" or module.startswith(
            (
                "harness.evaluator.clinical",
                "harness.evaluator.guards",
                "harness.evaluator.survival",
            )
        ):
            saw_eligibility_frame = True
        traceback = traceback.tb_next
    if saw_eligibility_frame:
        return ICCAEvaluationFailureStage.ELIGIBILITY_EVALUATION
    return ICCAEvaluationFailureStage.UNKNOWN
