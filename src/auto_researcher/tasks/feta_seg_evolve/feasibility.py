"""Versioned scientific feasibility guardrails for FeTA evolution."""

from __future__ import annotations

import math

from auto_researcher.contracts.models import EvaluationResult, ResearchContract
from auto_researcher.tasks.feta_seg.metrics import LABEL_NAMES

SCIENTIFIC_FEASIBILITY_POLICY = "feta-evolve-scientific-feasibility-v1"
MAXIMUM_EMPTY_PREDICTIONS = 0
MINIMUM_PER_TISSUE_DICE = 0.5
EXPECTED_TISSUES = frozenset(LABEL_NAMES.values())

EMPTY_PREDICTION_REASON = "feta_evolve_empty_prediction_guardrail_failed"
PER_TISSUE_DICE_REASON = "feta_evolve_per_tissue_dice_guardrail_failed"
INVALID_METRICS_REASON = "feta_evolve_scientific_feasibility_metrics_invalid"
CONTRACT_MISMATCH_REASON = "feta_evolve_scientific_feasibility_contract_mismatch"


def contract_has_exact_feasibility_policy(contract: ResearchContract) -> bool:
    """Return whether the immutable contract binds the exact v1 thresholds."""

    maximum_empty = contract.constraints.get("maximum_empty_predictions")
    minimum_dice = contract.constraints.get("minimum_per_tissue_dice")
    return (
        contract.constraints.get("scientific_feasibility_policy")
        == SCIENTIFIC_FEASIBILITY_POLICY
        and type(maximum_empty) is int
        and maximum_empty == MAXIMUM_EMPTY_PREDICTIONS
        and type(minimum_dice) in {int, float}
        and not isinstance(minimum_dice, bool)
        and math.isfinite(float(minimum_dice))
        and float(minimum_dice) == MINIMUM_PER_TISSUE_DICE
    )


def scientific_feasibility_reasons(
    evaluation: EvaluationResult, contract: ResearchContract
) -> tuple[str, ...]:
    """Evaluate the contract-bound FeTA feasibility panel, failing closed."""

    if not contract_has_exact_feasibility_policy(contract):
        return (CONTRACT_MISMATCH_REASON,)

    reasons: list[str] = []
    empty_count = evaluation.metrics.get("empty_prediction_count")
    if type(empty_count) is not int or empty_count < 0:
        reasons.append(INVALID_METRICS_REASON)
    elif empty_count > MAXIMUM_EMPTY_PREDICTIONS:
        reasons.append(EMPTY_PREDICTION_REASON)

    panel = evaluation.metrics.get("per_tissue_dice")
    panel_valid = isinstance(panel, dict) and set(panel) == EXPECTED_TISSUES
    if panel_valid:
        values = tuple(panel[tissue] for tissue in sorted(EXPECTED_TISSUES))
        panel_valid = all(
            type(value) in {int, float}
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and 0 <= float(value) <= 1
            for value in values
        )
    if not panel_valid:
        if INVALID_METRICS_REASON not in reasons:
            reasons.append(INVALID_METRICS_REASON)
    elif any(float(value) < MINIMUM_PER_TISSUE_DICE for value in values):
        reasons.append(PER_TISSUE_DICE_REASON)

    return tuple(reasons)
