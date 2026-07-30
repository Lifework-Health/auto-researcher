"""Adapter skeleton for the existing v2 scientific evaluator.

The v2 repository remains the owner of propagation, clustering, PAC, survival,
clinical constraints, and selection methodology. This adapter will call a
packaged v2 entry point and map its selected result:

* v2 ``selected.metrics`` -> ``EvaluationResult.metrics``
* v2 ``selected.eligibility`` -> boolean ``constraint_results``
* the registered v2 objective metric -> ``primary_score``
* v2 artefact paths/version metadata -> corresponding stable contract fields

No scientific rule is reproduced here. PR 2 must provide an explicit input
translator for datasets/configuration and select the registered objective key.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from auto_researcher.contracts.models import EvaluationResult, ExperimentSpec, ResearchContract


class V2EvaluatorAdapter:
    evaluator_id = "v2-evaluator"
    version = "adapter-skeleton-pr1"

    def __init__(
        self,
        evaluate_v2: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        self._evaluate_v2 = evaluate_v2

    @property
    def available(self) -> bool:
        return self._evaluate_v2 is not None

    def evaluate(
        self,
        experiment: ExperimentSpec,
        contract: ResearchContract,
    ) -> EvaluationResult:
        raise NotImplementedError(
            "The v2 scientific input/output translator is intentionally not implemented in PR 1"
        )
