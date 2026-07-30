from __future__ import annotations

import importlib.util

import pytest

from auto_researcher.evaluation.v2_adapter import V2EvaluatorAdapter


@pytest.mark.v2
def test_v2_adapter_boundary_when_reference_package_is_available():
    if importlib.util.find_spec("harness") is None:
        pytest.skip("auto_agent_v2 is not installed; adapter remains an optional boundary")
    from harness.evaluator.evaluator import evaluate

    adapter = V2EvaluatorAdapter(evaluate)
    assert adapter.available is True
    assert adapter.evaluator_id == "v2-evaluator"
