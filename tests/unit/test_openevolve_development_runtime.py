from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from auto_researcher.agents.models import (
    ModelCallConfig,
    ModelPricing,
    StructuredModelResponse,
)
from auto_researcher.search.openevolve.backend import OpenEvolveBackend
from auto_researcher.search.openevolve.development_runtime import (
    DevelopmentLiveOpenEvolveConfiguration,
    DevelopmentStructuredMutationClient,
)
from auto_researcher.search.openevolve.models import CandidateOutcome, CandidateStatus
from auto_researcher.search.openevolve.upstream_models import UpstreamMutationEnvelope
from auto_researcher.secrets import SecretProviderKind, SecretReference


def _model_configuration() -> ModelCallConfig:
    return ModelCallConfig(
        provider="anthropic",
        model_id="claude-sonnet-4-5-20250929",
        temperature=0.2,
        maximum_output_tokens=4096,
        timeout_seconds=180,
        maximum_attempts=1,
        maximum_cost_per_call=0.2,
        prompt_version="openevolve-mutation-prompt-v2",
        pricing=ModelPricing(
            version="test-pricing-v1",
            input_cost_per_million_tokens=3,
            output_cost_per_million_tokens=15,
            currency="USD",
        ),
    )


def _configuration(tmp_path) -> DevelopmentLiveOpenEvolveConfiguration:
    return DevelopmentLiveOpenEvolveConfiguration(
        acknowledgement="public-data-non-production-development",
        model=_model_configuration(),
        credential=SecretReference(
            logical_name="anthropic_api_key",
            provider=SecretProviderKind.ENVIRONMENT,
            provider_identifier="ANTHROPIC_API_KEY",
        ),
        maximum_total_cost_usd=0.4,
        usage_log_file=tmp_path / "usage.jsonl",
    )


class _FakeStructuredModelClient:
    provider = "anthropic"
    model_id = "claude-sonnet-4-5-20250929"

    def generate_structured(self, **kwargs):
        envelope = UpstreamMutationEnvelope(
            mutable_file="candidate.py",
            source=(
                "def evolve(configuration):\n"
                "    return {**configuration, 'learning_rate': 0.0002}\n"
            ),
            description="Increase the learning rate within the bounded policy.",
        )
        return StructuredModelResponse(
            call_id=kwargs["call_id"],
            provider=self.provider,
            model_id=self.model_id,
            structured_output=envelope.model_dump(mode="json"),
            input_tokens=100,
            output_tokens=50,
            estimated_cost=0.00105,
            latency_ms=10,
            prompt_version=kwargs["call_config"].prompt_version,
            context_hash=kwargs["context_hash"],
            response_hash="a" * 64,
        )


def test_development_client_is_two_call_bounded_and_logs_only_safe_usage(tmp_path):
    configuration = _configuration(tmp_path)
    client = DevelopmentStructuredMutationClient(
        configuration,
        model_client=_FakeStructuredModelClient(),
    )
    request = {"protocol": "test", "parent": {"code": "public source"}}

    first = client.propose_mutation(request)
    second = client.propose_mutation(request)

    assert first["mutable_file"] == second["mutable_file"] == "candidate.py"
    assert client.calls_used == 2
    assert client.total_cost == pytest.approx(0.0021)
    with pytest.raises(ValueError, match="development_model_call_budget_exhausted"):
        client.propose_mutation(request)

    text = configuration.usage_log_file.read_text()
    assert "public source" not in text
    assert "def evolve" not in text
    rows = [json.loads(line) for line in text.splitlines()]
    assert [row["status"] for row in rows] == ["COMPLETED", "COMPLETED"]


def test_development_configuration_rejects_budget_drift(tmp_path):
    payload = _configuration(tmp_path).model_dump(mode="python")
    payload["maximum_total_cost_usd"] = 0.3
    with pytest.raises(
        ValidationError,
        match="development_live_mutation_configuration_invalid",
    ):
        DevelopmentLiveOpenEvolveConfiguration.model_validate(payload)


def test_development_parent_relaxation_is_explicit_and_truthfully_labelled():
    backend = OpenEvolveBackend.__new__(OpenEvolveBackend)
    backend.development_allow_verified_infeasible_parents = False
    outcome = CandidateOutcome(
        candidate_id="candidate",
        source_hash="a" * 64,
        status=CandidateStatus.VERIFIED,
        objective_value=0.49,
        constraint_compliant=False,
        verified=True,
        selection_outcome="scientifically_ineligible",
        rejection_reason="minimum_tissue_dice",
        replacement_outcome="archive_only",
    )
    assert backend.parent_eligible(outcome) is False

    backend.development_allow_verified_infeasible_parents = True
    assert backend.parent_eligible(outcome) is True
    assert backend.selection_disposition(
        verified=True,
        constraint_compliant=False,
        objective_value=0.49,
        reasons=("minimum_tissue_dice",),
    ) == (
        "development_ranked_scientifically_ineligible",
        "development_population_only",
        "minimum_tissue_dice",
    )
