"""Explicitly non-production live mutation for public-data development runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from auto_researcher.agents.models import ModelCallConfig
from auto_researcher.providers.protocols import ProviderCallError, StructuredModelClient
from auto_researcher.runtime.identity import payload_hash
from auto_researcher.search.openevolve.live_models import (
    OPENEVOLVE_MUTATION_PROMPT_V2,
)
from auto_researcher.search.openevolve.production_bridge import load_mutation_prompt
from auto_researcher.search.openevolve.upstream import (
    AutoResearcherOpenEvolveModelBridge,
    UpstreamOpenEvolveAdapter,
    default_adapter_contract,
)
from auto_researcher.search.openevolve.upstream_models import (
    UpstreamMutationEnvelope,
)
from auto_researcher.secrets import (
    ResolvedSecret,
    SecretProvider,
    SecretReference,
    SecretResolutionError,
    SecretResolutionErrorCode,
    provider_for_reference,
)


class DevelopmentRuntimeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DevelopmentLiveOpenEvolveConfiguration(DevelopmentRuntimeModel):
    """Small, explicit live-model boundary that is never a production default."""

    protocol_version: Literal["development-live-openevolve-runtime-v1"] = (
        "development-live-openevolve-runtime-v1"
    )
    mode: Literal["local_development"] = "local_development"
    acknowledgement: Literal["public-data-non-production-development"]
    model: ModelCallConfig
    credential: SecretReference
    maximum_model_calls: Literal[2] = 2
    maximum_total_cost_usd: float = Field(gt=0, le=1)
    usage_log_file: Path

    @model_validator(mode="after")
    def bounded_anthropic_canary(self) -> "DevelopmentLiveOpenEvolveConfiguration":
        if (
            self.model.provider != "anthropic"
            or self.model.maximum_attempts != 1
            or self.model.prompt_version != OPENEVOLVE_MUTATION_PROMPT_V2
            or self.model.maximum_cost_per_call * self.maximum_model_calls
            > self.maximum_total_cost_usd
            or not self.credential.required
        ):
            raise ValueError("development_live_mutation_configuration_invalid")
        return self

    @field_validator("usage_log_file")
    @classmethod
    def usage_log_is_absolute(cls, value: Path) -> Path:
        path = value.expanduser()
        if not path.is_absolute():
            raise ValueError("development_usage_log_must_be_absolute")
        return path.resolve()


class DevelopmentStructuredMutationClient:
    """Two-call provider adapter with a small, source-free usage ledger."""

    def __init__(
        self,
        configuration: DevelopmentLiveOpenEvolveConfiguration,
        *,
        secret_provider: SecretProvider | None = None,
        model_client: StructuredModelClient | None = None,
    ) -> None:
        self.configuration = configuration
        self._secret_provider = secret_provider
        self._model_client = model_client
        self._resolved_credential: ResolvedSecret | None = None
        self._calls_used = 0
        self._total_cost = 0.0
        self._load_existing_usage()

    @property
    def calls_used(self) -> int:
        return self._calls_used

    @property
    def total_cost(self) -> float:
        return self._total_cost

    def _load_existing_usage(self) -> None:
        path = self.configuration.usage_log_file
        if not path.exists():
            return
        if not path.is_file():
            raise ValueError("development_usage_log_invalid")
        try:
            rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("development_usage_log_invalid") from exc
        for row in rows:
            if (
                not isinstance(row, dict)
                or row.get("protocol_version")
                != "development-openevolve-model-usage-v1"
                or not isinstance(row.get("estimated_cost_usd"), (int, float))
            ):
                raise ValueError("development_usage_log_invalid")
            self._calls_used += 1
            self._total_cost += float(row["estimated_cost_usd"])
        if self._calls_used > self.configuration.maximum_model_calls:
            raise ValueError("development_model_call_budget_exhausted")

    def _client(self) -> StructuredModelClient:
        if self._model_client is not None:
            return self._model_client
        reference = self.configuration.credential
        provider = self._secret_provider or provider_for_reference(reference)
        if self._resolved_credential is None:
            self._resolved_credential = provider.resolve(reference)
            if self._resolved_credential is None:
                raise SecretResolutionError(
                    SecretResolutionErrorCode.MISSING,
                    reference,
                ) from None
        from auto_researcher.providers.anthropic import create_anthropic_client

        self._model_client = create_anthropic_client(
            self.configuration.model,
            credential=self._resolved_credential,
        )
        return self._model_client

    def _append_usage(
        self,
        *,
        call_id: str,
        status: str,
        input_tokens: int,
        output_tokens: int,
        estimated_cost: float,
        response_hash: str | None,
    ) -> None:
        path = self.configuration.usage_log_file
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "protocol_version": "development-openevolve-model-usage-v1",
            "call_id": call_id,
            "status": status,
            "provider": self.configuration.model.provider,
            "model_id": self.configuration.model.model_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": estimated_cost,
            "response_hash": response_hash,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            handle.write("\n")

    def propose_mutation(self, request: dict) -> dict:
        if self._calls_used >= self.configuration.maximum_model_calls:
            raise ValueError("development_model_call_budget_exhausted")
        ordinal = self._calls_used + 1
        context_hash = payload_hash(
            {
                "domain": "development-openevolve-mutation-input-v1",
                "request": request,
            }
        )
        call_id = f"openevolve-development-call-{ordinal}-{context_hash[:16]}"
        self._calls_used += 1
        try:
            response = self._client().generate_structured(
                call_id=call_id,
                system_prompt=load_mutation_prompt(_default_prompt_path()),
                user_prompt=json.dumps(request, sort_keys=True, separators=(",", ":")),
                response_model=UpstreamMutationEnvelope,
                call_config=self.configuration.model,
                context_hash=context_hash,
            )
        except ProviderCallError as exc:
            self._total_cost += exc.estimated_cost
            self._append_usage(
                call_id=call_id,
                status="FAILED",
                input_tokens=exc.input_tokens,
                output_tokens=exc.output_tokens,
                estimated_cost=exc.estimated_cost,
                response_hash=None,
            )
            raise RuntimeError("development_model_call_failed") from None
        envelope = UpstreamMutationEnvelope.model_validate(response.structured_output)
        self._total_cost += response.estimated_cost
        self._append_usage(
            call_id=call_id,
            status="COMPLETED",
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            estimated_cost=response.estimated_cost,
            response_hash=response.response_hash,
        )
        if self._total_cost > self.configuration.maximum_total_cost_usd:
            raise ValueError("development_model_cost_budget_exhausted")
        return envelope.model_dump(mode="json")


class DevelopmentOpenEvolveModelBridge(AutoResearcherOpenEvolveModelBridge):
    @property
    def creation_provenance(self) -> str:
        return "LIVE_MODEL"


def _default_prompt_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "prompts"
        / "openevolve"
        / "openevolve-mutation-prompt-v2.md"
    )


def _default_adapter_lock_path() -> Path:
    return Path(__file__).resolve().parents[4] / "constraints" / "openevolve-0.3.2.lock"


def assemble_development_live_openevolve(
    configuration: DevelopmentLiveOpenEvolveConfiguration,
    *,
    secret_provider: SecretProvider | None = None,
    model_client: StructuredModelClient | None = None,
) -> tuple[UpstreamOpenEvolveAdapter, DevelopmentStructuredMutationClient]:
    """Return a local-sandbox mutation operator for an explicit dev profile."""

    client = DevelopmentStructuredMutationClient(
        configuration,
        secret_provider=secret_provider,
        model_client=model_client,
    )
    bridge = DevelopmentOpenEvolveModelBridge(
        client,
        provider=configuration.model.provider,
        model_id=configuration.model.model_id,
        prompt_version=configuration.model.prompt_version,
    )
    operator = UpstreamOpenEvolveAdapter(
        default_adapter_contract(_default_adapter_lock_path()),
        bridge,
    )
    return operator, client


__all__ = [
    "DevelopmentLiveOpenEvolveConfiguration",
    "DevelopmentStructuredMutationClient",
    "assemble_development_live_openevolve",
]
