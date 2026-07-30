from __future__ import annotations

import hashlib
import json

from auto_researcher.agents.models import (
    ModelCallConfig,
    StructuredModelResponse,
)


class FakeStructuredModelClient:
    provider = "fake"
    model_id = "fake-model-2026-07-30"

    def __init__(self, hypothesis: dict, planner: dict) -> None:
        self.hypothesis = hypothesis
        self.planner = planner
        self.calls: list[dict] = []

    def generate_structured(
        self,
        *,
        call_id,
        system_prompt,
        user_prompt,
        response_model,
        call_config: ModelCallConfig,
        context_hash,
    ) -> StructuredModelResponse:
        output = (
            self.hypothesis
            if response_model.__name__ == "HypothesisProposal"
            else self.planner
        )
        self.calls.append(
            {
                "call_id": call_id,
                "response_model": response_model.__name__,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        encoded = json.dumps(output, sort_keys=True, separators=(",", ":")).encode()
        return StructuredModelResponse(
            call_id=call_id,
            provider=self.provider,
            model_id=self.model_id,
            structured_output=output,
            input_tokens=100,
            output_tokens=50,
            estimated_cost=call_config.pricing.estimate(
                input_tokens=100,
                output_tokens=50,
            ),
            latency_ms=10,
            attempts=1,
            finish_reason="end_turn",
            provider_request_id=f"request-{len(self.calls)}",
            prompt_version=call_config.prompt_version,
            context_hash=context_hash,
            response_hash=hashlib.sha256(encoded).hexdigest(),
        )
