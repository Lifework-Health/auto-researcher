"""Hypothesis agent invocation."""

import math

from auto_researcher.agents.context import AgentContextAssemblyError
from auto_researcher.agents.live.base import LiveAgentExecutionError
from auto_researcher.agents.telemetry import (
    apply_agent_telemetry,
    consume_agent_telemetry,
)
from auto_researcher.agents.models import HypothesisAgentContext
from auto_researcher.contracts.enums import (
    EvidenceStatus,
    GroundingStatus,
    HypothesisStatus,
    ProposalSource,
    ProvenanceKind,
    RunStatus,
)
from auto_researcher.contracts.models import Hypothesis
from auto_researcher.graph.state import ResearchState
from auto_researcher.runtime.dependencies import RuntimeDependencies
from auto_researcher.runtime.identity import payload_hash


def _deterministic_prior_hypothesis_fallback(
    state: ResearchState,
    context: HypothesisAgentContext | None,
    dependencies: RuntimeDependencies,
    *,
    failure_code: str,
) -> Hypothesis | None:
    """Keep a development campaign alive using its best verified prior result."""

    compatible: list[tuple[float, str, dict, tuple[str, ...]]] = []
    for finding in context.prior_verified_findings if context is not None else ():
        if (
            finding.primary_score is None
            or not math.isfinite(finding.primary_score)
            or not finding.constraint_compliant
            or finding.evidence_status != EvidenceStatus.SUPPORTED
        ):
            continue
        try:
            configuration = dependencies.task.normalise_configuration(
                dict(finding.safe_configuration)
            )
        except (TypeError, ValueError):
            continue
        compatible.append(
            (
                float(finding.primary_score),
                finding.experiment_reference,
                configuration,
                (
                    finding.hypothesis_reference,
                    finding.experiment_reference,
                ),
            )
        )
    if compatible:
        score, experiment_reference, configuration, references = max(
            compatible,
            key=lambda item: (item[0], item[1]),
        )
        grounding = GroundingStatus.PRIOR_RESULTS_GROUNDED
        fallback_basis = "the best verified prior result"
        expected_observation = (
            f"{state['contract'].primary_metric} is at least {score}."
        )
        falsification_condition = (
            f"{state['contract'].primary_metric} is lower than {score}."
        )
    else:
        raw_incumbent = dependencies.runtime_context.task_options.get(
            "initial_incumbent_configuration"
        )
        if not isinstance(raw_incumbent, dict):
            return None
        try:
            configuration = dependencies.task.normalise_configuration(raw_incumbent)
        except (TypeError, ValueError):
            return None
        experiment_reference = "configured-initial-incumbent"
        references = (state["contract"].contract_id,)
        grounding = GroundingStatus.CONTRACT_GROUNDED
        fallback_basis = "the configured bounded incumbent"
        expected_observation = (
            f"{state['contract'].primary_metric} remains measurable for the incumbent."
        )
        falsification_condition = (
            f"{state['contract'].primary_metric} cannot be measured for the incumbent."
        )
    identity = payload_hash(
        {
            "run_id": state["run_id"],
            "cycle": state["cycle"],
            "failure_code": failure_code,
            "experiment_reference": experiment_reference,
            "configuration": configuration,
        }
    )
    return Hypothesis(
        hypothesis_id=f"hyp-fallback-{identity[:20]}",
        statement=(
            "The bounded configuration remains a valid incumbent for the next "
            "planner decision."
        ),
        rationale=(
            f"Deterministic recovery using {fallback_basis} after the hypothesis "
            f"boundary returned {failure_code}."
        ),
        predicted_subspace=configuration,
        expected_observation=expected_observation,
        falsification_condition=falsification_condition,
        evidence_references=references,
        prior_weight=0.5,
        status=HypothesisStatus.OPEN,
        provenance=ProvenanceKind.MOCK,
        proposal_source=ProposalSource.DETERMINISTIC,
        grounding_status=grounding,
        prompt_version="deterministic-incumbent-fallback-v1",
    )


def generate_hypothesis(
    state: ResearchState,
    dependencies: RuntimeDependencies,
) -> dict:
    context: HypothesisAgentContext | None = None
    stage = "context_assembly"
    try:
        context = dependencies.agent_context_assembler.hypothesis_context(
            state,
            dependencies.task_agent_context,
        )
        stage = "model_call"
        hypothesis = dependencies.hypothesis_agent.generate(context)
    except Exception as exc:
        telemetry = consume_agent_telemetry(dependencies.hypothesis_agent)
        code = (
            exc.code
            if isinstance(exc, (LiveAgentExecutionError, AgentContextAssemblyError))
            else "hypothesis_agent_failed"
        )
        fallback = (
            _deterministic_prior_hypothesis_fallback(
                state,
                context,
                dependencies,
                failure_code=code,
            )
            if isinstance(exc, (LiveAgentExecutionError, AgentContextAssemblyError))
            else None
        )
        if fallback is not None:
            return {
                "budget": apply_agent_telemetry(state["budget"], telemetry),
                "active_hypothesis": fallback,
                "hypothesis_fallback_code": code,
                "hypothesis_failure_stage": stage,
                "executed_nodes": ["generate_hypothesis"],
            }
        return {
            "status": RunStatus.FAILED,
            "budget": apply_agent_telemetry(state["budget"], telemetry),
            "active_hypothesis": None,
            "errors": [code],
            "stop_reason": code,
            "hypothesis_failure_code": code,
            "hypothesis_failure_stage": stage,
            "executed_nodes": ["generate_hypothesis"],
        }
    telemetry = consume_agent_telemetry(dependencies.hypothesis_agent)
    update = {
        "active_hypothesis": hypothesis,
        "budget": apply_agent_telemetry(state["budget"], telemetry),
        "hypothesis_failure_code": None,
        "hypothesis_failure_stage": None,
        "hypothesis_fallback_code": None,
        "executed_nodes": ["generate_hypothesis"],
    }
    if telemetry is not None and telemetry.cost_limit_exceeded:
        update.update(
            status=RunStatus.STOPPED,
            stop_reason="maximum_agent_call_cost_exceeded",
        )
    return update
