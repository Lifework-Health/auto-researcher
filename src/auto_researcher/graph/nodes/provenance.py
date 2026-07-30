"""Single writer for append-only scientific decision events."""

from __future__ import annotations

from auto_researcher.contracts.enums import EventType, ProvenanceKind
from auto_researcher.contracts.models import DecisionEvent
from auto_researcher.agents.provenance import append_model_call_events
from auto_researcher.graph.state import ResearchState
from auto_researcher.runtime.dependencies import RuntimeDependencies

CODE_VERSION = "auto-researcher-v2.1-pr4"


def record_provenance(
    state: ResearchState,
    dependencies: RuntimeDependencies,
) -> dict:
    run_id = state["run_id"]
    cycle = state["cycle"]
    rows: list[tuple[EventType, str, tuple[str, ...], tuple[str, ...], str, ProvenanceKind]] = []
    hypothesis = state.get("active_hypothesis")
    request = state.get("search_request")
    backend = state.get("search_backend_result")
    experiment = state.get("experiment_spec")
    evaluation = state.get("evaluation_result")
    verification = state.get("verification_result")
    model_event_ids = append_model_call_events(
        dependencies.provenance_store,
        dependencies.agent_call_store,
        run_id=run_id,
        cycle=cycle,
    )

    if hypothesis:
        rows.append(
            (
                EventType.HYPOTHESIS_PROPOSED,
                "hypothesis_agent",
                (
                    state["contract"].contract_id,
                    *((hypothesis.agent_call_id,) if hypothesis.agent_call_id else ()),
                ),
                (
                    hypothesis.hypothesis_id,
                    f"source:{hypothesis.proposal_source.value}",
                    f"grounding:{hypothesis.grounding_status.value}",
                    f"prompt:{hypothesis.prompt_version or 'none'}",
                ),
                hypothesis.rationale,
                hypothesis.provenance,
            )
        )
    if request:
        rows.append(
            (
                EventType.SEARCH_PLANNED,
                "planner_agent",
                (
                    request.hypothesis_id,
                    *((request.agent_call_id,) if request.agent_call_id else ()),
                ),
                (
                    request.request_id,
                    f"search_type:{request.search_type.value}",
                    f"source:{request.proposal_source.value}",
                    f"grounding:{request.grounding_status.value}",
                    f"prompt:{request.prompt_version or 'none'}",
                ),
                request.rationale,
                ProvenanceKind.MOCK,
            )
        )
    if state.get("human_approval_granted") is not None and request:
        approved = bool(state["human_approval_granted"])
        rows.append(
            (
                EventType.HUMAN_DECISION,
                "human",
                (request.request_id,),
                (),
                "approved" if approved else "rejected",
                ProvenanceKind.REAL,
            )
        )
    if backend and not backend.available:
        rows.append(
            (
                EventType.BACKEND_UNAVAILABLE,
                "search_router",
                (request.request_id,) if request else (),
                (),
                backend.message,
                ProvenanceKind.REAL,
            )
        )
    if experiment:
        rows.append(
            (
                EventType.EXPERIMENT_PREPARED,
                "direct_search",
                (experiment.search_request_id,),
                (experiment.experiment_id,),
                "Prepared one deterministic DIRECT experiment without evaluating it.",
                experiment.provenance,
            )
        )
    if evaluation:
        evaluation_outputs = (
            f"score:{evaluation.primary_score}",
            *evaluation.artefact_references,
        )
        rows.append(
            (
                EventType.EVALUATION_OBSERVED,
                "evaluator",
                (evaluation.experiment_id,),
                evaluation_outputs,
                "Recorded evaluator measurements and explicit constraint results.",
                evaluation.provenance,
            )
        )
    if verification:
        rows.append(
            (
                EventType.EVIDENCE_VERIFIED,
                "verifier",
                (verification.experiment_id,),
                (
                    f"evidence:{verification.evidence_status.value}",
                    f"verified:{str(verification.verified).lower()}",
                    f"constraints:{str(verification.constraint_compliant).lower()}",
                    f"score:{verification.measured_score}",
                    f"search_type:{request.search_type.value if request else 'DIRECT'}",
                    f"hypothesis:{hypothesis.hypothesis_id if hypothesis else 'unknown'}",
                    *(
                        f"artefact:{reference}"
                        for reference in (
                            evaluation.artefact_references if evaluation else ()
                        )
                    ),
                ),
                "; ".join(verification.reasons) or "Evidence reconciled without issues.",
                verification.provenance,
            )
        )
    if not rows:
        rows.append(
            (
                EventType.RUN_STOPPED,
                "supervisor",
                (state["contract"].contract_id,),
                (),
                state.get("stop_reason") or "run stopped before a research proposal",
                ProvenanceKind.REAL,
            )
        )

    event_ids: list[str] = list(model_event_ids)
    for event_type, actor, inputs, outputs, rationale, provenance in rows:
        event = DecisionEvent(
            event_id=dependencies.id_generator("event"),
            run_id=run_id,
            cycle=cycle,
            event_type=event_type,
            actor=actor,
            input_references=inputs,
            output_references=outputs,
            rationale=rationale,
            timestamp=dependencies.clock(),
            code_version=CODE_VERSION,
            provenance=provenance,
        )
        dependencies.provenance_store.append_event(event)
        event_ids.append(event.event_id)
    return {
        "decision_event_ids": event_ids,
        "executed_nodes": ["record_provenance"],
    }
