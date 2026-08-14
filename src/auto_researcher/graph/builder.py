"""Compile the domain-neutral control plane with injected task dependencies."""

from __future__ import annotations

from functools import partial
from typing import Literal

from langgraph.graph import END, START, StateGraph

from auto_researcher.graph.nodes.direct_search import direct_search
from auto_researcher.graph.nodes.evaluate import evaluate_experiment
from auto_researcher.graph.nodes.human_approval import approval_router, human_approval
from auto_researcher.graph.nodes.hypothesis import generate_hypothesis
from auto_researcher.graph.nodes.knowledge import retrieve_knowledge
from auto_researcher.graph.nodes.initialise import initialise_run
from auto_researcher.graph.nodes.planner import plan_search
from auto_researcher.graph.nodes.optuna import (
    optuna_ask_trial,
    optuna_create_experiment,
    optuna_decide_study,
    optuna_finalise_study,
    optuna_prepare_study,
    optuna_record_trial,
    optuna_tell_trial,
)
from auto_researcher.graph.nodes.openevolve import (
    decide_openevolve_continue,
    finalise_openevolve,
    initialise_openevolve,
    prepare_openevolve_candidate,
    propose_openevolve_candidate,
    record_openevolve_candidate,
    select_openevolve_parent,
    validate_openevolve_candidate,
)
from auto_researcher.graph.nodes.native_openevolve import run_native_openevolve
from auto_researcher.graph.nodes.provenance import record_provenance
from auto_researcher.graph.nodes.search_router import search_router
from auto_researcher.graph.nodes.stop import supervisor_decide
from auto_researcher.graph.nodes.supervisor import supervisor_prepare
from auto_researcher.graph.nodes.unavailable_backend import unavailable_backend
from auto_researcher.graph.nodes.verify import verify_evidence
from auto_researcher.graph.routing import (
    route_after_decision,
    route_after_human,
    route_after_initialise,
    route_after_knowledge,
    route_after_optuna_decision,
    route_after_evaluation,
    route_after_optuna_prepare,
    route_after_openevolve_decision,
    route_after_openevolve_preparation,
    route_after_openevolve_validation,
    route_after_native_openevolve,
    route_after_prepare,
    route_after_verification,
    route_approval,
    route_search_backend,
)
from auto_researcher.graph.state import ResearchState
from auto_researcher.runtime.dependencies import RuntimeDependencies


def build_graph(
    dependencies: RuntimeDependencies,
    *,
    interrupt_after: list[str] | Literal["*"] | None = None,
):
    graph = StateGraph(ResearchState)
    graph.add_node("initialise_run", initialise_run)
    graph.add_node("supervisor_prepare", supervisor_prepare)
    graph.add_node(
        "retrieve_knowledge",
        partial(retrieve_knowledge, dependencies=dependencies),
    )
    graph.add_node(
        "generate_hypothesis",
        partial(generate_hypothesis, dependencies=dependencies),
    )
    graph.add_node("plan_search", partial(plan_search, dependencies=dependencies))
    graph.add_node(
        "approval_router", partial(approval_router, dependencies=dependencies)
    )
    graph.add_node("human_approval", human_approval)
    graph.add_node(
        "search_router",
        partial(search_router, dependencies=dependencies),
    )
    graph.add_node("direct_search", partial(direct_search, dependencies=dependencies))
    graph.add_node("unavailable_backend", unavailable_backend)
    graph.add_node(
        "evaluate_experiment",
        partial(evaluate_experiment, dependencies=dependencies),
    )
    graph.add_node(
        "verify_evidence", partial(verify_evidence, dependencies=dependencies)
    )
    graph.add_node(
        "record_provenance",
        partial(record_provenance, dependencies=dependencies),
    )
    graph.add_node("supervisor_decide", supervisor_decide)
    graph.add_node(
        "optuna_prepare_study",
        partial(optuna_prepare_study, dependencies=dependencies),
    )
    graph.add_node(
        "optuna_ask_trial",
        partial(optuna_ask_trial, dependencies=dependencies),
    )
    graph.add_node(
        "optuna_create_experiment",
        partial(optuna_create_experiment, dependencies=dependencies),
    )
    graph.add_node(
        "optuna_tell_trial",
        partial(optuna_tell_trial, dependencies=dependencies),
    )
    graph.add_node(
        "optuna_record_trial",
        partial(optuna_record_trial, dependencies=dependencies),
    )
    graph.add_node("optuna_decide_study", optuna_decide_study)
    graph.add_node(
        "optuna_finalise_study",
        partial(optuna_finalise_study, dependencies=dependencies),
    )
    graph.add_node(
        "initialise_openevolve",
        partial(initialise_openevolve, dependencies=dependencies),
    )
    graph.add_node(
        "select_openevolve_parent",
        partial(select_openevolve_parent, dependencies=dependencies),
    )
    graph.add_node(
        "propose_openevolve_candidate",
        partial(propose_openevolve_candidate, dependencies=dependencies),
    )
    graph.add_node(
        "validate_openevolve_candidate",
        partial(validate_openevolve_candidate, dependencies=dependencies),
    )
    graph.add_node(
        "prepare_openevolve_candidate",
        partial(prepare_openevolve_candidate, dependencies=dependencies),
    )
    graph.add_node(
        "record_openevolve_candidate",
        partial(record_openevolve_candidate, dependencies=dependencies),
    )
    graph.add_node(
        "decide_openevolve_continue",
        partial(decide_openevolve_continue, dependencies=dependencies),
    )
    graph.add_node(
        "finalise_openevolve",
        partial(finalise_openevolve, dependencies=dependencies),
    )
    graph.add_node(
        "run_native_openevolve",
        partial(run_native_openevolve, dependencies=dependencies),
    )

    graph.add_edge(START, "initialise_run")
    graph.add_conditional_edges(
        "initialise_run",
        route_after_initialise,
        {"supervisor_prepare": "supervisor_prepare", "__end__": END},
    )
    graph.add_conditional_edges("supervisor_prepare", route_after_prepare)
    graph.add_conditional_edges("retrieve_knowledge", route_after_knowledge)
    graph.add_edge("generate_hypothesis", "plan_search")
    graph.add_edge("plan_search", "approval_router")
    graph.add_conditional_edges("approval_router", route_approval)
    graph.add_conditional_edges("human_approval", route_after_human)
    graph.add_conditional_edges("search_router", route_search_backend)
    graph.add_edge("direct_search", "evaluate_experiment")
    graph.add_conditional_edges(
        "optuna_prepare_study",
        route_after_optuna_prepare,
    )
    graph.add_edge("optuna_ask_trial", "optuna_create_experiment")
    graph.add_edge("optuna_create_experiment", "evaluate_experiment")
    graph.add_conditional_edges("evaluate_experiment", route_after_evaluation)
    graph.add_conditional_edges("verify_evidence", route_after_verification)
    graph.add_edge("optuna_tell_trial", "optuna_record_trial")
    graph.add_edge("optuna_record_trial", "optuna_decide_study")
    graph.add_conditional_edges(
        "optuna_decide_study",
        route_after_optuna_decision,
    )
    graph.add_edge("optuna_finalise_study", "supervisor_decide")
    graph.add_edge("initialise_openevolve", "validate_openevolve_candidate")
    graph.add_conditional_edges(
        "validate_openevolve_candidate",
        route_after_openevolve_validation,
    )
    graph.add_conditional_edges(
        "prepare_openevolve_candidate",
        route_after_openevolve_preparation,
    )
    graph.add_edge("record_openevolve_candidate", "decide_openevolve_continue")
    graph.add_conditional_edges(
        "decide_openevolve_continue",
        route_after_openevolve_decision,
    )
    graph.add_edge("select_openevolve_parent", "propose_openevolve_candidate")
    graph.add_edge("propose_openevolve_candidate", "validate_openevolve_candidate")
    graph.add_edge("finalise_openevolve", "supervisor_decide")
    graph.add_conditional_edges(
        "run_native_openevolve",
        route_after_native_openevolve,
        {"run_native_openevolve": "run_native_openevolve", "__end__": END},
    )
    graph.add_edge("unavailable_backend", "record_provenance")
    graph.add_edge("record_provenance", "supervisor_decide")
    graph.add_conditional_edges(
        "supervisor_decide",
        route_after_decision,
        {"retrieve_knowledge": "retrieve_knowledge", "__end__": END},
    )
    return graph.compile(
        checkpointer=dependencies.checkpointer,
        interrupt_after=interrupt_after,
        name="auto-researcher-v2.1",
    )
