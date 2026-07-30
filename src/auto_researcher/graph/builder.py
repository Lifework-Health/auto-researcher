"""Compile the PR 1 control plane with injected dependencies."""

from __future__ import annotations

from functools import partial
from typing import Literal

from langgraph.graph import END, START, StateGraph

from auto_researcher.graph.nodes.direct_search import direct_search
from auto_researcher.graph.nodes.evaluate import evaluate_experiment
from auto_researcher.graph.nodes.human_approval import approval_router, human_approval
from auto_researcher.graph.nodes.hypothesis import generate_hypothesis
from auto_researcher.graph.nodes.initialise import initialise_run
from auto_researcher.graph.nodes.planner import plan_search
from auto_researcher.graph.nodes.provenance import record_provenance
from auto_researcher.graph.nodes.search_router import search_router
from auto_researcher.graph.nodes.stop import supervisor_decide
from auto_researcher.graph.nodes.supervisor import supervisor_prepare
from auto_researcher.graph.nodes.unavailable_backend import unavailable_backend
from auto_researcher.graph.nodes.verify import verify_evidence
from auto_researcher.graph.routing import (
    route_after_decision,
    route_after_human,
    route_after_prepare,
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
        "generate_hypothesis",
        partial(generate_hypothesis, dependencies=dependencies),
    )
    graph.add_node("plan_search", partial(plan_search, dependencies=dependencies))
    graph.add_node("approval_router", partial(approval_router, dependencies=dependencies))
    graph.add_node("human_approval", human_approval)
    graph.add_node("search_router", search_router)
    graph.add_node("direct_search", partial(direct_search, dependencies=dependencies))
    graph.add_node("unavailable_backend", unavailable_backend)
    graph.add_node(
        "evaluate_experiment",
        partial(evaluate_experiment, dependencies=dependencies),
    )
    graph.add_node("verify_evidence", partial(verify_evidence, dependencies=dependencies))
    graph.add_node(
        "record_provenance",
        partial(record_provenance, dependencies=dependencies),
    )
    graph.add_node("supervisor_decide", supervisor_decide)

    graph.add_edge(START, "initialise_run")
    graph.add_edge("initialise_run", "supervisor_prepare")
    graph.add_conditional_edges("supervisor_prepare", route_after_prepare)
    graph.add_edge("generate_hypothesis", "plan_search")
    graph.add_edge("plan_search", "approval_router")
    graph.add_conditional_edges("approval_router", route_approval)
    graph.add_conditional_edges("human_approval", route_after_human)
    graph.add_conditional_edges("search_router", route_search_backend)
    graph.add_edge("direct_search", "evaluate_experiment")
    graph.add_edge("evaluate_experiment", "verify_evidence")
    graph.add_edge("verify_evidence", "record_provenance")
    graph.add_edge("unavailable_backend", "record_provenance")
    graph.add_edge("record_provenance", "supervisor_decide")
    graph.add_conditional_edges(
        "supervisor_decide",
        route_after_decision,
        {"generate_hypothesis": "generate_hypothesis", "__end__": END},
    )
    return graph.compile(
        checkpointer=dependencies.checkpointer,
        interrupt_after=interrupt_after,
        name="auto-researcher-v2.1",
    )
