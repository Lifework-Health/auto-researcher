from auto_researcher.agents.mock import MockHypothesisAgent
from auto_researcher.contracts.enums import SearchType
from auto_researcher.graph.builder import build_graph
from auto_researcher.runtime.dependencies import task_memory_dependencies
from auto_researcher.tasks.models import TaskRuntimeContext
from auto_researcher.tasks.synthetic import (
    SyntheticTask,
    default_synthetic_configuration,
    default_synthetic_contract,
)


class CapturingHypothesisAgent:
    def __init__(self):
        self.contexts = []
        self.delegate = MockHypothesisAgent()

    def generate(self, context):
        self.contexts.append(context)
        return self.delegate.generate(context)


def test_only_compact_verified_prior_results_enter_later_agent_context(tmp_path):
    contract = default_synthetic_contract(
        maximum_cycles=2,
        maximum_experiments=2,
        search_types=frozenset({SearchType.DIRECT}),
    )
    agent = CapturingHypothesisAgent()
    dependencies = task_memory_dependencies(
        SyntheticTask(),
        TaskRuntimeContext(run_id="prior-run", output_dir=tmp_path),
        contract,
        default_synthetic_configuration(),
        hypothesis_agent=agent,
    )
    final = build_graph(dependencies).invoke(
        {
            "run_id": "prior-run",
            "thread_id": "prior-thread",
            "contract": contract,
        },
        {"configurable": {"thread_id": "prior-thread"}},
    )
    assert final["budget"].experiments_used == 2
    assert len(agent.contexts) == 2
    assert agent.contexts[0].prior_verified_findings == ()
    assert len(agent.contexts[1].prior_verified_findings) == 1
    prior = agent.contexts[1].prior_verified_findings[0]
    assert prior.constraint_compliant is True
    assert prior.experiment_reference
