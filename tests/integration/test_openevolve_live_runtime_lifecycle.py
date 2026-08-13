from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from auto_researcher.agents.models import StructuredModelResponse
from auto_researcher.cli import app
from auto_researcher.contracts.enums import RunStatus, SearchType
from auto_researcher.graph.builder import build_graph
from auto_researcher.providers.fake_production import (
    FakeProductionStructuredModelClient,
)
from auto_researcher.runtime.dependencies import task_sqlite_dependencies
from auto_researcher.runtime.execution import resume_run, start_run
from auto_researcher.runtime.identity import payload_hash
from auto_researcher.search.openevolve.hardened_executor import (
    HardenedDockerExecutor,
    docker_policy,
)
from auto_researcher.search.openevolve.identity import component_interface_identity
from auto_researcher.search.openevolve.live_boundary import (
    MetadataOnlyMutationBoundary,
    metadata_only_model_exposure_identity,
)
from auto_researcher.search.openevolve.live_models import (
    MetadataOnlyLiveMutationApproval,
    MetadataOnlyOpenEvolveModelCallContext,
    metadata_only_approval_content_hash,
)
from auto_researcher.search.openevolve.live_runtime import (
    MetadataOnlyLiveOpenEvolveConfiguration,
    MetadataOnlyLiveOpenEvolveRuntime,
)
from auto_researcher.search.openevolve.models import EvolvableComponentSpec
from auto_researcher.search.openevolve.production_bridge import load_mutation_prompt
from auto_researcher.search.openevolve.sandbox import LocalSandboxRunner
from auto_researcher.search.openevolve.upstream import default_adapter_contract
from auto_researcher.search.openevolve.upstream_models import ExecutorIsolationResult
from auto_researcher.tasks.models import TaskRuntimeContext
from auto_researcher.tasks.synthetic import (
    SyntheticTask,
    default_synthetic_contract,
    default_synthetic_openevolve_configuration,
)
from auto_researcher.tasks.synthetic.openevolve import (
    NEURAL_SOURCE,
    SyntheticEvolvableComponent,
)
from tests.unit.test_openevolve_metadata_only_boundary import (
    PROMPT,
    metadata_only_contract,
)


ROOT = Path(__file__).parents[2]
NOW = datetime(2030, 2, 1, tzinfo=UTC)
DIGEST = "sha256:" + "7" * 64
POOR_SOURCE = """def evolve(configuration):
    return {"model_family": "linear", "complexity": 10, "learning_rate": 0.05}
"""


class MetadataOnlySyntheticComponent(SyntheticEvolvableComponent):
    def component_spec(self) -> EvolvableComponentSpec:
        spec = super().component_spec()
        return spec.model_copy(
            update={
                "parameter_schema": {
                    **spec.parameter_schema,
                    "mutation_context": spec.task_mutation_context,
                }
            }
        )


class MetadataOnlySyntheticTask(SyntheticTask):
    def live_mutation_boundary(self) -> MetadataOnlyMutationBoundary:
        return MetadataOnlyMutationBoundary(underlying_dataset_class="synthetic")

    def create_evolvable_component(self, contract, runtime_context):
        self.validate_contract(contract)
        return MetadataOnlySyntheticComponent()


class SequencedProvider(FakeProductionStructuredModelClient):
    def __init__(self, responses: list[dict]) -> None:
        super().__init__(
            provider="fake-production",
            model_id="fake-model-20260101",
            response=responses[0],
        )
        self.responses = responses

    def generate_structured(self, **kwargs) -> StructuredModelResponse:
        self.response = self.responses[self.invocation_count]
        return super().generate_structured(**kwargs)


def _write_json(path: Path, value) -> Path:
    payload = (
        value.model_dump(mode="json", by_alias=True)
        if hasattr(value, "model_dump")
        else value
    )
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _configuration():
    configuration = default_synthetic_openevolve_configuration()
    configuration["openevolve"].update(
        {
            "maximum_generations": 2,
            "maximum_model_calls": 2,
            "objective_threshold": None,
            "sandbox_policy_id": "openevolve-hardened-executor-v2",
        }
    )
    return configuration


def _runtime_files(tmp_path: Path, run_id: str, thread_id: str, contract):
    task = MetadataOnlySyntheticTask()
    component = task.create_evolvable_component(contract, TaskRuntimeContext())
    spec = component.component_spec()
    adapter = default_adapter_contract(ROOT / "constraints/openevolve-0.3.2.lock")
    policy = docker_policy(
        "auto-researcher/openevolve-executor:test",
        DIGEST,
        ROOT / "docker/openevolve-executor/Dockerfile",
        ROOT / "docker/openevolve-executor/worker.py",
        "fixture-runtime",
    )
    policy_hash = payload_hash(policy)
    bridge = metadata_only_contract()
    context = MetadataOnlyOpenEvolveModelCallContext(
        run_id=run_id,
        thread_id=thread_id,
        contract_id=contract.contract_id,
        contract_hash=payload_hash(contract),
        task_id=task.task_id,
        task_version=task.task_version,
        search_request_id="runtime-bound-by-authoritative-reservation",
        generation=1,
        parent_candidate_id="seed-placeholder",
        component_id=spec.component_id,
        component_version=spec.component_version,
        component_interface_hash=component_interface_identity(spec),
        model_exposure_identity=metadata_only_model_exposure_identity(spec),
        underlying_dataset_class="synthetic",
        adapter_id=adapter.adapter_id,
        adapter_version=adapter.adapter_version,
        adapter_identity_hash=payload_hash(adapter),
        executor_policy_hash=policy_hash,
        image_digest=policy.image_digest,
        mutable_file=spec.mutable_file,
        model_budget_identity="fixture-budget",
        maximum_model_calls=2,
        maximum_model_cost=0.04,
    )
    approval_payload = {
        "approval_id": "metadata-only-standard-runtime",
        "run_id": run_id,
        "contract_id": contract.contract_id,
        "contract_hash": payload_hash(contract),
        "task_id": task.task_id,
        "task_version": task.task_version,
        "component_id": spec.component_id,
        "component_version": spec.component_version,
        "component_interface_hash": context.component_interface_hash,
        "model_exposure_identity": context.model_exposure_identity,
        "underlying_dataset_class": "synthetic",
        "exposure_class": "metadata_only",
        "adapter_id": adapter.adapter_id,
        "adapter_version": adapter.adapter_version,
        "adapter_identity_hash": payload_hash(adapter),
        "provider": "fake-production",
        "model_id": "fake-model-20260101",
        "prompt_id": bridge.prompt_id,
        "prompt_version": bridge.prompt_version,
        "prompt_hash": payload_hash(PROMPT),
        "mutation_operator_version": bridge.mutation_operator_version,
        "maximum_model_calls": 2,
        "maximum_input_tokens": 20_000,
        "maximum_output_tokens": 1_000,
        "maximum_total_cost": 0.04,
        "currency": "USD",
        "pricing_version": "fake-pricing-v1",
        "executor_policy_hash": policy_hash,
        "image_digest": policy.image_digest,
        "mutable_file": spec.mutable_file,
        "created_at": NOW - timedelta(minutes=1),
        "expires_at": NOW + timedelta(minutes=10),
        "reviewer_identity": "offline-test-operator",
        "residual_risk_acknowledged": True,
    }
    approval_payload["approval_hash"] = metadata_only_approval_content_hash(
        approval_payload
    )
    approval = MetadataOnlyLiveMutationApproval.model_validate(approval_payload)
    isolation = ExecutorIsolationResult(
        executor_policy_hash=policy_hash,
        network_isolation_verified=True,
        mount_isolation_verified=True,
        environment_sanitisation_verified=True,
        safe_checks={"offline_fixture": True},
    )
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text(PROMPT, encoding="utf-8")
    configuration = MetadataOnlyLiveOpenEvolveConfiguration(
        approval_file=_write_json(tmp_path / "approval.json", approval),
        bridge_contract_file=_write_json(tmp_path / "bridge.json", bridge),
        adapter_lock_file=ROOT / "constraints/openevolve-0.3.2.lock",
        prompt_file=prompt_file,
        executor_policy_file=_write_json(tmp_path / "executor.json", policy),
        isolation_evidence_file=_write_json(tmp_path / "isolation.json", isolation),
    )
    assert load_mutation_prompt(prompt_file) == PROMPT
    return task, configuration


def _responses():
    return [
        {
            "protocol_version": "upstream-mutation-envelope-v1",
            "mutable_file": "candidate.py",
            "source": POOR_SOURCE,
            "description": "A valid but deliberately weaker tree child.",
        },
        {
            "protocol_version": "upstream-mutation-envelope-v1",
            "mutable_file": "candidate.py",
            "source": NEURAL_SOURCE,
            "description": "A stronger valid neural child.",
        },
    ]


def _manager(tmp_path, runtime, task, contract):
    return task_sqlite_dependencies(
        task,
        TaskRuntimeContext(
            run_id=runtime.thread_id.removesuffix("-thread"),
            output_dir=tmp_path / "artefacts",
            workspace_dir=tmp_path / "workspace",
            manifest_created_at=NOW,
        ),
        contract,
        _configuration(),
        tmp_path / "checkpoints.sqlite",
        tmp_path / "provenance.sqlite",
        agent_calls_path=tmp_path / "agent-calls.sqlite",
        knowledge_retrievals_path=tmp_path / "knowledge.sqlite",
        search_type=SearchType.OPENEVOLVE,
        openevolve_live_runtime=runtime,
        clock=lambda: NOW,
    )


def test_standard_live_runtime_runs_multiple_generations_and_restart_without_redispatch(
    tmp_path, monkeypatch
):
    run_id = "standard-live-run"
    thread_id = f"{run_id}-thread"
    contract = default_synthetic_contract(
        search_types=frozenset({SearchType.OPENEVOLVE}),
        maximum_experiments=3,
    )
    task, configuration = _runtime_files(tmp_path, run_id, thread_id, contract)
    provider = SequencedProvider(_responses())
    runtime = MetadataOnlyLiveOpenEvolveRuntime(
        configuration=configuration,
        thread_id=thread_id,
        provider_factory=lambda: provider,
        executor_validator=lambda executor: None,
    )

    def cheap_hardened_prepare(self, candidate, component, policy, configuration):
        return LocalSandboxRunner(tmp_path / "cpu-preparation").prepare(
            candidate, component, policy, configuration
        )

    monkeypatch.setattr(HardenedDockerExecutor, "prepare", cheap_hardened_prepare)
    monkeypatch.setattr(
        HardenedDockerExecutor, "validate_environment", lambda self: None
    )
    graph_config = {"configurable": {"thread_id": thread_id}}
    initial = {"run_id": run_id, "thread_id": thread_id, "contract": contract}

    with _manager(tmp_path, runtime, task, contract) as dependencies:
        paused = start_run(
            build_graph(
                dependencies,
                interrupt_after=["propose_openevolve_candidate"],
            ),
            initial,
            graph_config,
        )
        assert paused["status"] == RunStatus.RUNNING
        assert provider.invocation_count == 1

    with _manager(tmp_path, runtime, task, contract) as dependencies:
        final = resume_run(build_graph(dependencies), graph_config)
        records = dependencies.agent_call_store.list_records()

    assert final["status"] == RunStatus.COMPLETED
    assert provider.invocation_count == 2
    assert len({record.call_id for record in records}) == 2
    assert final["openevolve_population_state"].budget.model_calls == 2
    candidates = sorted(
        final["openevolve_candidates"].candidates,
        key=lambda candidate: candidate.generation,
    )
    assert [candidate.creation_provenance for candidate in candidates] == [
        "SEED",
        "FAKE_MODEL",
        "FAKE_MODEL",
    ]
    assert [
        outcome.objective_value
        for outcome in final["openevolve_population_state"].outcomes
    ] == [
        0.78,
        0.57,
        0.88,
    ]
    assert candidates[-1].parent_candidate_ids == (candidates[0].candidate_id,)
    assert final["openevolve_search_result"].best_candidate_ids == (
        candidates[-1].candidate_id,
    )


def test_standard_cli_launches_metadata_only_live_lifecycle(tmp_path, monkeypatch):
    run_id = "standard-cli-live-run"
    thread_id = f"{run_id}-thread"
    monkeypatch.setattr("auto_researcher.cli.utc_now", lambda: NOW)
    contract = default_synthetic_contract(
        search_types=frozenset({SearchType.OPENEVOLVE}),
        maximum_experiments=3,
    )
    task, configuration = _runtime_files(tmp_path, run_id, thread_id, contract)
    provider = SequencedProvider(_responses())

    class Registry:
        @staticmethod
        def get(task_id, task_version):
            assert (task_id, task_version) == ("synthetic", "1.0")
            return task

    monkeypatch.setattr("auto_researcher.cli.default_task_registry", lambda: Registry())
    monkeypatch.setattr(
        "auto_researcher.search.openevolve.live_runtime.default_live_mutation_provider_factory",
        lambda bridge: lambda: provider,
    )

    def cheap_hardened_prepare(self, candidate, component, policy, base):
        return LocalSandboxRunner(tmp_path / "cli-cpu-preparation").prepare(
            candidate, component, policy, base
        )

    monkeypatch.setattr(HardenedDockerExecutor, "prepare", cheap_hardened_prepare)
    monkeypatch.setattr(
        HardenedDockerExecutor, "validate_environment", lambda self: None
    )
    contract_file = _write_json(tmp_path / "contract.json", contract)
    config_file = tmp_path / "task.yaml"
    config_file.write_text(
        yaml.safe_dump(
            {
                "task": {"id": "synthetic", "version": "1.0"},
                "agents": {"mode": "mock"},
                "openevolve_live_mutation": configuration.model_dump(mode="json"),
                "search": {"type": "OPENEVOLVE", **_configuration()},
                "runtime": {
                    "output_dir": str((tmp_path / "cli-artefacts").resolve()),
                    "workspace_dir": str((tmp_path / "cli-workspace").resolve()),
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "start",
            "--task",
            "synthetic",
            "--contract",
            str(contract_file),
            "--task-config",
            str(config_file),
            "--run-id",
            run_id,
            "--thread-id",
            thread_id,
            "--checkpoint-db",
            str(tmp_path / "cli-checkpoints.sqlite"),
            "--provenance-db",
            str(tmp_path / "cli-provenance.sqlite"),
            "--agent-calls-db",
            str(tmp_path / "cli-agent-calls.sqlite"),
            "--knowledge-retrievals-db",
            str(tmp_path / "cli-knowledge.sqlite"),
        ],
    )
    assert result.exit_code == 0, (result.stdout, result.stderr, result.exception)
    assert "Status: COMPLETED" in result.stdout
    assert "OpenEvolve mutation mode: metadata_only_live" in result.stdout
    assert provider.invocation_count == 2
