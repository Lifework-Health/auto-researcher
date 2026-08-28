from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from enum import StrEnum
from pathlib import Path

import ormsgpack
import pytest
from langgraph.checkpoint.base import empty_checkpoint

from auto_researcher.agents.models import ResearchDirective
from auto_researcher.contracts.enums import (
    EvidenceStatus,
    KnowledgeGroundingMode,
    ProvenanceKind,
    ReadSafetyMode,
    RunStatus,
)
from auto_researcher.contracts.models import (
    BudgetState,
    EvaluationResult,
    KnowledgeGroundingRequirement,
    ResearchContract,
    VerificationResult,
)
from auto_researcher.runtime.checkpoints import (
    ALLOWED_CHECKPOINT_TYPES,
    checkpoint_serializer,
    sqlite_checkpointer,
)
from auto_researcher.runtime.execution import execution_identity
from auto_researcher.runtime.identity import payload_hash
from auto_researcher.tasks.models import TaskRuntimeContext
from auto_researcher.tasks.synthetic import (
    SyntheticTask,
    default_synthetic_configuration,
    default_synthetic_contract,
)
from tests.helpers_read_safety import operator_configuration


RUN_ID = "checkpoint-04c-fixture"
THREAD_ID = "checkpoint-04c-fixture-thread"
ATTESTATION_HASH = "1" * 64
CONFIGURATION_HASH = "2" * 64
EVALUATION_REUSE_IDENTITY = "3" * 64
VERIFICATION_REUSE_IDENTITY = "4" * 64


def _operator_contract() -> ResearchContract:
    return ResearchContract(
        contract_id="checkpoint-04c-contract",
        schema_version="1.0",
        task_id="icca_nbs",
        task_version="1.0",
        objective_version="0.9",
        primary_metric="stability_objective",
        task_constraints_version="1.0",
        question="Does the reviewed DIRECT configuration satisfy eligibility?",
        objective="maximise the imported v2 stability objective",
        constraints={
            "read_safety_fixture": {
                "credential_class": "MANAGED_INSTANCE_PRIMARY",
                "privilege_verified": False,
                "attestation_id": "aura-professional-checkpoint-04",
                "attestation_version": "1.0.0",
                "attestation_hash": ATTESTATION_HASH,
                "configuration_hash": CONFIGURATION_HASH,
                "attestation_hash_algorithm": "canonical-json-sha256-v1",
                "configuration_hash_algorithm": "canonical-json-sha256-v1",
                "service_tier": "PROFESSIONAL",
                "residual_risk": "DATABASE_CREDENTIAL_NOT_ENFORCED_READ_ONLY",
                "template_ids": [
                    "generic.schema_preflight@1.0.0",
                    "icca_nbs.network_catalog@1.0.0",
                    "icca_nbs.gene_signature_pathway@1.0.0",
                ],
            }
        },
        allowed_search_types=frozenset({"DIRECT"}),
        evaluator_id="icca-nbs-v2-evaluator",
        verifier_id="deterministic-verifier",
        maximum_cycles=1,
        maximum_experiments=1,
        maximum_cost=2.0,
        grounding=KnowledgeGroundingRequirement(
            mode=KnowledgeGroundingMode.REQUIRED,
            permitted_providers=frozenset({"neo4j"}),
            permitted_trust_tiers=frozenset({"CURATED", "CORPUS"}),
            minimum_assertion_confidence=0.6,
            maximum_knowledge_references=20,
            maximum_query_records=100,
            maximum_graph_hops=3,
            maximum_retrieval_duration=20,
            knowledge_schema_version="knowledge-graph-auto-v0.1",
            knowledge_content_version="backbone-2026-06",
            permitted_read_safety_modes=frozenset(
                {
                    ReadSafetyMode.PRIVILEGE_VERIFIED,
                    ReadSafetyMode.OPERATOR_ATTESTED,
                }
            ),
        ),
        provenance=ProvenanceKind.REAL,
    )


def test_research_directive_round_trips_through_checkpoint_serializer():
    directive = ResearchDirective(
        directive_id="directive-checkpoint-roundtrip",
        trigger="campaign_start",
        mechanism_hypothesis="A bounded mechanism may improve Dice.",
        rationale="Use the registered search envelope.",
        selected_operators=("OPENEVOLVE",),
        experiment_allocation={"OPENEVOLVE": 1},
        targeted_dimensions=("feature_width",),
        expected_observation="objective score improves",
        falsification_condition="objective score does not improve",
        confidence=0.6,
        agent_call_id="model-call-checkpoint-roundtrip",
        prompt_version="2.0.0",
        context_hash="context-checkpoint-roundtrip",
    )
    serializer = checkpoint_serializer()

    restored = serializer.loads_typed(serializer.dumps_typed(directive))

    assert restored == directive
    assert isinstance(restored, ResearchDirective)


def _terminal_fixture() -> tuple[dict, dict, dict]:
    contract = _operator_contract()
    config = {"configurable": {"thread_id": THREAD_ID}}
    initial = {"run_id": RUN_ID, "thread_id": THREAD_ID, "contract": contract}
    identity = execution_identity(initial, config)
    references = (
        f"runs/{RUN_ID}/experiment-04c/experiment_spec.json",
        f"runs/{RUN_ID}/experiment-04c/evaluation_result.json",
        f"runs/{RUN_ID}/experiment-04c/dataset_manifest.json",
        f"runs/{RUN_ID}/experiment-04c/evaluator_manifest.json",
    )
    evaluation = EvaluationResult(
        experiment_id="experiment-04c",
        success=True,
        primary_score=-0.025,
        metrics={
            "stability": 0.975,
            "checkpoint_identities": {
                "evaluation_reuse_v2": EVALUATION_REUSE_IDENTITY,
                "verification_reuse": VERIFICATION_REUSE_IDENTITY,
                "attestation_id": "aura-professional-checkpoint-04",
                "attestation_hash": ATTESTATION_HASH,
            },
        },
        constraint_results={"clinical_pass": False, "floors_pass": True},
        artefact_references=references,
        evaluator_version="icca-adapter-v1.2:pinned",
        provenance=ProvenanceKind.REAL,
    )
    verification = VerificationResult(
        experiment_id="experiment-04c",
        verified=True,
        claimed_score=-0.025,
        measured_score=-0.025,
        constraint_compliant=False,
        evidence_status=EvidenceStatus.REFUTED,
        reasons=("icca_eligibility_gate_failed",),
        provenance=ProvenanceKind.REAL,
    )
    state = {
        "run_id": RUN_ID,
        "thread_id": THREAD_ID,
        "contract": contract,
        "execution_identity": identity,
        "status": RunStatus.COMPLETED,
        "cycle": 1,
        "budget": BudgetState(
            maximum_cycles=1,
            maximum_experiments=1,
            maximum_cost=2.0,
            cycles_used=1,
            experiments_used=1,
            cost_used=0.0,
            evaluator_cost_used=0.0,
            model_cost_used=0.0,
        ),
        "evaluation_result": evaluation,
        "verification_result": verification,
        "decision_event_ids": [
            "event-hypothesis",
            "event-plan",
            "event-experiment",
            "event-evaluation",
            "event-verification",
        ],
        "executed_nodes": ["verify_evidence", "stop_run"],
        "errors": [],
        "knowledge_errors": [],
        "knowledge_warnings": ["DATABASE_CREDENTIAL_NOT_ENFORCED_READ_ONLY"],
        "stop_reason": "maximum_cycles_reached",
    }
    return state, initial, config


def _write_checkpoint(path: Path, state: dict) -> None:
    saver, connection = sqlite_checkpointer(path)
    try:
        checkpoint = empty_checkpoint()
        versions = {name: "1" for name in state}
        checkpoint["channel_values"] = state
        checkpoint["channel_versions"] = versions
        saver.put(
            {"configurable": {"thread_id": THREAD_ID, "checkpoint_ns": ""}},
            checkpoint,
            {},
            versions,
        )
    finally:
        connection.close()


@pytest.mark.parametrize("mode", tuple(ReadSafetyMode))
def test_read_safety_mode_round_trips_with_exact_enum_identity(mode):
    serializer = checkpoint_serializer()

    restored = serializer.loads_typed(serializer.dumps_typed(mode))

    assert restored is mode
    assert type(restored) is ReadSafetyMode
    assert restored == mode
    assert hash(restored) == hash(mode)


def test_read_safety_type_graph_is_narrowly_allowlisted():
    assert (
        "auto_researcher.contracts.enums",
        "ReadSafetyMode",
    ) in ALLOWED_CHECKPOINT_TYPES
    assert not any(
        module == "auto_researcher.knowledge.read_safety"
        for module, _ in ALLOWED_CHECKPOINT_TYPES
    )


def _enum_payload(module: str, name: str, value: str) -> tuple[str, bytes]:
    body = ormsgpack.packb([module, name, value])
    return "msgpack", ormsgpack.packb(ormsgpack.Ext(0, body))


def test_unallowlisted_and_malformed_types_are_not_instantiated():
    serializer = checkpoint_serializer()

    assert serializer.loads_typed(_enum_payload("os", "system", "echo")) == "echo"
    assert serializer.loads_typed(_enum_payload("not.a.module", "Unknown", "X")) == "X"
    assert (
        serializer.loads_typed(
            ("msgpack", ormsgpack.packb(ormsgpack.Ext(0, ormsgpack.packb(["bad"]))))
        )
        is None
    )
    assert (
        serializer.loads_typed(
            _enum_payload(
                "auto_researcher.contracts.enums",
                "ReadSafetyMode",
                "NOT_A_READ_SAFETY_MODE",
            )
        )
        is None
    )


def test_arbitrary_classes_callables_and_unallowlisted_subclasses_are_rejected():
    serializer = checkpoint_serializer()

    class ArbitraryClass:
        pass

    class ArbitraryEnum(StrEnum):
        VALUE = "VALUE"

    class ContractSubclass(ResearchContract):
        pass

    with pytest.raises(TypeError):
        serializer.dumps_typed(ArbitraryClass())
    with pytest.raises(TypeError):
        serializer.dumps_typed(lambda: None)
    assert (
        serializer.loads_typed(serializer.dumps_typed(ArbitraryEnum.VALUE)) == "VALUE"
    )
    restored = serializer.loads_typed(
        serializer.dumps_typed(
            ContractSubclass.model_validate(
                _operator_contract().model_dump(mode="python")
            )
        )
    )
    assert type(restored) is not ContractSubclass


def test_provider_configuration_and_attestation_types_are_not_instantiated():
    serializer = checkpoint_serializer()
    restored = serializer.loads_typed(serializer.dumps_typed(operator_configuration()))

    assert type(restored) is dict
    assert type(restored["read_safety_attestation"]) is dict
    assert restored["read_safety_mode"] == "OPERATOR_ATTESTED"
    assert restored["read_safety_attestation"]["credential_class"] == (
        "MANAGED_INSTANCE_PRIMARY"
    )


def test_operator_terminal_checkpoint_reconstructs_in_fresh_process_without_side_effects(
    tmp_path,
):
    checkpoint_path = tmp_path / "checkpoint-04c.sqlite"
    state, initial, config = _terminal_fixture()
    expected_state_hash = payload_hash(state)
    expected_identity = state["execution_identity"]
    expected_database_hash = (
        hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if checkpoint_path.exists()
        else None
    )
    assert expected_database_hash is None
    _write_checkpoint(checkpoint_path, state)
    expected_database_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    expected_mtime = checkpoint_path.stat().st_mtime_ns

    script = r"""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from auto_researcher.contracts.enums import ReadSafetyMode
from auto_researcher.runtime.checkpoints import sqlite_checkpointer
from auto_researcher.runtime.execution import RunExecutionError, inspect_terminal_run, resume_run, start_run
from auto_researcher.runtime.identity import payload_hash

path = Path(sys.argv[1])
expected_state_hash = sys.argv[2]
saver, connection = sqlite_checkpointer(path)
calls = {name: 0 for name in (
    "provider_construction", "neo4j", "model_construction", "claude",
    "evaluator_construction", "evaluator", "verifier_construction", "verifier",
    "graph_nodes", "artefact_writes", "provenance_appends", "checkpoint_writes",
    "evaluation_reuse_writes", "verification_reuse_writes",
)}

class View:
    def get_state(self, config):
        item = saver.get_tuple(config)
        values = item.checkpoint["channel_values"] if item else {}
        return SimpleNamespace(values=values)

    def invoke(self, *args, **kwargs):
        calls["graph_nodes"] += 1
        raise AssertionError("execution guard allowed graph invocation")

view = View()
config = {"configurable": {"thread_id": "checkpoint-04c-fixture-thread"}}
before = hashlib.sha256(path.read_bytes()).hexdigest()
before_mtime = path.stat().st_mtime_ns
restored = inspect_terminal_run(view, config)
contract = restored["contract"]
initial = {"run_id": restored["run_id"], "thread_id": restored["thread_id"], "contract": contract}

def code_for(payload, operation="start"):
    try:
        if operation == "start":
            start_run(view, payload, config)
        else:
            resume_run(view, config)
    except RunExecutionError as exc:
        return exc.code
    raise AssertionError("execution guard did not reject")

codes = {
    "duplicate": code_for(initial),
    "run": code_for({**initial, "run_id": "different-run"}),
    "contract": code_for({**initial, "contract": contract.model_copy(update={"question": "changed"})}),
    "task": code_for({**initial, "contract": contract.model_copy(update={"task_id": "different-task"})}),
    "input": code_for({**initial, "operator_request": "changed"}),
    "resume": code_for(None, "resume"),
}
after = hashlib.sha256(path.read_bytes()).hexdigest()
after_mtime = path.stat().st_mtime_ns
result = {
    "state_hash": payload_hash(restored),
    "expected_state_hash": expected_state_hash,
    "identity": restored["execution_identity"].model_dump(mode="json"),
    "contract_hash": restored["execution_identity"].contract_hash,
    "initial_input_hash": restored["execution_identity"].initial_input_hash,
    "operator_attested_exact": any(
        mode is ReadSafetyMode.OPERATOR_ATTESTED
        for mode in contract.grounding.permitted_read_safety_modes
    ),
    "all_modes_exact": all(
        type(mode) is ReadSafetyMode
        for mode in contract.grounding.permitted_read_safety_modes
    ),
    "evaluation_identity": restored["evaluation_result"].metrics["checkpoint_identities"]["evaluation_reuse_v2"],
    "verification_identity": restored["evaluation_result"].metrics["checkpoint_identities"]["verification_reuse"],
    "attestation_identity": restored["evaluation_result"].metrics["checkpoint_identities"]["attestation_id"],
    "artefact_references": list(restored["evaluation_result"].artefact_references),
    "provenance_sequence": list(restored["decision_event_ids"]),
    "codes": codes,
    "calls": calls,
    "database_hash_unchanged": before == after,
    "database_mtime_unchanged": before_mtime == after_mtime,
}
connection.close()
print(json.dumps(result, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(checkpoint_path), expected_state_hash],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(completed.stdout)

    assert result["state_hash"] == expected_state_hash
    assert result["identity"] == expected_identity.model_dump(mode="json")
    assert result["contract_hash"] == expected_identity.contract_hash
    assert result["initial_input_hash"] == expected_identity.initial_input_hash
    assert result["operator_attested_exact"] is True
    assert result["all_modes_exact"] is True
    assert result["evaluation_identity"] == EVALUATION_REUSE_IDENTITY
    assert result["verification_identity"] == VERIFICATION_REUSE_IDENTITY
    assert result["attestation_identity"] == "aura-professional-checkpoint-04"
    assert result["artefact_references"] == list(
        state["evaluation_result"].artefact_references
    )
    assert result["provenance_sequence"] == state["decision_event_ids"]
    assert result["codes"] == {
        "duplicate": "thread_already_exists_use_resume_or_inspect",
        "run": "conflicting_run_identity",
        "contract": "conflicting_contract_identity",
        "task": "conflicting_task_identity",
        "input": "conflicting_initial_input_identity",
        "resume": "thread_is_terminal_use_inspect",
    }
    assert set(result["calls"].values()) == {0}
    assert result["database_hash_unchanged"] is True
    assert result["database_mtime_unchanged"] is True
    assert (
        hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        == expected_database_hash
    )
    assert checkpoint_path.stat().st_mtime_ns == expected_mtime


def test_operator_contract_hash_is_stable_across_python_hash_seeds():
    expected = payload_hash(_operator_contract())
    script = """
from tests.unit.test_checkpoint_serialization import _operator_contract
from auto_researcher.runtime.identity import payload_hash
print(payload_hash(_operator_contract()))
"""

    for seed in ("17", "131"):
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[2],
            env={**__import__("os").environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        )
        assert completed.stdout.strip() == expected


def test_reconstructed_nonterminal_checkpoint_resumes_with_exact_enum(tmp_path):
    from auto_researcher.graph.builder import build_graph
    from auto_researcher.runtime.dependencies import task_sqlite_dependencies
    from auto_researcher.runtime.execution import resume_run, start_run

    contract = default_synthetic_contract().model_copy(
        update={
            "grounding": default_synthetic_contract().grounding.model_copy(
                update={
                    "permitted_read_safety_modes": frozenset(
                        {ReadSafetyMode.OPERATOR_ATTESTED}
                    )
                }
            )
        }
    )
    checkpoint = tmp_path / "checkpoints.sqlite"
    provenance = tmp_path / "provenance.sqlite"
    agent_calls = tmp_path / "agent-calls.sqlite"
    knowledge = tmp_path / "knowledge.sqlite"
    context = TaskRuntimeContext(run_id="resume-operator", output_dir=tmp_path)
    config = {"configurable": {"thread_id": "resume-operator-thread"}}
    initial = {
        "run_id": "resume-operator",
        "thread_id": "resume-operator-thread",
        "contract": contract,
    }

    with task_sqlite_dependencies(
        SyntheticTask(),
        context,
        contract,
        default_synthetic_configuration(),
        checkpoint,
        provenance,
        agent_calls_path=agent_calls,
        knowledge_retrievals_path=knowledge,
    ) as dependencies:
        paused = start_run(
            build_graph(dependencies, interrupt_after=["plan_search"]),
            initial,
            config,
        )
        assert paused["status"] == RunStatus.RUNNING

    with task_sqlite_dependencies(
        SyntheticTask(),
        context,
        contract,
        default_synthetic_configuration(),
        checkpoint,
        provenance,
        agent_calls_path=agent_calls,
        knowledge_retrievals_path=knowledge,
    ) as reconstructed:
        final = resume_run(build_graph(reconstructed), config)

    assert final["status"] == RunStatus.COMPLETED
    restored_modes = final["contract"].grounding.permitted_read_safety_modes
    assert restored_modes == frozenset({ReadSafetyMode.OPERATOR_ATTESTED})
    assert all(type(mode) is ReadSafetyMode for mode in restored_modes)


def test_checkpoint_identity_validation_rejects_string_substitution_and_untyped_models():
    state, initial, config = _terminal_fixture()

    class View:
        def __init__(self, values):
            self.values = values

        def get_state(self, config):
            return type("Snapshot", (), {"values": self.values})()

    from auto_researcher.runtime.execution import RunExecutionError, validate_start_run

    substituted_contract = state["contract"].model_copy(
        update={
            "grounding": state["contract"].grounding.model_copy(
                update={"permitted_read_safety_modes": frozenset({"OPERATOR_ATTESTED"})}
            )
        }
    )
    for changed in (
        {**state, "contract": substituted_contract},
        {**state, "contract": state["contract"].model_dump(mode="python")},
        {
            **state,
            "execution_identity": state["execution_identity"].model_dump(mode="python"),
        },
    ):
        with pytest.raises(
            RunExecutionError, match="checkpoint_execution_identity_invalid"
        ):
            validate_start_run(View(changed), initial, config)
