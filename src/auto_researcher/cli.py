"""Generic task-oriented command line interface."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import typer
import yaml

from auto_researcher.agents.call_store import SQLiteAgentCallStore
from auto_researcher.agents.models import (
    AgentBudgetPolicy,
    ModelCallConfig,
    ModelPricing,
)
from auto_researcher.contracts.enums import (
    KnowledgeGroundingMode,
    RunStatus,
    SearchType,
)
from auto_researcher.contracts.models import ResearchContract
from auto_researcher.graph.builder import build_graph
from auto_researcher.knowledge.models import KnowledgeProviderConfiguration
from auto_researcher.knowledge.providers.neo4j import Neo4jKnowledgeProvider
from auto_researcher.knowledge.providers.static import StaticKnowledgeProvider
from auto_researcher.knowledge.schemas.knowledge_graph_auto_v0_1 import (
    KnowledgeGraphAutoProfile,
)
from auto_researcher.knowledge.store import SQLiteKnowledgeRetrievalStore
from auto_researcher.knowledge.templates import default_template_registry
from auto_researcher.provenance.sqlite_store import SQLiteProvenanceStore
from auto_researcher.runtime.dependencies import (
    task_sqlite_dependencies,
    utc_now,
)
from auto_researcher.tasks import TaskRuntimeContext, default_task_registry
from auto_researcher.tasks.models import TaskPluginError
from auto_researcher.tasks.synthetic import (
    default_synthetic_configuration,
    default_synthetic_contract,
)

app = typer.Typer(no_args_is_help=True, help="Run task plugins on Auto Researcher.")
agent_calls_app = typer.Typer(
    no_args_is_help=True,
    help="Inspect or explicitly retry durable live-agent calls.",
)
app.add_typer(agent_calls_app, name="agent-calls")
knowledge_app = typer.Typer(
    no_args_is_help=True,
    help="Inspect providers and durable knowledge retrievals.",
)
knowledge_retrievals_app = typer.Typer(
    no_args_is_help=True,
    help="Inspect or explicitly retry knowledge retrievals.",
)
app.add_typer(knowledge_app, name="knowledge")
knowledge_app.add_typer(knowledge_retrievals_app, name="retrievals")
DEFAULT_DATA_DIR = Path(".auto-researcher")


def _mock_contract(max_cycles: int) -> ResearchContract:
    """PR 1 compatibility alias for the default synthetic contract."""
    return default_synthetic_contract(max_cycles)


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def _load_task_configuration(
    path: Path | None,
    task_id: str,
    task_version: str,
) -> tuple[dict, dict]:
    if path is None:
        if task_id != "synthetic":
            raise ValueError("--task-config is required for non-synthetic tasks")
        return default_synthetic_configuration(), {}
    payload = _load_yaml(path)
    task_identity = payload.get("task", {})
    if task_identity.get("id") != task_id:
        raise ValueError(
            f"task config identifies {task_identity.get('id')!r}, not {task_id!r}"
        )
    if str(task_identity.get("version")) != task_version:
        raise ValueError(
            "task config identifies version "
            f"{task_identity.get('version')!r}, not {task_version!r}"
        )
    experiment = payload.get("experiment")
    search = payload.get("search")
    runtime = payload.get("runtime", {})
    if (experiment is None) == (search is None):
        raise ValueError("task config requires exactly one of experiment or search")
    if not isinstance(runtime, dict):
        raise ValueError("task config runtime section must be a mapping")
    selected = experiment if experiment is not None else search
    if not isinstance(selected, dict):
        raise ValueError("task config experiment/search section must be a mapping")
    selected = dict(selected)
    if search is not None:
        kind = selected.pop("type", None)
        if kind != SearchType.OPTUNA.value:
            raise ValueError("PR 3 search section type must be OPTUNA")
    return selected, runtime


def _configured_search_type(path: Path | None) -> SearchType:
    if path is None:
        return SearchType.DIRECT
    payload = _load_yaml(path)
    return SearchType.OPTUNA if "search" in payload else SearchType.DIRECT


def _load_live_agents(payload: dict[str, Any]):
    configured = payload.get("agents", {"mode": "mock"})
    if not isinstance(configured, dict):
        raise ValueError("agents section must be a mapping")
    mode = configured.get("mode", "mock")
    if mode == "mock":
        return None, None, None, None, AgentBudgetPolicy(), "mock"
    if mode != "live":
        raise ValueError("agents.mode must be 'mock' or 'live'")
    provider = configured.get("provider")
    model_id = configured.get("model_id")
    if not isinstance(provider, str) or not isinstance(model_id, str):
        raise ValueError("live agents require explicit provider and model_id")
    pricing_payload = configured.get("pricing")
    if not isinstance(pricing_payload, dict):
        raise ValueError("live agents require an explicit versioned pricing mapping")
    pricing = ModelPricing.model_validate(pricing_payload)
    policy_payload = configured.get("budget", {})
    if not isinstance(policy_payload, dict):
        raise ValueError("agents.budget must be a mapping")
    policy = AgentBudgetPolicy.model_validate(policy_payload)

    def call_config(role: str, default_temperature: float) -> ModelCallConfig:
        role_payload = configured.get(role, {})
        if not isinstance(role_payload, dict):
            raise ValueError(f"agents.{role} must be a mapping")
        return ModelCallConfig(
            provider=provider,
            model_id=model_id,
            temperature=role_payload.get("temperature", default_temperature),
            maximum_output_tokens=role_payload.get("maximum_output_tokens"),
            timeout_seconds=role_payload.get("timeout_seconds"),
            maximum_attempts=role_payload.get("maximum_attempts"),
            maximum_cost_per_call=role_payload.get("maximum_cost_per_call"),
            pricing=pricing,
            prompt_version=role_payload.get("prompt_version", "2.0.0"),
            structured_output_strategy=role_payload.get(
                "structured_output_strategy",
                "pydantic",
            ),
        )

    hypothesis_config = call_config("hypothesis", 0.2)
    planner_config = call_config("planner", 0.0)
    if provider.casefold() == "anthropic":
        from auto_researcher.providers.anthropic import create_anthropic_client

        hypothesis_client = create_anthropic_client(hypothesis_config)
        planner_client = create_anthropic_client(planner_config)
    else:
        raise ValueError(
            f"unsupported live provider {provider!r}; live mode implements 'anthropic'"
        )
    return (
        hypothesis_client,
        planner_client,
        hypothesis_config,
        planner_config,
        policy,
        "live",
    )


def _load_grounding(
    payload: dict[str, Any],
    contract: ResearchContract,
):
    raw = payload.get("grounding")
    if raw is None:
        if contract.grounding.mode != KnowledgeGroundingMode.DISABLED:
            raise ValueError("enabled contract grounding requires a grounding section")
        return None, None
    if not isinstance(raw, dict):
        raise ValueError("grounding section must be a mapping")
    prohibited = {
        "uri",
        "username",
        "password",
        "credentials",
        "neo4j_uri",
        "neo4j_username",
        "neo4j_password",
    }
    if prohibited & _nested_keys(raw):
        raise ValueError("grounding credentials and URI must come from the environment")
    mode = KnowledgeGroundingMode(raw.get("mode", contract.grounding.mode.value))
    if mode != contract.grounding.mode:
        raise ValueError("runtime grounding mode must match the research contract")
    if mode == KnowledgeGroundingMode.DISABLED:
        return None, None
    provider_id = raw.get("provider")
    if not isinstance(provider_id, str) or not provider_id:
        raise ValueError("enabled grounding requires an explicit provider")
    allowed_tiers = frozenset(
        str(item)
        for item in raw.get(
            "allowed_trust_tiers",
            contract.grounding.permitted_trust_tiers,
        )
    )
    if not allowed_tiers.issubset(contract.grounding.permitted_trust_tiers):
        raise ValueError("runtime grounding trust tiers weaken the contract")
    confidence = float(
        raw.get(
            "minimum_assertion_confidence",
            contract.grounding.minimum_assertion_confidence,
        )
    )
    if confidence < contract.grounding.minimum_assertion_confidence:
        raise ValueError("runtime grounding confidence threshold weakens the contract")
    configuration = KnowledgeProviderConfiguration(
        provider_id=provider_id,
        graph_alias=raw.get("graph_alias", provider_id),
        database=raw.get("database") or os.getenv("NEO4J_DATABASE", "neo4j"),
        schema_version=raw.get(
            "schema_version",
            contract.grounding.knowledge_schema_version,
        ),
        content_version=raw.get(
            "content_version",
            contract.grounding.knowledge_content_version,
        ),
        query_timeout_seconds=raw.get(
            "query_timeout_seconds",
            contract.grounding.maximum_retrieval_duration,
        ),
        maximum_records=raw.get(
            "maximum_records",
            contract.grounding.maximum_query_records,
        ),
        maximum_attempts=raw.get("maximum_attempts", 2),
        minimum_assertion_confidence=confidence,
        allowed_trust_tiers=allowed_tiers,
        require_verified_read_only=raw.get("require_verified_read_only", True),
        enabled=raw.get("enabled", True),
    )
    templates = default_template_registry()
    if provider_id == "static":
        provider = StaticKnowledgeProvider(configuration, clock=utc_now)
    elif provider_id == "neo4j":
        provider = Neo4jKnowledgeProvider(
            configuration,
            templates,
            uri=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            clock=utc_now,
            schema_profile=(
                KnowledgeGraphAutoProfile()
                if configuration.schema_version == KnowledgeGraphAutoProfile.profile_id
                else None
            ),
        )
    else:
        raise ValueError(f"unknown knowledge provider {provider_id!r}")
    return configuration, provider


def _nested_keys(value) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key).casefold())
            keys.update(_nested_keys(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            keys.update(_nested_keys(item))
    return keys


@app.command()
def run(
    task_id: str = typer.Option("synthetic", "--task"),
    contract_path: Path | None = typer.Option(None, "--contract"),
    task_config: Path | None = typer.Option(None, "--task-config"),
    mock: bool = typer.Option(False, "--mock", help="Alias for --task synthetic."),
    run_id: str = typer.Option(..., "--run-id"),
    thread_id: str = typer.Option(..., "--thread-id"),
    max_cycles: int = typer.Option(1, "--max-cycles", min=1),
    checkpoint_db: Path = typer.Option(
        DEFAULT_DATA_DIR / "checkpoints.sqlite",
        "--checkpoint-db",
    ),
    provenance_db: Path = typer.Option(
        DEFAULT_DATA_DIR / "provenance.sqlite",
        "--provenance-db",
    ),
    optuna_db: Path = typer.Option(
        DEFAULT_DATA_DIR / "optuna.sqlite",
        "--optuna-db",
    ),
    agent_calls_db: Path = typer.Option(
        DEFAULT_DATA_DIR / "agent-calls.sqlite",
        "--agent-calls-db",
    ),
    knowledge_retrievals_db: Path = typer.Option(
        DEFAULT_DATA_DIR / "knowledge-retrievals.sqlite",
        "--knowledge-retrievals-db",
    ),
) -> None:
    """Run a registered task through the unchanged LangGraph control plane."""
    if mock:
        task_id = "synthetic"
    registry = default_task_registry()
    try:
        search_type = _configured_search_type(task_config)
        raw_config = _load_yaml(task_config) if task_config else {}
        (
            model_client,
            planner_model_client,
            hypothesis_call_config,
            planner_call_config,
            agent_budget_policy,
            agent_mode,
        ) = _load_live_agents(raw_config)
        requested_budget = int(
            raw_config.get("search", {}).get("trial_budget", max_cycles)
        )
        contract = (
            ResearchContract.model_validate(_load_yaml(contract_path))
            if contract_path
            else default_synthetic_contract(
                max_cycles,
                search_types=frozenset({search_type}),
                maximum_experiments=requested_budget,
            )
        )
        if contract.task_id != task_id:
            raise ValueError(
                f"contract targets task {contract.task_id!r}, not {task_id!r}"
            )
        task = registry.get(task_id, contract.task_version)
        knowledge_configuration, knowledge_provider = _load_grounding(
            raw_config,
            contract,
        )
        experiment, runtime = _load_task_configuration(
            task_config,
            task_id,
            contract.task_version,
        )
        runtime_options = dict(runtime.get("options", {}))
        if isinstance(raw_config.get("grounding"), dict):
            runtime_options["grounding"] = dict(raw_config["grounding"])
        runtime_context = TaskRuntimeContext(
            run_id=run_id,
            data_dir=runtime.get("data_dir"),
            workspace_dir=runtime.get("workspace_dir"),
            output_dir=runtime.get("output_dir", DEFAULT_DATA_DIR),
            environment=runtime.get("environment", {}),
            task_options=runtime_options,
        )
        config = {"configurable": {"thread_id": thread_id}}
        with task_sqlite_dependencies(
            task,
            runtime_context,
            contract,
            experiment,
            checkpoint_db,
            provenance_db,
            optuna_db if search_type == SearchType.OPTUNA else None,
            agent_calls_db,
            knowledge_retrievals_db,
            model_client=model_client,
            planner_model_client=planner_model_client,
            hypothesis_call_config=hypothesis_call_config,
            planner_call_config=planner_call_config,
            agent_budget_policy=agent_budget_policy,
            knowledge_provider=knowledge_provider,
            knowledge_configuration=knowledge_configuration,
            search_type=search_type,
        ) as dependencies:
            graph = build_graph(dependencies)
            final = graph.invoke(
                {"run_id": run_id, "thread_id": thread_id, "contract": contract},
                config,
            )
    except (TaskPluginError, ValueError, FileNotFoundError, RuntimeError) as exc:
        typer.echo(f"Task setup failed: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    verification = final.get("verification_result")
    evaluation = final.get("evaluation_result")
    typer.echo(f"Task: {contract.task_id}@{contract.task_version}")
    typer.echo(f"Search type: {search_type.value}")
    typer.echo(f"Agent mode: {agent_mode}")
    typer.echo(f"Grounding mode: {contract.grounding.mode.value}")
    typer.echo(
        "Knowledge provider: "
        f"{knowledge_configuration.provider_id if knowledge_configuration else 'none'}"
    )
    if knowledge_configuration is not None:
        typer.echo(f"Graph alias: {knowledge_configuration.graph_alias}")
        typer.echo(f"Knowledge schema: {knowledge_configuration.schema_version}")
        typer.echo(f"Knowledge content: {knowledge_configuration.content_version}")
    if agent_mode == "live":
        assert hypothesis_call_config is not None and planner_call_config is not None
        typer.echo(
            "Live model: "
            f"{hypothesis_call_config.provider}/{hypothesis_call_config.model_id}"
        )
        typer.echo(
            "Prompt versions: "
            f"hypothesis@{hypothesis_call_config.prompt_version}, "
            f"planner@{planner_call_config.prompt_version}"
        )
    typer.echo(f"Run: {run_id}")
    typer.echo(f"Status: {final['status'].value}")
    typer.echo(
        f"Cycles: {final['budget'].cycles_used}/{final['budget'].maximum_cycles}"
    )
    typer.echo(
        "Model usage: "
        f"calls={final['budget'].model_calls_used} "
        f"input_tokens={final['budget'].model_input_tokens_used} "
        f"output_tokens={final['budget'].model_output_tokens_used} "
        f"cache_creation_tokens={final['budget'].model_cache_creation_tokens_used} "
        f"cache_read_tokens={final['budget'].model_cache_read_tokens_used} "
        f"cost={final['budget'].model_cost_used}"
    )
    typer.echo(f"Evaluator cost: {final['budget'].evaluator_cost_used}")
    typer.echo(f"Total cost: {final['budget'].cost_used}")
    hypothesis = final.get("active_hypothesis")
    knowledge_reference = final.get("knowledge_bundle_reference")
    typer.echo(
        f"Knowledge retrieval: {final.get('knowledge_retrieval_status', 'DISABLED')}"
    )
    if knowledge_reference is not None:
        typer.echo(f"Knowledge bundle: {knowledge_reference.bundle_id or 'none'}")
        typer.echo(
            f"Knowledge bundle hash: {knowledge_reference.bundle_hash or 'none'}"
        )
        typer.echo(
            f"Valid knowledge references: {len(knowledge_reference.reference_ids)}"
        )
        typer.echo(f"Knowledge trust tiers: {dict(knowledge_reference.trust_summary)}")
        typer.echo(
            f"Knowledge artefact: {knowledge_reference.artefact_reference or 'none'}"
        )
    if hypothesis is not None:
        typer.echo(f"Grounding: {hypothesis.grounding_status.value}")
        typer.echo(
            "Cited knowledge references: "
            f"{', '.join(hypothesis.evidence_references) or 'none'}"
        )
    typer.echo(f"Primary score: {evaluation.primary_score if evaluation else 'n/a'}")
    typer.echo(
        "Evidence: "
        f"{verification.evidence_status.value if verification else 'n/a'} "
        f"({verification.provenance.value if verification else 'n/a'})"
    )
    typer.echo(f"Stop reason: {final.get('stop_reason') or 'none'}")
    typer.echo(f"Event IDs: {', '.join(final['decision_event_ids']) or 'none'}")
    study = final.get("optuna_study_result")
    if study is not None:
        typer.echo(f"Study: {study.study_name}")
        typer.echo(f"Direction: {study.direction.value}")
        typer.echo(f"Trial budget: {study.trial_budget}")
        typer.echo(f"Trials asked: {study.trials_asked}")
        typer.echo(f"Trials completed: {study.trials_completed}")
        typer.echo(f"Trials failed: {study.trials_failed}")
        typer.echo(f"Best feasible score: {study.best_feasible_score}")
        typer.echo(f"Best feasible trial: {study.best_feasible_trial_number}")
        typer.echo(f"Best overall diagnostic score: {study.best_overall_score}")
        typer.echo(f"Study finish reason: {study.finish_reason}")
        typer.echo(f"Study artefacts: {', '.join(study.artefact_references) or 'none'}")
    if final["status"] == RunStatus.FAILED:
        raise typer.Exit(code=1)


@app.command("tasks")
def list_tasks() -> None:
    """List registered task descriptors and readiness without importing v2 eagerly."""
    registry = default_task_registry()
    for task in registry.list_tasks():
        descriptor = task.descriptor()
        readiness = task.readiness(TaskRuntimeContext())
        searches = ",".join(
            sorted(item.value for item in descriptor.supported_search_types)
        )
        typer.echo(
            f"{descriptor.task_id}\t{descriptor.task_version}\t"
            f"{descriptor.display_name}\tready={str(readiness.ready).lower()}\t"
            f"search={searches}"
        )


@app.command()
def provenance(
    run_id: str = typer.Option(..., "--run-id"),
    provenance_db: Path = typer.Option(
        DEFAULT_DATA_DIR / "provenance.sqlite",
        "--provenance-db",
    ),
) -> None:
    """Print ordered append-only provenance events for a run."""
    store = SQLiteProvenanceStore(provenance_db)
    try:
        events = store.list_events(run_id)
    finally:
        store.close()
    if not events:
        typer.echo(f"No provenance events found for run {run_id!r}.")
        raise typer.Exit(code=1)
    for event in events:
        typer.echo(
            f"{event.timestamp.isoformat()} "
            f"{event.event_id} {event.event_type.value} "
            f"actor={event.actor} provenance={event.provenance.value}"
        )


@agent_calls_app.command("list")
def list_agent_calls(
    run_id: str = typer.Option(..., "--run-id"),
    agent_calls_db: Path = typer.Option(
        DEFAULT_DATA_DIR / "agent-calls.sqlite",
        "--agent-calls-db",
    ),
) -> None:
    """List append-only model-call snapshots for one run."""
    store = SQLiteAgentCallStore(agent_calls_db)
    try:
        records = store.list_records(run_id)
    finally:
        store.close()
    if not records:
        typer.echo(f"No agent calls found for run {run_id!r}.")
        raise typer.Exit(code=1)
    for record in records:
        typer.echo(
            f"{record.created_at.isoformat()} {record.call_id} "
            f"role={record.role.value} status={record.status.value} "
            f"provider={record.provider} model={record.model_id} "
            f"attempts={record.attempt_count} cost={record.estimated_cost}"
        )


@agent_calls_app.command("show")
def show_agent_call(
    call_id: str = typer.Option(..., "--call-id"),
    agent_calls_db: Path = typer.Option(
        DEFAULT_DATA_DIR / "agent-calls.sqlite",
        "--agent-calls-db",
    ),
) -> None:
    """Show safe snapshots for a call; rendered prompts are never stored."""
    store = SQLiteAgentCallStore(agent_calls_db)
    try:
        records = store.records_for_call(call_id)
    finally:
        store.close()
    if not records:
        typer.echo(f"No agent call found for {call_id!r}.")
        raise typer.Exit(code=1)
    for record in records:
        typer.echo(record.model_dump_json(indent=2))


@agent_calls_app.command("retry")
def retry_agent_call(
    call_id: str = typer.Option(..., "--call-id"),
    agent_calls_db: Path = typer.Option(
        DEFAULT_DATA_DIR / "agent-calls.sqlite",
        "--agent-calls-db",
    ),
) -> None:
    """Authorise one linked retry of an indeterminate paid-call reservation."""
    from auto_researcher.runtime.dependencies import utc_now

    store = SQLiteAgentCallStore(agent_calls_db)
    try:
        retry = store.create_retry(call_id, created_at=utc_now())
    except (KeyError, ValueError) as exc:
        typer.echo(f"Retry not authorised: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    finally:
        store.close()
    typer.echo(
        f"Authorised retry {retry.call_id} linked to {retry.retry_of_call_id}. "
        "Resume the same LangGraph thread to execute it."
    )


@knowledge_app.command("providers")
def list_knowledge_providers() -> None:
    """List built-in provider adapters without opening any connection."""
    typer.echo("static\tinstalled=true\toffline=true")
    typer.echo(
        "neo4j\tinstalled="
        f"{str(importlib.util.find_spec('neo4j') is not None).lower()}\toffline=false"
    )


@knowledge_app.command("readiness")
def knowledge_readiness(
    task_id: str = typer.Option(..., "--task"),
    contract_path: Path = typer.Option(..., "--contract"),
    task_config: Path = typer.Option(..., "--task-config"),
) -> None:
    """Check task, configuration, connectivity, and read-only readiness."""
    provider = None
    try:
        contract = ResearchContract.model_validate(_load_yaml(contract_path))
        if contract.task_id != task_id:
            raise ValueError("contract task does not match --task")
        default_task_registry().get(task_id, contract.task_version)
        raw = _load_yaml(task_config)
        configuration, provider = _load_grounding(raw, contract)
        if configuration is None or provider is None:
            typer.echo("Grounding is explicitly disabled.")
            return
        result = provider.readiness(configuration)
    except (TaskPluginError, ValueError, RuntimeError) as exc:
        typer.echo(f"Knowledge readiness failed: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    finally:
        if provider is not None:
            provider.close()
    typer.echo(f"Task: {task_id}@{contract.task_version}")
    typer.echo(f"Provider: {result.provider_id}@{result.provider_version}")
    typer.echo(f"Schema: {result.schema_version}")
    typer.echo(f"Content: {result.content_version}")
    typer.echo(f"Ready: {str(result.ready).lower()}")
    for check in result.checks:
        typer.echo(f"{check.code}\tpassed={str(check.passed).lower()}\t{check.message}")
    for warning in result.warnings:
        typer.echo(f"warning\t{warning}")
    for error in result.errors:
        typer.echo(f"error\t{error.value}")
    if not result.ready:
        raise typer.Exit(code=1)


@knowledge_retrievals_app.command("list")
def list_knowledge_retrievals(
    run_id: str = typer.Option(..., "--run-id"),
    knowledge_retrievals_db: Path = typer.Option(
        DEFAULT_DATA_DIR / "knowledge-retrievals.sqlite",
        "--knowledge-retrievals-db",
    ),
) -> None:
    """List append-only knowledge retrieval snapshots for one run."""
    store = SQLiteKnowledgeRetrievalStore(knowledge_retrievals_db)
    try:
        records = store.list_records(run_id)
    finally:
        store.close()
    if not records:
        typer.echo(f"No knowledge retrievals found for run {run_id!r}.")
        raise typer.Exit(code=1)
    for record in records:
        bundle = record.bundle
        typer.echo(
            f"{record.created_at.isoformat()} {record.retrieval_id} "
            f"status={record.status.value} "
            f"provider={record.request.provider_id} "
            f"bundle={bundle.bundle_id if bundle else 'none'} "
            f"references={len(bundle.references) if bundle else 0}"
        )


@knowledge_retrievals_app.command("show")
def show_knowledge_retrieval(
    retrieval_id: str = typer.Option(..., "--retrieval-id"),
    knowledge_retrievals_db: Path = typer.Option(
        DEFAULT_DATA_DIR / "knowledge-retrievals.sqlite",
        "--knowledge-retrievals-db",
    ),
) -> None:
    """Show safe identity, policy, status, and artefact metadata."""
    store = SQLiteKnowledgeRetrievalStore(knowledge_retrievals_db)
    try:
        records = store.records_for_retrieval(retrieval_id)
    finally:
        store.close()
    if not records:
        typer.echo(f"No knowledge retrieval found for {retrieval_id!r}.")
        raise typer.Exit(code=1)
    for record in records:
        bundle = record.bundle
        typer.echo(
            yaml.safe_dump(
                {
                    "created_at": record.created_at.isoformat(),
                    "retrieval_id": record.retrieval_id,
                    "status": record.status.value,
                    "retry_of": record.retry_of_retrieval_id,
                    "provider": record.request.provider_id,
                    "graph_alias": record.request.graph_alias,
                    "schema_version": record.request.schema_version,
                    "content_version": record.request.content_version,
                    "query_plan_hash": record.request.query_plan_hash,
                    "templates": [
                        f"{item.template_id}@{item.template_version}"
                        for item in record.request.query_plan.template_requests
                    ],
                    "errors": [item.value for item in record.errors],
                    "bundle_id": bundle.bundle_id if bundle else None,
                    "bundle_hash": bundle.bundle_hash if bundle else None,
                    "reference_ids": (
                        [item.reference_id for item in bundle.references]
                        if bundle
                        else []
                    ),
                    "trust_tiers": (
                        dict(bundle.validation_result.trust_tier_summary)
                        if bundle
                        else {}
                    ),
                    "artefacts": list(bundle.artefact_references) if bundle else [],
                },
                sort_keys=False,
            ).rstrip()
        )


@knowledge_retrievals_app.command("retry")
def retry_knowledge_retrieval(
    retrieval_id: str = typer.Option(..., "--retrieval-id"),
    knowledge_retrievals_db: Path = typer.Option(
        DEFAULT_DATA_DIR / "knowledge-retrievals.sqlite",
        "--knowledge-retrievals-db",
    ),
) -> None:
    """Authorise one linked retry of an indeterminate external read."""
    store = SQLiteKnowledgeRetrievalStore(knowledge_retrievals_db)
    try:
        retry = store.create_retry(retrieval_id, created_at=utc_now())
    except (KeyError, ValueError) as exc:
        typer.echo(f"Retry not authorised: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    finally:
        store.close()
    typer.echo(
        f"Authorised retry {retry.retrieval_id} linked to "
        f"{retry.retry_of_retrieval_id}. Resume the same LangGraph thread."
    )


if __name__ == "__main__":
    app()
