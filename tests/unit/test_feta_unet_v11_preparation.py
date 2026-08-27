import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from auto_researcher.contracts.enums import SearchType
from auto_researcher.contracts.models import ResearchContract, SearchRequest
from auto_researcher.graph.nodes.hypothesis import (
    _deterministic_portfolio_hypothesis,
)
from auto_researcher.tasks import TaskRuntimeContext, default_task_registry
from auto_researcher.tasks.feta_seg.manifests import EXPECTED_MANIFEST_HASH
from auto_researcher.tasks.feta_seg.splits import (
    EXPECTED_FOLD_HASH,
    EXPECTED_SPLIT_HASH,
)
from auto_researcher.tasks.feta_seg.transforms import PREPROCESSING_VERSION
from auto_researcher.tasks.feta_unet_ensemble.cross_validation import (
    load_cross_validation_member_source,
)
from auto_researcher.tasks.feta_unet_search.configuration import (
    FeTAUNetSearchConfiguration,
    normalise_search_configuration,
)
from auto_researcher.tasks.feta_unet_search.portfolio import (
    V11_PORTFOLIO_VERSION,
    V11PortfolioPolicy,
    apply_portfolio_policy,
)
from auto_researcher.tasks.feta_unet_search.v11_preflight import (
    static_v11_preflight,
)

EXAMPLES = Path("examples/tasks/feta_unet_search")
CONFIG = EXAMPLES / "campaign-72h-v11-template.yaml"
CONTRACT = EXAMPLES / "contract-72h-v11.yaml"
EVIDENCE = EXAMPLES / "v11-bound-evidence.json"


def _inputs():
    configuration = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    contract = ResearchContract.model_validate(
        yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    )
    context = TaskRuntimeContext(task_options=configuration["runtime"]["options"])
    return configuration, contract, context


def test_v11_configuration_scope_is_explicit_and_identity_bearing():
    _, _, context = _inputs()
    policy = V11PortfolioPolicy.from_runtime(context)
    candidate = FeTAUNetSearchConfiguration.model_validate(policy.roots[0])
    canonical = normalise_search_configuration(candidate.model_dump(mode="json"))
    assert canonical["profile"] == "five_fold_confirmation"
    assert canonical["fold_count"] == 5

    legacy = normalise_search_configuration({"maximum_epochs": 25})
    assert "profile" not in legacy
    assert "fold_count" not in legacy

    with pytest.raises(ValueError, match="validation_scope_invalid"):
        FeTAUNetSearchConfiguration(profile="five_fold_confirmation", fold_count=1)


def test_v11_contract_and_portfolio_compile_first_root_without_search():
    _, contract, context = _inputs()
    task = default_task_registry().get("feta_unet_search")
    task.validate_contract(contract)
    policy = V11PortfolioPolicy.from_runtime(context)
    request = SearchRequest(
        request_id="seed",
        hypothesis_id="hypothesis",
        search_type=SearchType.DIRECT,
        target=contract.primary_metric,
        search_space={},
        experiment_budget=1,
        rationale="test",
    )
    projected = apply_portfolio_policy(
        request,
        run_id="v11-test",
        cycle=1,
        events=(),
        runtime_context=context,
    )
    assert projected is not None
    assert projected.search_type == SearchType.DIRECT
    assert projected.rationale.startswith(V11_PORTFOLIO_VERSION)
    assert dict(projected.search_space) == policy.roots[0]
    assert task.estimate_search_duration_seconds(projected, context) == 45_000.0


def test_v11_hypothesis_boundary_is_deterministic_and_value_free():
    _, contract, context = _inputs()
    task = default_task_registry().get("feta_unet_search")
    dependencies = SimpleNamespace(runtime_context=context, task=task)
    hypothesis = _deterministic_portfolio_hypothesis(
        {
            "run_id": "v11-test",
            "cycle": 1,
            "contract": contract,
        },
        dependencies,
    )
    assert hypothesis is not None
    assert hypothesis.prompt_version == "deterministic-campaign-confirmation-v1"
    assert hypothesis.predicted_subspace["profile"] == "five_fold_confirmation"
    assert hypothesis.agent_call_id is None


def test_v11_static_preflight_is_fail_closed_and_holdout_safe():
    report = static_v11_preflight(
        config_path=CONFIG,
        contract_path=CONTRACT,
        evidence_path=EVIDENCE,
    )
    assert report["root_count"] == 4
    assert report["fold_count_per_root"] == 5
    assert report["oof_development_subjects"] == 68
    assert report["holdout_subjects_evaluated"] == 0
    assert report["model_calls_performed"] == 0
    assert report["launch_ready"] is False


def test_v11_cross_validation_member_binds_all_five_checkpoints(tmp_path):
    _, _, context = _inputs()
    root = V11PortfolioPolicy.from_runtime(context).roots[0]
    experiment_id = "experiment-v11-test"
    checkpoint_paths = {}
    references = []
    for fold in range(5):
        path = tmp_path / f"fold-{fold}" / "best.pt"
        path.parent.mkdir()
        path.write_bytes(f"checkpoint-{fold}".encode())
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        checkpoint_paths[str(fold)] = str(path)
        references.append(
            {
                "fold": fold,
                "relative_path": f"checkpoints/fold-{fold}/best.pt",
                "sha256": sha,
            }
        )
    specification_path = tmp_path / "experiment_spec.json"
    result_path = tmp_path / "evaluation_result.json"
    specification_path.write_text(json.dumps({"experiment_id": experiment_id}))
    result_path.write_text(
        json.dumps(
            {
                "success": True,
                "experiment_id": experiment_id,
                "primary_score": 0.8,
                "metrics": {
                    "configuration": root,
                    "configuration_identity": "1" * 64,
                    "architecture_identity": "v11-test-architecture",
                    "dataset_manifest_hash": EXPECTED_MANIFEST_HASH,
                    "split_hash": EXPECTED_SPLIT_HASH,
                    "fold_hash": EXPECTED_FOLD_HASH,
                    "preprocessing_version": PREPROCESSING_VERSION,
                    "inference_identity": "v11-test-inference",
                    "checkpoint_references": references,
                    "folds_completed": 5,
                    "oof_subject_count": 68,
                    "holdout_subjects_evaluated": 0,
                    "contains_subject_identifiers": False,
                    "validation_scope": "five-fold-confirmation-oof",
                },
            }
        )
    )
    source = load_cross_validation_member_source(
        {
            "experiment_id": experiment_id,
            "checkpoint_paths": checkpoint_paths,
            "experiment_spec_path": str(specification_path),
            "evaluation_result_path": str(result_path),
        }
    )
    assert len(source.checkpoint_paths) == 5
    assert len(source.checkpoint_sha256s) == 5
    assert source.configuration.fold_count == 5
