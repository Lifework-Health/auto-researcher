"""Fail-closed static readiness checks for the provisional V9 campaign."""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from auto_researcher.runtime.identity import payload_hash
from auto_researcher.tasks.feta_unet_direct.model import (
    architecture_identity,
    create_unet_model,
    trainable_parameter_count,
)
from auto_researcher.tasks.feta_unet_direct.trainer import (
    create_loss,
    create_optimizer,
    deep_supervision_training_loss,
)
from auto_researcher.tasks.feta_unet_search.configuration import (
    V9_CONFIGURATION_SCHEMA_VERSION,
    V9_ATTENTION_ARCHITECTURE_BUDGET,
    V9_MAXIMUM_PEAK_GPU_MEMORY_BYTES,
    V9_MAXIMUM_TRAINABLE_PARAMETERS,
    V9_MINIMUM_TRAINABLE_PARAMETERS,
    V9_TRANSFORMER_ARCHITECTURE_BUDGET,
    FeTAUNetSearchConfiguration,
)
from auto_researcher.tasks.feta_unet_search.v9_research_intelligence import (
    build_v9_knowledge_library,
    build_v9_literature_brief,
)
from auto_researcher.research_intelligence import LiteratureScoutMode

V9_PREFLIGHT_SCHEMA_VERSION = "feta-unet-v9-static-preflight-v1"
V9_CUDA_CALIBRATION_SCHEMA_VERSION = "feta-unet-v9-cuda-calibration-v1"
V9_BOUND_EVIDENCE_SCHEMA_VERSION = "feta-unet-v9-bound-evidence-v1"
V9_EXPECTED_DURATION_HOURS = 36
V9_EXPECTED_FINALISATION_HOURS = 8
V9_EXPECTED_FIDELITIES = [15, 30, 50, 100, 150]
V9_EXPECTED_FAMILY_FRACTIONS = {
    "dynunet": 0.7,
    "attention_unet": 0.2,
    "transformer_pilots": 0.1,
}
V9_EXPECTED_ROOT_COUNTS = {
    "dynunet": 6,
    "attention_unet": 2,
    "unetr": 1,
    "swin_unetr": 1,
}
MAXIMUM_EXISTING_GPU_ALLOCATION_BYTES = 1024**3


def _load(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError("feta_unet_v9_input_invalid") from exc
    if not isinstance(raw, dict):
        raise ValueError("feta_unet_v9_input_invalid")
    return raw


def _hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_bound_evidence(raw: dict[str, Any]) -> tuple[str, ...]:
    if raw.get("schema_version") != V9_BOUND_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("feta_unet_v9_bound_evidence_schema_invalid")
    if raw.get("development_fold") != 0 or raw.get("sealed_holdout_evaluations") != 0:
        raise ValueError("feta_unet_v9_holdout_boundary_invalid")
    for name in ("dataset_manifest_hash", "split_hash", "fold_hash"):
        if not _hash(raw.get(name)):
            raise ValueError("feta_unet_v9_scientific_identity_invalid")
    sources = raw.get("source_artifacts")
    if not isinstance(sources, dict) or not all(
        _hash(item) for item in sources.values()
    ):
        raise ValueError("feta_unet_v9_source_artifact_invalid")
    campaign = raw.get("v8_campaign")
    ensemble = raw.get("v8_ensemble")
    if not isinstance(campaign, dict) or not isinstance(ensemble, dict):
        raise ValueError("feta_unet_v9_prior_evidence_invalid")
    champion = campaign.get("champion")
    if (
        not isinstance(champion, dict)
        or champion.get("model_variant") != "dynunet"
        or champion.get("origin_search_type") != "OPTUNA"
        or champion.get("best_score") != 0.8260410594691577
        or campaign.get("verified_execution_count") != 86
        or ensemble.get("mean_subject_macro_dice") != 0.8303179703378446
        or ensemble.get("delta_vs_best_single") <= 0
        or ensemble.get("sealed_holdout_evaluations") != 0
    ):
        raise ValueError("feta_unet_v9_prior_evidence_invalid")
    intelligence = raw.get("research_intelligence")
    knowledge = build_v9_knowledge_library()
    literature = build_v9_literature_brief(mode=LiteratureScoutMode.LIVE)
    if (
        not isinstance(intelligence, dict)
        or intelligence.get("knowledge_library_id") != knowledge.library_id
        or intelligence.get("knowledge_library_hash") != knowledge.library_hash
        or intelligence.get("literature_brief_id") != literature.brief_id
        or intelligence.get("literature_brief_hash") != literature.brief_hash
        or intelligence.get("source_count") != len(literature.evidence)
        or intelligence.get("experiment_authority_exercised") is not False
    ):
        raise ValueError("feta_unet_v9_research_intelligence_invalid")
    pilots = raw.get("v9_architecture_pilots")
    profiles = pilots.get("profiles") if isinstance(pilots, dict) else None
    if (
        not isinstance(profiles, list)
        or len(profiles) != 4
        or pilots.get("full_cuda_step_performed") is not True
        or pilots.get("calibration_sha256")
        != "fc1e9dbe57e423e308674d81d32e72a9ecafcb5dceee0816f6654ab7ff384d73"
        or pilots.get("gpu") != "NVIDIA RTX A6000"
        or pilots.get("holdout_subjects_evaluated") != 0
        or pilots.get("model_calls_performed") != 0
        or {item.get("feature_width") for item in profiles if isinstance(item, dict)}
        != {
            "v9_attn_compact_5",
            "v9_attn_balanced_5",
            "v9_unetr_base_16",
            "v9_swin_tiny_24",
        }
        or not all(
            isinstance(item, dict)
            and V9_MINIMUM_TRAINABLE_PARAMETERS
            <= item.get("trainable_parameters", 0)
            <= V9_MAXIMUM_TRAINABLE_PARAMETERS
            and isinstance(item.get("architecture_identity"), str)
            and item.get("measured_full_step_passed") is True
            and 0
            < item.get("peak_gpu_memory_bytes", 0)
            <= V9_MAXIMUM_PEAK_GPU_MEMORY_BYTES
            and item.get("amp_step_seconds", 0) > 0
            for item in profiles
        )
    ):
        raise ValueError("feta_unet_v9_architecture_pilot_evidence_invalid")
    design = raw.get("provisional_campaign_design")
    if (
        not isinstance(design, dict)
        or design.get("duration_hours") != V9_EXPECTED_DURATION_HOURS
        or design.get("protected_finalisation_hours") != V9_EXPECTED_FINALISATION_HOURS
        or design.get("family_budget_fraction") != V9_EXPECTED_FAMILY_FRACTIONS
        or design.get("fixed_root_count") != V9_EXPECTED_ROOT_COUNTS
        or design.get("fidelity_ladder") != V9_EXPECTED_FIDELITIES
        or design.get("minimum_100_epoch_finalists") < 3
        or design.get("target_150_epoch_finalists") < 2
    ):
        raise ValueError("feta_unet_v9_campaign_design_invalid")
    blockers = raw.get("launch_blockers")
    if (
        raw.get("launch_ready") is not False
        or not isinstance(blockers, list)
        or not blockers
    ):
        raise ValueError("feta_unet_v9_launch_boundary_invalid")
    return tuple(str(item) for item in blockers)


def _validate_template(raw: dict[str, Any]) -> tuple[FeTAUNetSearchConfiguration, ...]:
    options = raw.get("runtime", {}).get("options", {})
    if not isinstance(options, dict):
        raise ValueError("feta_unet_v9_runtime_options_invalid")
    budget = raw.get("agents", {}).get("budget", {})
    director = raw.get("agents", {}).get("research_director", {})
    if (
        budget.get("maximum_research_director_valid_decisions_total") != 16
        or budget.get("maximum_total_model_calls") != 768
        or director.get("model_id") != "claude-opus-5"
        or director.get("effort") != "xhigh"
    ):
        raise ValueError("feta_unet_v9_research_director_budget_invalid")
    calibration = options.get("v9_cuda_calibration")
    if (
        options.get("launch_gate") != "blocked_pending_v9_validation"
        or options.get("v9_portfolio_controller_implemented") is not False
        or not isinstance(calibration, dict)
        or calibration.get("schema_version") != V9_CUDA_CALIBRATION_SCHEMA_VERSION
        or calibration.get("calibration_sha256")
        != "fc1e9dbe57e423e308674d81d32e72a9ecafcb5dceee0816f6654ab7ff384d73"
        or calibration.get("calibration_file_sha256")
        != "0fb60cf98dd14f9202f331bdd7f4d2fe88a7e70522ecb50c6891e305368ecd38"
        or calibration.get("pilot_count") != 4
        or calibration.get("maximum_peak_gpu_memory_bytes") != 9176368640
        or calibration.get("holdout_subjects_evaluated") != 0
        or calibration.get("model_calls_performed") != 0
        or calibration.get("passed") is not True
        or options.get("maximum_peak_gpu_memory_bytes")
        != V9_MAXIMUM_PEAK_GPU_MEMORY_BYTES
    ):
        raise ValueError("feta_unet_v9_launch_boundary_invalid")
    roots = options.get("v9_fixed_roots")
    if not isinstance(roots, list) or len(roots) != sum(
        V9_EXPECTED_ROOT_COUNTS.values()
    ):
        raise ValueError("feta_unet_v9_root_portfolio_invalid")
    validated = tuple(
        FeTAUNetSearchConfiguration.model_validate(item) for item in roots
    )
    counts: dict[str, int] = {}
    for item in validated:
        counts[item.model_variant] = counts.get(item.model_variant, 0) + 1
        if item.maximum_epochs != 15:
            raise ValueError("feta_unet_v9_root_fidelity_invalid")
    if counts != V9_EXPECTED_ROOT_COUNTS:
        raise ValueError("feta_unet_v9_root_portfolio_invalid")
    if not all(
        item.architecture_budget
        in {
            "dynunet-15m-150m-v1",
            V9_ATTENTION_ARCHITECTURE_BUDGET,
            V9_TRANSFORMER_ARCHITECTURE_BUDGET,
        }
        for item in validated
    ):
        raise ValueError("feta_unet_v9_root_portfolio_invalid")
    return validated


def static_v9_preflight(*, config_path: Path, evidence_path: Path) -> dict[str, Any]:
    evidence = _load(evidence_path)
    blockers = _validate_bound_evidence(evidence)
    roots = _validate_template(_load(config_path))
    payload = {
        "schema_version": V9_PREFLIGHT_SCHEMA_VERSION,
        "configuration_schema_version": V9_CONFIGURATION_SCHEMA_VERSION,
        "bound_evidence_sha256": payload_hash(evidence),
        "root_count": len(roots),
        "root_model_variant_counts": V9_EXPECTED_ROOT_COUNTS,
        "root_feature_widths": [item.feature_width for item in roots],
        "duration_hours": V9_EXPECTED_DURATION_HOURS,
        "protected_finalisation_hours": V9_EXPECTED_FINALISATION_HOURS,
        "cuda_calibration_sha256": (
            _load(config_path)["runtime"]["options"]["v9_cuda_calibration"]
        )["calibration_sha256"],
        "research_director_valid_decision_budget": 16,
        "knowledge_library_hash": build_v9_knowledge_library().library_hash,
        "literature_brief_hash": build_v9_literature_brief(
            mode=LiteratureScoutMode.LIVE
        ).brief_hash,
        "holdout_subjects_evaluated": 0,
        "model_calls_performed": 0,
        "launch_ready": False,
        "launch_blockers": blockers,
    }
    return {**payload, "preflight_sha256": payload_hash(payload)}


def _run_cuda_amp_step(
    configuration: FeTAUNetSearchConfiguration,
    torch: Any,
) -> dict[str, float | int]:
    model = optimizer = inputs = target = output = loss = None
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = create_unet_model(configuration).to("cuda")
        model.train()
        optimizer = create_optimizer(model, configuration)
        loss_function = create_loss(configuration)
        inputs = torch.randn(
            (
                configuration.batch_size,
                configuration.in_channels,
                *configuration.patch_size,
            ),
            device="cuda",
            dtype=torch.float32,
        )
        target = torch.randint(
            0,
            configuration.out_channels,
            (configuration.batch_size, 1, *configuration.patch_size),
            device="cuda",
            dtype=torch.int64,
        )
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(inputs)
            tensors = output if isinstance(output, (list, tuple)) else (output,)
            if not all(bool(torch.isfinite(item).all()) for item in tensors):
                raise ValueError("feta_unet_v9_preflight_prediction_non_finite")
            loss = deep_supervision_training_loss(
                output,
                target,
                loss_function,
                configuration,
            )
        if not bool(torch.isfinite(loss)):
            raise ValueError("feta_unet_v9_preflight_loss_non_finite")
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        peak = int(torch.cuda.max_memory_allocated())
        if peak > V9_MAXIMUM_PEAK_GPU_MEMORY_BYTES:
            raise RuntimeError("feta_unet_v9_preflight_memory_ceiling_exceeded")
        return {"peak_gpu_memory_bytes": peak, "amp_step_seconds": elapsed}
    except torch.OutOfMemoryError as exc:
        raise RuntimeError("feta_unet_v9_preflight_cuda_out_of_memory") from exc
    finally:
        del loss, output, target, inputs, optimizer, model
        gc.collect()
        torch.cuda.empty_cache()


def run_v9_cuda_calibration(
    *,
    config_path: Path,
    torch_module: Any | None = None,
    step_runner: Callable[[FeTAUNetSearchConfiguration], dict[str, float | int]]
    | None = None,
) -> dict[str, Any]:
    """Measure one full AMP step for every new V9 architecture pilot."""

    roots = _validate_template(_load(config_path))
    pilots = tuple(item for item in roots if item.model_variant != "dynunet")
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as exc:
            raise RuntimeError("feta_unet_v9_preflight_torch_unavailable") from exc
    torch = torch_module
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("feta_unet_v9_preflight_single_cuda_required")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    if total_bytes < V9_MAXIMUM_PEAK_GPU_MEMORY_BYTES:
        raise RuntimeError("feta_unet_v9_preflight_gpu_capacity_insufficient")
    if total_bytes - free_bytes > MAXIMUM_EXISTING_GPU_ALLOCATION_BYTES:
        raise RuntimeError("feta_unet_v9_preflight_gpu_not_idle")
    run_step = step_runner or (
        lambda configuration: _run_cuda_amp_step(configuration, torch)
    )

    measured = []
    for configuration in pilots:
        model = create_unet_model(configuration)
        parameters = trainable_parameter_count(model)
        identity = architecture_identity(configuration)
        del model
        if not (
            V9_MINIMUM_TRAINABLE_PARAMETERS
            <= parameters
            <= V9_MAXIMUM_TRAINABLE_PARAMETERS
        ):
            raise ValueError("feta_unet_v9_preflight_parameter_envelope_invalid")
        step = run_step(configuration)
        peak = int(step["peak_gpu_memory_bytes"])
        seconds = float(step["amp_step_seconds"])
        if peak <= 0 or peak > V9_MAXIMUM_PEAK_GPU_MEMORY_BYTES or seconds <= 0:
            raise RuntimeError("feta_unet_v9_preflight_measurement_invalid")
        measured.append(
            {
                "model_variant": configuration.model_variant,
                "feature_width": configuration.feature_width,
                "architecture_identity": identity,
                "trainable_parameters": parameters,
                "peak_gpu_memory_bytes": peak,
                "peak_gpu_memory_gib": peak / 1024**3,
                "amp_step_seconds": seconds,
                "measured_full_step_passed": True,
            }
        )
    payload = {
        "schema_version": V9_CUDA_CALIBRATION_SCHEMA_VERSION,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": int(total_bytes),
        "memory_ceiling_gib": V9_MAXIMUM_PEAK_GPU_MEMORY_BYTES / 1024**3,
        "holdout_subjects_evaluated": 0,
        "model_calls_performed": 0,
        "pilots": measured,
        "passed": True,
    }
    return {**payload, "calibration_sha256": payload_hash(payload)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("static", "cuda"), default="static")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.mode == "cuda":
        result = run_v9_cuda_calibration(config_path=arguments.config)
    else:
        if arguments.evidence is None:
            parser.error("--evidence is required in static mode")
        result = static_v9_preflight(
            config_path=arguments.config,
            evidence_path=arguments.evidence,
        )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
