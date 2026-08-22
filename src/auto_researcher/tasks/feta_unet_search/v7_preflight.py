"""Fail-closed static and real-CUDA preflight for the V7 mechanism campaign."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import yaml

from auto_researcher.contracts.models import ResearchContract
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
    V7_ARCHITECTURE_BUDGET,
    V7_MAXIMUM_PEAK_GPU_MEMORY_BYTES,
    V7_MAXIMUM_TRAINABLE_PARAMETERS,
    V7_MINIMUM_TRAINABLE_PARAMETERS,
    FeTAUNetSearchConfiguration,
)
from auto_researcher.tasks.feta_unet_search.portfolio import (
    V7_MECHANISM_PORTFOLIO_VERSION,
    V7MechanismPortfolioPolicy,
)
from auto_researcher.tasks.feta_unet_search.task import FeTAUNetSearchTask
from auto_researcher.tasks.models import TaskRuntimeContext

V7_PREFLIGHT_SCHEMA_VERSION = "feta-unet-v7-real-cuda-preflight-v1"
MAXIMUM_EXISTING_GPU_ALLOCATION_BYTES = 1024**3
WORST_CASE_GRADUATION_SOURCE_EPOCH = 25
GRADUATING_FINALIST_COUNT = 2


def _mapping(path: Path, reason: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(reason) from exc
    if not isinstance(value, dict):
        raise ValueError(reason)
    return value


def build_v7_preflight_plan(
    task_config_path: Path,
    contract_path: Path,
) -> dict[str, Any]:
    """Validate immutable campaign identities without touching CUDA or secrets."""

    raw_config = _mapping(
        task_config_path, "feta_unet_v7_preflight_configuration_invalid"
    )
    raw_experiment = raw_config.get("experiment")
    if not isinstance(raw_experiment, dict) or raw_config.get("search") is not None:
        raise ValueError("feta_unet_v7_preflight_direct_launch_shape_invalid")
    scientific_seed = dict(raw_experiment)
    raw_openevolve = scientific_seed.pop("openevolve", None)
    if (
        not isinstance(raw_openevolve, dict)
        or isinstance(raw_openevolve.get("maximum_candidate_evaluations"), bool)
        or not isinstance(raw_openevolve.get("maximum_candidate_evaluations"), int)
        or raw_openevolve["maximum_candidate_evaluations"] <= 0
        or isinstance(raw_openevolve.get("maximum_wall_time_seconds"), bool)
        or not isinstance(raw_openevolve.get("maximum_wall_time_seconds"), int)
        or raw_openevolve["maximum_wall_time_seconds"] <= 0
    ):
        raise ValueError("feta_unet_v7_preflight_openevolve_controls_invalid")
    initial_configuration = FeTAUNetSearchConfiguration.model_validate(scientific_seed)
    raw_runtime = raw_config.get("runtime")
    if not isinstance(raw_runtime, dict):
        raise ValueError("feta_unet_v7_preflight_configuration_invalid")
    raw_options = raw_runtime.get("options")
    if not isinstance(raw_options, dict):
        raise ValueError("feta_unet_v7_preflight_configuration_invalid")
    if (
        raw_options.get("maximum_peak_gpu_memory_bytes")
        != V7_MAXIMUM_PEAK_GPU_MEMORY_BYTES
    ):
        raise ValueError("feta_unet_v7_preflight_memory_ceiling_invalid")
    raw_rates = raw_options.get("campaign_seconds_per_epoch_by_model_variant")
    reporting_reserve = raw_options.get("campaign_reporting_reserve_seconds")
    finalisation_reserve = raw_options.get("campaign_finalisation_reserve_seconds")
    if (
        not isinstance(raw_rates, dict)
        or isinstance(raw_rates.get("structural_basic_unet"), bool)
        or not isinstance(raw_rates.get("structural_basic_unet"), (int, float))
        or float(raw_rates["structural_basic_unet"]) <= 0
        or isinstance(reporting_reserve, bool)
        or not isinstance(reporting_reserve, (int, float))
        or float(reporting_reserve) < 0
        or isinstance(finalisation_reserve, bool)
        or not isinstance(finalisation_reserve, (int, float))
    ):
        raise ValueError("feta_unet_v7_preflight_graduation_budget_invalid")
    required_graduation_reserve = GRADUATING_FINALIST_COUNT * (
        150 - WORST_CASE_GRADUATION_SOURCE_EPOCH
    ) * float(raw_rates["structural_basic_unet"]) + float(reporting_reserve)
    if float(finalisation_reserve) < required_graduation_reserve:
        raise ValueError("feta_unet_v7_preflight_graduation_budget_invalid")
    calibration = raw_options.get("inference_calibration")
    if calibration != {
        "enabled": True,
        "finalist_count": 2,
        "maximum_variants": 8,
        "overlaps": [0.25, 0.5, 0.75],
        "blending_modes": ["gaussian", "constant"],
        "flip_tta": [False, True],
        "class_specific_postprocessing": "diagnostic_gated",
    }:
        raise ValueError("feta_unet_v7_preflight_calibration_plan_invalid")
    environment = raw_runtime.get("environment")
    if (
        not isinstance(environment, dict)
        or str(environment.get("CUDA_VISIBLE_DEVICES", "")).strip() == ""
    ):
        raise ValueError("feta_unet_v7_preflight_visible_gpu_invalid")

    contract = ResearchContract.model_validate(
        _mapping(contract_path, "feta_unet_v7_preflight_contract_invalid")
    )
    FeTAUNetSearchTask().validate_contract(contract)
    policy = V7MechanismPortfolioPolicy.from_runtime(
        TaskRuntimeContext(task_options=raw_options)
    )
    first_root = FeTAUNetSearchConfiguration.model_validate(policy.structural_roots[0])
    if initial_configuration != first_root:
        raise ValueError("feta_unet_v7_preflight_direct_seed_invalid")
    roots: list[dict[str, Any]] = []
    identities: set[str] = set()
    for index, raw_root in enumerate(policy.structural_roots):
        configuration = FeTAUNetSearchConfiguration.model_validate(raw_root)
        model = create_unet_model(configuration)
        parameters = trainable_parameter_count(model)
        identity = architecture_identity(configuration)
        del model
        if not (
            configuration.architecture_budget == V7_ARCHITECTURE_BUDGET
            and V7_MINIMUM_TRAINABLE_PARAMETERS
            <= parameters
            <= V7_MAXIMUM_TRAINABLE_PARAMETERS
            and identity not in identities
        ):
            raise ValueError("feta_unet_v7_preflight_structural_root_invalid")
        identities.add(identity)
        roots.append(
            {
                "root_index": index,
                "architecture_identity": identity,
                "feature_width": configuration.feature_width,
                "kernel_profile": configuration.kernel_profile,
                "residual_blocks": configuration.residual_blocks,
                "deep_supervision_heads": configuration.deep_supervision_heads,
                "trainable_parameters": parameters,
            }
        )
    return {
        "schema_version": V7_PREFLIGHT_SCHEMA_VERSION,
        "portfolio_version": V7_MECHANISM_PORTFOLIO_VERSION,
        "contract_id": contract.contract_id,
        "maximum_peak_gpu_memory_bytes": V7_MAXIMUM_PEAK_GPU_MEMORY_BYTES,
        "required_graduation_reserve_seconds": required_graduation_reserve,
        "configured_graduation_reserve_seconds": float(finalisation_reserve),
        "graduating_finalist_count": GRADUATING_FINALIST_COUNT,
        "inference_calibration": calibration,
        "visible_gpu": str(environment["CUDA_VISIBLE_DEVICES"]),
        "initial_search_type": "DIRECT",
        "openevolve_candidate_evaluation_limit": raw_openevolve[
            "maximum_candidate_evaluations"
        ],
        "root_count": len(roots),
        "v6_parent_evidence_count": len(policy.v6_parent_evidence),
        "v6_parent_experiment_ids": [
            item["experiment_id"] for item in policy.v6_parent_evidence
        ],
        "req11_diagnostic_bound": True,
        "req11_panel_identity": policy.req11_diagnostic["panel_identity"],
        "req11_case_count": policy.req11_diagnostic["case_count"],
        "req11_priorities": policy.req11_diagnostic["priorities"],
        "roots": roots,
        "static_preflight_passed": True,
        "cuda_preflight_passed": False,
        "model_calls_performed": 0,
    }


def run_v7_cuda_preflight(
    task_config_path: Path,
    contract_path: Path,
) -> dict[str, Any]:
    """Run one AMP forward/backward step for every frozen root on real CUDA."""

    plan = build_v7_preflight_plan(task_config_path, contract_path)
    raw_config = _mapping(
        task_config_path, "feta_unet_v7_preflight_configuration_invalid"
    )
    options = raw_config["runtime"]["options"]
    policy = V7MechanismPortfolioPolicy.from_runtime(
        TaskRuntimeContext(task_options=options)
    )
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("feta_unet_v7_preflight_torch_unavailable") from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("feta_unet_v7_preflight_single_cuda_required")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    if total_bytes < V7_MAXIMUM_PEAK_GPU_MEMORY_BYTES:
        raise RuntimeError("feta_unet_v7_preflight_gpu_capacity_insufficient")
    if total_bytes - free_bytes > MAXIMUM_EXISTING_GPU_ALLOCATION_BYTES:
        raise RuntimeError("feta_unet_v7_preflight_gpu_not_idle")

    root_results: list[dict[str, Any]] = []
    for index, raw_root in enumerate(policy.structural_roots):
        configuration = FeTAUNetSearchConfiguration.model_validate(raw_root)
        model = None
        optimizer = None
        inputs = None
        target = None
        output = None
        loss = None
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
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(inputs)
                if not bool(torch.isfinite(output).all()):
                    raise ValueError("feta_unet_v7_preflight_prediction_non_finite")
                loss = deep_supervision_training_loss(
                    output,
                    target,
                    loss_function,
                    configuration,
                )
            if not bool(torch.isfinite(loss)):
                raise ValueError("feta_unet_v7_preflight_loss_non_finite")
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            peak = int(torch.cuda.max_memory_allocated())
            if peak > V7_MAXIMUM_PEAK_GPU_MEMORY_BYTES:
                raise RuntimeError("feta_unet_v7_preflight_memory_ceiling_exceeded")
            root_results.append(
                {
                    **plan["roots"][index],
                    "peak_gpu_memory_bytes": peak,
                    "full_amp_step_passed": True,
                }
            )
        except torch.OutOfMemoryError as exc:
            raise RuntimeError("feta_unet_v7_preflight_cuda_out_of_memory") from exc
        finally:
            del loss, output, target, inputs, optimizer, model
            gc.collect()
            torch.cuda.empty_cache()

    return {
        **plan,
        "device_name": torch.cuda.get_device_name(0),
        "device_total_memory_bytes": int(total_bytes),
        "roots": root_results,
        "cuda_preflight_passed": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("static", "cuda"), required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args(argv)
    value = (
        build_v7_preflight_plan(args.task_config, args.contract)
        if args.mode == "static"
        else run_v7_cuda_preflight(args.task_config, args.contract)
    )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
