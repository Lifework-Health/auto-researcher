"""Durable, fail-closed multi-fidelity continuation for FeTA search."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from auto_researcher.runtime.identity import payload_hash
from auto_researcher.tasks.feta_seg_search.configuration import (
    FIDELITY_LEVELS,
    FeTASegSearchConfiguration,
)

CONTINUATION_VERSION = "feta-search-stateful-optimisation-continuation-v1"
CONTINUATION_SEMANTICS = (
    "stateful optimisation continuation with deterministic fixed candidate seed"
)


@dataclass(frozen=True)
class ResumePlan:
    source_candidate_root: Path
    source_last_checkpoint: Path
    source_best_checkpoint: Path
    source_checkpoint_sha256: str
    source_best_checkpoint_sha256: str
    completed_epoch: int
    start_epoch: int
    trajectory_identity: str
    last_payload: dict[str, Any]
    best_payload: dict[str, Any]


def candidate_trajectory_payload(
    configuration: FeTASegSearchConfiguration,
) -> dict[str, Any]:
    """Bind every scientific setting except the promotion fidelity."""

    return configuration.model_dump(mode="json", exclude={"maximum_epochs"})


def candidate_trajectory_identity(
    configuration: FeTASegSearchConfiguration,
) -> str:
    return payload_hash(candidate_trajectory_payload(configuration))


def prediction_set_identity(
    trajectory_identity: str,
    epoch: int,
    score: float,
    validation_subject_ids: tuple[str, ...],
) -> str:
    return payload_hash(
        {
            "trajectory_identity": trajectory_identity,
            "epoch": epoch,
            "validation_score": score,
            "validation_subject_ids": list(validation_subject_ids),
        }
    )


def checkpoint_file_reference(
    path: Path,
    *,
    output_root: Path,
    checkpoint_type: str,
    completed_epoch: int,
    trajectory_identity: str,
) -> dict[str, Any]:
    resolved = path.resolve()
    relative = resolved.relative_to(output_root.resolve())
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return {
        "checkpoint_type": checkpoint_type,
        "relative_path": relative.as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": digest,
        "completed_epoch": completed_epoch,
        "trajectory_identity": trajectory_identity,
    }


def capture_rng_state(torch_module: Any, numpy_module: Any, generator: Any) -> dict[str, Any]:
    numpy_state = numpy_module.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": str(numpy_state[0]),
            "state": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch_module.get_rng_state(),
        "torch_cuda": torch_module.cuda.get_rng_state_all(),
        "data_loader_generator": generator.get_state(),
    }


def restore_rng_state(
    state: dict[str, Any], torch_module: Any, numpy_module: Any, generator: Any
) -> None:
    try:
        random.setstate(state["python"])
        numpy_state = state["numpy"]
        numpy_module.random.set_state(
            (
                numpy_state["bit_generator"],
                numpy_module.asarray(numpy_state["state"], dtype="uint32"),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
        torch_module.set_rng_state(state["torch_cpu"])
        torch_module.cuda.set_rng_state_all(state["torch_cuda"])
        generator.set_state(state["data_loader_generator"])
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("feta_search_resume_rng_state_invalid") from exc


def checkpoint_metadata_identity(payload: dict[str, Any]) -> str:
    names = (
        "completed_epoch",
        "fold",
        "seed",
        "configuration",
        "configuration_identity",
        "trajectory_identity",
        "runner_version",
        "data_loader_version",
        "continuation_version",
        "continuation_semantics",
        "best_epoch",
        "best_score",
        "best_checkpoint_sha256",
        "best_prediction_identity",
    )
    return payload_hash({name: payload[name] for name in names})


def build_last_checkpoint_payload(
    *,
    model_state_dict: Any,
    optimizer_state_dict: Any,
    scaler_state_dict: Any,
    completed_epoch: int,
    configuration: FeTASegSearchConfiguration,
    trajectory_identity: str,
    runner_version: str,
    data_loader_version: str,
    best_epoch: int,
    best_score: float,
    best_checkpoint_sha256: str,
    best_prediction_identity: str,
    rng_state: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "scaler_state_dict": scaler_state_dict,
        "completed_epoch": completed_epoch,
        "fold": configuration.fold,
        "seed": configuration.seed + configuration.fold,
        "configuration": configuration.scientific_configuration(),
        "configuration_identity": payload_hash(configuration),
        "trajectory_identity": trajectory_identity,
        "runner_version": runner_version,
        "data_loader_version": data_loader_version,
        "continuation_version": CONTINUATION_VERSION,
        "continuation_semantics": CONTINUATION_SEMANTICS,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_checkpoint_sha256": best_checkpoint_sha256,
        "best_prediction_identity": best_prediction_identity,
        "rng_state": rng_state,
    }
    payload["checkpoint_identity"] = checkpoint_metadata_identity(payload)
    return payload


def validate_resume_checkpoint_payload(
    payload: Any,
    requested: FeTASegSearchConfiguration,
    *,
    expected_runner_version: str,
    expected_data_loader_version: str,
) -> tuple[int, str]:
    if not isinstance(payload, dict):
        raise ValueError("feta_search_resume_checkpoint_invalid")
    required = {
        "model_state_dict",
        "optimizer_state_dict",
        "scaler_state_dict",
        "completed_epoch",
        "fold",
        "seed",
        "configuration",
        "configuration_identity",
        "trajectory_identity",
        "runner_version",
        "data_loader_version",
        "continuation_version",
        "continuation_semantics",
        "best_epoch",
        "best_score",
        "best_checkpoint_sha256",
        "best_prediction_identity",
        "rng_state",
        "checkpoint_identity",
    }
    if not required.issubset(payload):
        raise ValueError("feta_search_resume_checkpoint_invalid")
    if payload["checkpoint_identity"] != checkpoint_metadata_identity(payload):
        raise ValueError("feta_search_resume_checkpoint_identity_mismatch")
    if payload["continuation_version"] != CONTINUATION_VERSION or payload[
        "continuation_semantics"
    ] != CONTINUATION_SEMANTICS:
        raise ValueError("feta_search_resume_continuation_identity_mismatch")
    if (
        payload["runner_version"] != expected_runner_version
        or payload["data_loader_version"] != expected_data_loader_version
    ):
        raise ValueError("feta_search_resume_runtime_identity_mismatch")
    if payload["fold"] != 0 or payload["fold"] != requested.fold:
        raise ValueError("feta_search_resume_fold_mismatch")
    expected_seed = requested.seed + requested.fold
    if payload["seed"] != expected_seed:
        raise ValueError("feta_search_resume_seed_mismatch")
    try:
        raw_completed_epoch = payload["completed_epoch"]
        if isinstance(raw_completed_epoch, bool) or not isinstance(
            raw_completed_epoch, int
        ):
            raise TypeError("completed_epoch must be an integer")
        completed_epoch = raw_completed_epoch
        source_configuration = FeTASegSearchConfiguration.model_validate(
            payload["configuration"]
        )
    except Exception as exc:
        raise ValueError("feta_search_resume_configuration_invalid") from exc
    if completed_epoch not in FIDELITY_LEVELS:
        raise ValueError("feta_search_resume_source_fidelity_invalid")
    if source_configuration.maximum_epochs != completed_epoch:
        raise ValueError("feta_search_resume_source_fidelity_mismatch")
    if payload["configuration_identity"] != payload_hash(source_configuration):
        raise ValueError("feta_search_resume_configuration_identity_mismatch")
    expected_trajectory = candidate_trajectory_identity(requested)
    if (
        payload["trajectory_identity"] != expected_trajectory
        or candidate_trajectory_identity(source_configuration) != expected_trajectory
    ):
        raise ValueError("feta_search_resume_trajectory_mismatch")
    if completed_epoch >= requested.maximum_epochs:
        raise ValueError("feta_search_resume_fidelity_not_higher")
    best_epoch = payload["best_epoch"]
    best_score = payload["best_score"]
    if (
        isinstance(best_epoch, bool)
        or not isinstance(best_epoch, int)
        or not 1 <= best_epoch <= completed_epoch
        or isinstance(best_score, bool)
        or not isinstance(best_score, (int, float))
        or not math.isfinite(float(best_score))
        or not 0 <= float(best_score) <= 1
    ):
        raise ValueError("feta_search_resume_best_checkpoint_invalid")
    return completed_epoch + 1, expected_trajectory


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_resume_plan(
    source_candidate_root: Path,
    requested: FeTASegSearchConfiguration,
    *,
    expected_runner_version: str,
    expected_data_loader_version: str,
    map_location: str = "cuda",
) -> ResumePlan:
    root = source_candidate_root.expanduser().resolve()
    last_path = root / "checkpoints" / "last.pt"
    best_path = root / "checkpoints" / "best.pt"
    if not last_path.is_file():
        raise ValueError("feta_search_resume_checkpoint_missing")
    if not best_path.is_file():
        raise ValueError("feta_search_resume_best_checkpoint_missing")
    try:
        import torch

        last_payload = torch.load(
            last_path, map_location=map_location, weights_only=True
        )
    except ImportError as exc:
        raise RuntimeError("feta_ml_dependencies_unavailable") from exc
    except Exception as exc:
        raise ValueError("feta_search_resume_checkpoint_unreadable") from exc
    start_epoch, trajectory_identity = validate_resume_checkpoint_payload(
        last_payload,
        requested,
        expected_runner_version=expected_runner_version,
        expected_data_loader_version=expected_data_loader_version,
    )
    best_sha = _sha256(best_path)
    if best_sha != last_payload["best_checkpoint_sha256"]:
        raise ValueError("feta_search_resume_best_checkpoint_identity_mismatch")
    try:
        best_payload = torch.load(
            best_path, map_location=map_location, weights_only=True
        )
    except Exception as exc:
        raise ValueError("feta_search_resume_best_checkpoint_unreadable") from exc
    required_best = {
        "model_state_dict",
        "fold",
        "epoch",
        "validation_score",
        "seed",
        "trajectory_identity",
        "prediction_identity",
    }
    if not isinstance(best_payload, dict) or not required_best.issubset(best_payload):
        raise ValueError("feta_search_resume_best_checkpoint_invalid")
    try:
        best_score = float(best_payload["validation_score"])
        source_best_score = float(last_payload["best_score"])
    except (TypeError, ValueError) as exc:
        raise ValueError("feta_search_resume_best_checkpoint_invalid") from exc
    if not math.isfinite(best_score):
        raise ValueError("feta_search_resume_best_checkpoint_invalid")
    if (
        best_payload["fold"] != requested.fold
        or best_payload["seed"] != requested.seed + requested.fold
        or best_payload["trajectory_identity"] != trajectory_identity
        or best_payload["epoch"] != last_payload["best_epoch"]
        or not math.isclose(
            best_score,
            source_best_score,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or best_payload["prediction_identity"]
        != last_payload["best_prediction_identity"]
    ):
        raise ValueError("feta_search_resume_best_checkpoint_identity_mismatch")
    return ResumePlan(
        source_candidate_root=root,
        source_last_checkpoint=last_path,
        source_best_checkpoint=best_path,
        source_checkpoint_sha256=_sha256(last_path),
        source_best_checkpoint_sha256=best_sha,
        completed_epoch=start_epoch - 1,
        start_epoch=start_epoch,
        trajectory_identity=trajectory_identity,
        last_payload=last_payload,
        best_payload=best_payload,
    )
