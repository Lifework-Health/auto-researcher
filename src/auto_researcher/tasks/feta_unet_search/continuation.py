"""Durable staged-fidelity continuation for the FeTA BasicUNet campaign."""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from auto_researcher.runtime.identity import payload_hash
from auto_researcher.tasks.artifacts import atomic_json_write
from auto_researcher.tasks.feta_unet_search.configuration import (
    FIDELITY_LEVELS,
    FeTAUNetSearchConfiguration,
)

CONTINUATION_VERSION = "feta-unet-staged-fidelity-continuation-v1"
CONTINUATION_SEMANTICS = "same-candidate stateful continuation with epoch-indexed deterministic data order and augmentation"


@dataclass(frozen=True)
class ResumePlan:
    source_candidate_root: Path
    completed_epoch: int
    start_epoch: int
    best_epoch: int
    best_score: float
    trajectory_identity: str
    last_payload: dict[str, Any]
    best_payload: dict[str, Any]
    validation_history: tuple[dict[str, Any], ...]


def trajectory_payload(
    configuration: FeTAUNetSearchConfiguration,
) -> dict[str, Any]:
    """Bind every scientific setting except the maximum training fidelity."""

    return configuration.model_dump(mode="json", exclude={"maximum_epochs"})


def trajectory_identity(configuration: FeTAUNetSearchConfiguration) -> str:
    return payload_hash(trajectory_payload(configuration))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture_rng_state(torch_module: Any, numpy_module: Any, generator: Any) -> dict:
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
    state: dict, torch_module: Any, numpy_module: Any, generator: Any
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
        # The training checkpoint is loaded onto CUDA so model and optimizer
        # tensors are immediately usable. RNG byte tensors are different: the
        # CPU and DataLoader generators require CPU ByteTensors, and CUDA's RNG
        # setter also accepts its states from CPU memory.
        torch_module.set_rng_state(state["torch_cpu"].cpu())
        torch_module.cuda.set_rng_state_all(
            [item.cpu() for item in state["torch_cuda"]]
        )
        generator.set_state(state["data_loader_generator"].cpu())
    except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("feta_unet_resume_rng_state_invalid") from exc


def build_last_payload(
    *,
    model_state_dict: Any,
    optimizer_state_dict: Any,
    scaler_state_dict: Any,
    completed_epoch: int,
    configuration: FeTAUNetSearchConfiguration,
    best_epoch: int,
    best_score: float,
    best_checkpoint_sha256: str,
    runner_id: str,
    data_loader_id: str,
    rng_state: dict,
) -> dict[str, Any]:
    identity = trajectory_identity(configuration)
    payload = {
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "scheduler_state_dict": None,
        "scaler_state_dict": scaler_state_dict,
        "completed_epoch": completed_epoch,
        "configuration": configuration.model_dump(mode="json"),
        "configuration_identity": payload_hash(configuration),
        "trajectory_identity": identity,
        "fold": configuration.smoke_fold,
        "seed": configuration.seed + configuration.smoke_fold,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_checkpoint_sha256": best_checkpoint_sha256,
        "runner_id": runner_id,
        "data_loader_id": data_loader_id,
        "continuation_version": CONTINUATION_VERSION,
        "continuation_semantics": CONTINUATION_SEMANTICS,
        "rng_state": rng_state,
    }
    payload["checkpoint_identity"] = payload_hash(
        {
            key: payload[key]
            for key in (
                "completed_epoch",
                "configuration",
                "configuration_identity",
                "trajectory_identity",
                "fold",
                "seed",
                "best_epoch",
                "best_score",
                "best_checkpoint_sha256",
                "runner_id",
                "data_loader_id",
                "continuation_version",
                "continuation_semantics",
            )
        }
    )
    return payload


def write_continuation_manifest(
    candidate_root: Path,
    *,
    configuration: FeTAUNetSearchConfiguration,
    completed_epoch: int,
    best_epoch: int,
    best_score: float,
) -> None:
    checkpoint_root = candidate_root / "checkpoints" / "fold-0"
    last_path = checkpoint_root / "last.pt"
    best_path = checkpoint_root / "best.pt"
    atomic_json_write(
        checkpoint_root / "continuation.json",
        {
            "schema_version": CONTINUATION_VERSION,
            "continuation_semantics": CONTINUATION_SEMANTICS,
            "trajectory_identity": trajectory_identity(configuration),
            "configuration_identity": payload_hash(configuration),
            "completed_epoch": completed_epoch,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "last_checkpoint_sha256": _sha256(last_path),
            "best_checkpoint_sha256": _sha256(best_path),
        },
    )


def _candidate_manifests(namespace_root: Path) -> list[tuple[Path, dict]]:
    rows: list[tuple[Path, dict]] = []
    for path in namespace_root.glob(
        "experiment-*/checkpoints/fold-0/continuation.json"
    ):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict):
            rows.append((path.parents[2], payload))
    return rows


def find_resume_source(
    namespace_root: Path,
    current_candidate_root: Path,
    requested: FeTAUNetSearchConfiguration,
) -> Path | None:
    """Select the highest completed lower rung for the same trajectory."""

    requested_identity = trajectory_identity(requested)
    candidates: list[tuple[int, str, Path]] = []
    for root, manifest in _candidate_manifests(namespace_root):
        completed = manifest.get("completed_epoch")
        if (
            root.resolve() == current_candidate_root.resolve()
            or manifest.get("schema_version") != CONTINUATION_VERSION
            or manifest.get("trajectory_identity") != requested_identity
            or isinstance(completed, bool)
            or not isinstance(completed, int)
            or completed not in FIDELITY_LEVELS
            or completed >= requested.maximum_epochs
        ):
            continue
        candidates.append((completed, root.name, root))
    if not candidates:
        return None
    return max(candidates)[2]


def load_resume_plan(
    source_candidate_root: Path,
    requested: FeTAUNetSearchConfiguration,
    *,
    runner_id: str,
    data_loader_id: str,
    map_location: str = "cuda",
) -> ResumePlan:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("feta_ml_dependencies_unavailable") from exc

    checkpoint_root = source_candidate_root / "checkpoints" / "fold-0"
    manifest_path = checkpoint_root / "continuation.json"
    history_path = checkpoint_root / "validation-history.json"
    last_path = checkpoint_root / "last.pt"
    best_path = checkpoint_root / "best.pt"
    if not all(
        path.is_file() for path in (manifest_path, history_path, last_path, best_path)
    ):
        raise ValueError("feta_unet_resume_checkpoint_missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        history_payload = json.loads(history_path.read_text(encoding="utf-8"))
        last = torch.load(last_path, map_location=map_location, weights_only=True)
        best = torch.load(best_path, map_location=map_location, weights_only=True)
    except Exception as exc:
        raise ValueError("feta_unet_resume_checkpoint_unreadable") from exc

    required = {
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "completed_epoch",
        "configuration",
        "configuration_identity",
        "trajectory_identity",
        "fold",
        "seed",
        "best_epoch",
        "best_score",
        "best_checkpoint_sha256",
        "runner_id",
        "data_loader_id",
        "continuation_version",
        "continuation_semantics",
        "rng_state",
        "checkpoint_identity",
    }
    if not isinstance(last, dict) or not required.issubset(last):
        raise ValueError("feta_unet_resume_checkpoint_invalid")
    source = FeTAUNetSearchConfiguration.model_validate(last["configuration"])
    completed = last["completed_epoch"]
    expected_identity = trajectory_identity(requested)
    checkpoint_identity = payload_hash(
        {
            key: last[key]
            for key in (
                "completed_epoch",
                "configuration",
                "configuration_identity",
                "trajectory_identity",
                "fold",
                "seed",
                "best_epoch",
                "best_score",
                "best_checkpoint_sha256",
                "runner_id",
                "data_loader_id",
                "continuation_version",
                "continuation_semantics",
            )
        }
    )
    if (
        last["checkpoint_identity"] != checkpoint_identity
        or last["continuation_version"] != CONTINUATION_VERSION
        or last["continuation_semantics"] != CONTINUATION_SEMANTICS
        or last["runner_id"] != runner_id
        or last["data_loader_id"] != data_loader_id
        or last["configuration_identity"] != payload_hash(source)
        or last["trajectory_identity"] != expected_identity
        or trajectory_identity(source) != expected_identity
        or source.maximum_epochs != completed
        or completed not in FIDELITY_LEVELS
        or completed >= requested.maximum_epochs
        or last["fold"] != requested.smoke_fold
        or last["seed"] != requested.seed + requested.smoke_fold
        or _sha256(last_path) != manifest.get("last_checkpoint_sha256")
        or _sha256(best_path) != last["best_checkpoint_sha256"]
        or _sha256(best_path) != manifest.get("best_checkpoint_sha256")
    ):
        raise ValueError("feta_unet_resume_checkpoint_identity_mismatch")
    best_epoch = last["best_epoch"]
    best_score = last["best_score"]
    if (
        isinstance(best_epoch, bool)
        or not isinstance(best_epoch, int)
        or not 1 <= best_epoch <= completed
        or isinstance(best_score, bool)
        or not isinstance(best_score, (int, float))
        or not math.isfinite(float(best_score))
    ):
        raise ValueError("feta_unet_resume_best_checkpoint_invalid")
    entries = (
        history_payload.get("entries") if isinstance(history_payload, dict) else None
    )
    if not isinstance(entries, list) or any(
        not isinstance(item, dict) for item in entries
    ):
        raise ValueError("feta_unet_resume_history_invalid")
    return ResumePlan(
        source_candidate_root=source_candidate_root,
        completed_epoch=completed,
        start_epoch=completed + 1,
        best_epoch=best_epoch,
        best_score=float(best_score),
        trajectory_identity=expected_identity,
        last_payload=last,
        best_payload=best,
        validation_history=tuple(entries),
    )


def copy_prior_checkpoints(plan: ResumePlan, target_checkpoint_root: Path) -> None:
    source = plan.source_candidate_root / "checkpoints" / "fold-0"
    target_checkpoint_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "best.pt", target_checkpoint_root / "best.pt")
    for milestone in source.glob("milestone-epoch-*.pt"):
        shutil.copy2(milestone, target_checkpoint_root / milestone.name)
