from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from auto_researcher.tasks.feta_unet_search.v9_checkpoint_import import (
    bind_v9_parent_checkpoints,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v9_parent_binding_is_verified_transactional_and_fresh(tmp_path):
    workspace = tmp_path / "workspace"
    results = tmp_path / "results"
    parents = []
    for index in range(2):
        experiment_id = f"experiment-{index}"
        checkpoint = workspace / experiment_id / "checkpoints/fold-0/best.pt"
        evaluation = results / experiment_id / "evaluation_result.json"
        checkpoint.parent.mkdir(parents=True)
        evaluation.parent.mkdir(parents=True)
        checkpoint.write_bytes(f"checkpoint-{index}".encode())
        evaluation.write_text(json.dumps({"success": True}), encoding="utf-8")
        parents.append(
            {
                "role": f"parent-{index}",
                "experiment_id": experiment_id,
                "files": {
                    "checkpoints/fold-0/best.pt": _hash(checkpoint),
                    "evaluation_result.json": _hash(evaluation),
                },
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "feta-unet-v9-parent-checkpoints-v1",
                "source_run_id": "v8",
                "development_fold": 0,
                "sealed_holdout_evaluations": 0,
                "parents": parents,
            }
        ),
        encoding="utf-8",
    )
    destination = tmp_path / "bound"

    result = bind_v9_parent_checkpoints(
        manifest_path=manifest,
        source_workspace=workspace,
        source_result_root=results,
        destination=destination,
    )

    assert result["sealed_holdout_evaluations"] == 0
    assert len(result["parents"]) == 2
    assert (destination / "experiment-0/checkpoints/fold-0/best.pt").is_file()
    with pytest.raises(ValueError, match="destination_not_fresh"):
        bind_v9_parent_checkpoints(
            manifest_path=manifest,
            source_workspace=workspace,
            source_result_root=results,
            destination=destination,
        )
