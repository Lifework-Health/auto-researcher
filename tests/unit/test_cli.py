from __future__ import annotations

from pathlib import Path

import pytest

from auto_researcher.cli import _load_task_configuration


def test_task_configuration_identity_requires_matching_id_and_version(tmp_path):
    configuration = tmp_path / "task.yaml"
    configuration.write_text(
        """
task:
  id: synthetic
  version: "2.0"
experiment:
  model_family: tree
  complexity: 4
  learning_rate: 0.05
runtime: {}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="version"):
        _load_task_configuration(configuration, "synthetic", "1.0")


def test_example_task_configurations_have_expected_sections():
    repository = Path(__file__).resolve().parents[2]
    for task_id in ("synthetic", "icca_nbs"):
        experiment, runtime = _load_task_configuration(
            repository / "examples" / "tasks" / task_id / "task.yaml",
            task_id,
            "1.0",
        )
        assert experiment
        assert "output_dir" in runtime
