"""The only module that directly imports the optional auto_agent_v2 package."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

from auto_researcher.tasks.models import TaskNotReadyError


@dataclass(frozen=True)
class ICCABindings:
    load_cohort: Callable[..., Any]
    harness_paths_factory: Callable[..., Any]
    network_type: type
    alignment_type: type
    propagation_cache_factory: Callable[..., Any]
    evaluate: Callable[..., Any]
    stability_objective: Callable[[Any], float]
    alpha_bounds: tuple[float, float]
    k_bounds: tuple[int, int]
    package_version: str
    code_version: str


def _installed_commit(package_file: str, package_version: str) -> str:
    package_root = Path(package_file).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "-C", str(package_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return f"harness-{package_version}"


def load_installed_icca_bindings() -> ICCABindings:
    """Lazily bind to v2; importing this module alone never imports ``harness``."""
    try:
        import harness
        from harness.core.codenames import Alignment, Network
        from harness.core.paths import HarnessPaths
        from harness.data.cohort import load_cohort
        from harness.evaluator.evaluator import evaluate
        from harness.v2.propagate import PropagationCache
        from harness.v2.search import ALPHA_BOUNDS, K_BOUNDS, stability_objective
    except ImportError as exc:
        raise TaskNotReadyError(
            "iCCA NBS requires auto_agent_v2. Install it locally with "
            "`pip install -e ../auto_agent_v2`."
        ) from exc
    try:
        package_version = version("harness")
    except PackageNotFoundError:
        package_version = getattr(harness, "__version__", "unknown")
    return ICCABindings(
        load_cohort=load_cohort,
        harness_paths_factory=HarnessPaths.from_workspace,
        network_type=Network,
        alignment_type=Alignment,
        propagation_cache_factory=PropagationCache,
        evaluate=evaluate,
        stability_objective=stability_objective,
        alpha_bounds=tuple(ALPHA_BOUNDS),
        k_bounds=tuple(K_BOUNDS),
        package_version=package_version,
        code_version=_installed_commit(harness.__file__, package_version),
    )
