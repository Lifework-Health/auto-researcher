"""Shared immutable runner identities for the frozen BasicUNet task."""

from typing import Literal

ENGINEERING_SMOKE_RUNNER_ID = "feta-basic-unet-engineering-smoke-runner-v1"
DEVELOPMENT_BASELINE_RUNNER_ID = "feta-basic-unet-fold0-25epoch-development-runner-v1"
BASELINE_RUNNER_ID = "feta-basic-unet-five-fold-oof-runner-v1"
DATA_LOADER_ID = "monai-persistent-train-spawn4-uncached-validation-v3"


def runner_id(
    profile: Literal["engineering_smoke", "development_baseline", "frozen_baseline"],
) -> str:
    if profile == "engineering_smoke":
        return ENGINEERING_SMOKE_RUNNER_ID
    if profile == "development_baseline":
        return DEVELOPMENT_BASELINE_RUNNER_ID
    return BASELINE_RUNNER_ID
