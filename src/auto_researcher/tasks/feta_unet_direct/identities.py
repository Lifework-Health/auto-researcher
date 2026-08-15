"""Shared immutable runner identities for the frozen BasicUNet task."""

from typing import Literal

ENGINEERING_SMOKE_RUNNER_ID = "feta-basic-unet-engineering-smoke-runner-v1"
BASELINE_RUNNER_ID = "feta-basic-unet-five-fold-oof-runner-v1"
DATA_LOADER_ID = "monai-persistent-train-spawn4-uncached-validation-v3"


def runner_id(profile: Literal["engineering_smoke", "frozen_baseline"]) -> str:
    return (
        ENGINEERING_SMOKE_RUNNER_ID
        if profile == "engineering_smoke"
        else BASELINE_RUNNER_ID
    )
