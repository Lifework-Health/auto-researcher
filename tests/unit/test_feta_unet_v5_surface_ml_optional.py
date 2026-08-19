from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("monai")

import torch

from auto_researcher.tasks.feta_unet_direct.model import (
    ARCHITECTURE_ID,
    architecture_identity,
    create_basic_unet,
    trainable_parameter_count,
)
from auto_researcher.tasks.feta_unet_direct.trainer import (
    create_loss,
    create_optimizer,
    create_scheduler,
)
from auto_researcher.tasks.feta_unet_search.configuration import (
    FEATURE_WIDTH_PROFILES,
    FeTAUNetSearchConfiguration,
)


def test_v5_feature_norm_and_activation_profiles_build_distinct_basic_unets():
    configurations = (
        FeTAUNetSearchConfiguration(),
        FeTAUNetSearchConfiguration(
            feature_width="narrow", norm="group", activation="PReLU"
        ),
        FeTAUNetSearchConfiguration(
            feature_width="wide", norm="group", activation="ReLU"
        ),
    )
    models = tuple(create_basic_unet(configuration) for configuration in configurations)
    identities = tuple(architecture_identity(item) for item in configurations)
    parameters = tuple(trainable_parameter_count(model) for model in models)

    assert configurations[0].features == FEATURE_WIDTH_PROFILES["baseline"]
    assert configurations[1].features == FEATURE_WIDTH_PROFILES["narrow"]
    assert configurations[2].features == FEATURE_WIDTH_PROFILES["wide"]
    assert identities[0] == ARCHITECTURE_ID
    assert len(set(identities)) == 3
    assert parameters[1] < parameters[0] < parameters[2]


@pytest.mark.parametrize("variant", ["dice_ce", "dice_focal"])
def test_v5_registered_loss_variants_are_finite(variant: str):
    configuration = FeTAUNetSearchConfiguration(loss_variant=variant)
    loss = create_loss(configuration)
    logits = torch.randn(1, 8, 4, 4, 4)
    labels = torch.randint(0, 8, (1, 1, 4, 4, 4))
    assert bool(torch.isfinite(loss(logits, labels)))


@pytest.mark.parametrize("optimizer_name", ["AdamW", "Adam"])
@pytest.mark.parametrize("schedule", ["constant", "cosine", "polynomial"])
def test_v5_optimizer_and_schedule_surface_is_executable(
    optimizer_name: str, schedule: str
):
    configuration = FeTAUNetSearchConfiguration(
        optimizer=optimizer_name,
        lr_schedule=schedule,
    )
    model = torch.nn.Linear(2, 1)
    optimizer = create_optimizer(model, configuration)
    scheduler = create_scheduler(optimizer, configuration)
    for _ in range(25):
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
    if schedule == "constant":
        assert scheduler is None
        assert optimizer.param_groups[0]["lr"] == configuration.learning_rate
    else:
        assert scheduler is not None
        assert optimizer.param_groups[0]["lr"] < configuration.learning_rate

