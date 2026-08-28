import pytest

pytest.importorskip("torch")
pytest.importorskip("monai")

from auto_researcher.tasks.feta_unet_direct.configuration import (  # noqa: E402
    FeTAUNetDirectConfiguration,
)
from auto_researcher.tasks.feta_unet_direct.model import (  # noqa: E402
    ARCHITECTURE_ID,
    TRAINABLE_PARAMETER_COUNT,
    create_basic_unet,
    trainable_parameter_count,
)


def test_frozen_basic_unet_identity_and_parameter_count():
    model = create_basic_unet(FeTAUNetDirectConfiguration())
    rendered = repr(model)
    assert ARCHITECTURE_ID == "monai-basic-unet-3d-v1"
    assert trainable_parameter_count(model) == TRAINABLE_PARAMETER_COUNT == 5_749_608
    assert "LeakyReLU(negative_slope=0.1, inplace=True)" in rendered
    assert "InstanceNorm3d(32, eps=1e-05, momentum=0.1, affine=True" in rendered
    assert "ConvTranspose3d" in rendered
