"""Search-specific FeTA training primitives."""

from __future__ import annotations

import random
from typing import Any

from auto_researcher.tasks.feta_seg.model import create_segresnet
from auto_researcher.tasks.feta_seg_search.configuration import (
    FeTASegSearchConfiguration,
)

ARCHITECTURE_VERSION = "monai-segresnet-3d-32-1224-111-search-dropout-v1"
LOSS_VERSION = "dice-ce-softmax-onehot-no-background-search-diceweight-v1"
OPTIMISER_VERSION = "adamw-search-lr-wd-v1"
INFERENCE_VERSION = "sliding-window-128-overlap0.5-gaussian-native-restore-v2"


def create_model(configuration: FeTASegSearchConfiguration):
    # SegResNet construction is attribute-based; this preserves the audited
    # baseline architecture while allowing only its registered dropout axis.
    return create_segresnet(configuration)  # type: ignore[arg-type]


def create_loss(configuration: FeTASegSearchConfiguration):
    try:
        from monai.losses import DiceCELoss
    except ImportError as exc:
        raise RuntimeError("feta_ml_dependencies_unavailable") from exc
    return DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        include_background=False,
        lambda_dice=configuration.dice_weight,
        lambda_ce=configuration.ce_weight,
    )


def create_optimizer(model: Any, configuration: FeTASegSearchConfiguration):
    try:
        from torch.optim import AdamW
    except ImportError as exc:
        raise RuntimeError("feta_ml_dependencies_unavailable") from exc
    return AdamW(
        model.parameters(),
        lr=configuration.learning_rate,
        weight_decay=configuration.weight_decay,
    )


def seed_everything(configuration: FeTASegSearchConfiguration) -> int:
    seed = configuration.seed + configuration.fold
    random.seed(seed)
    try:
        import numpy as np
        import torch
        from monai.utils import set_determinism
    except ImportError as exc:
        raise RuntimeError("feta_ml_dependencies_unavailable") from exc
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)
    return seed


def require_search_environment() -> dict[str, Any]:
    try:
        import monai
        import nibabel
        import numpy
        import scipy  # type: ignore[import-untyped]
        import torch
    except ImportError as exc:
        raise RuntimeError("feta_ml_dependencies_unavailable") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("feta_search_cuda_unavailable")
    return {
        "torch": torch.__version__,
        "monai": monai.__version__,
        "nibabel": nibabel.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "cuda": torch.version.cuda,
        "cudnn": str(torch.backends.cudnn.version()),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "amp": "cuda-float16",
        "deterministic_seed_base": 20260807,
    }


def sliding_window_predict(inputs: Any, model: Any, configuration: FeTASegSearchConfiguration):
    try:
        from monai.inferers import sliding_window_inference
    except ImportError as exc:
        raise RuntimeError("feta_ml_dependencies_unavailable") from exc
    return sliding_window_inference(
        inputs,
        roi_size=configuration.patch_size,
        sw_batch_size=configuration.sliding_window_batch_size,
        predictor=model,
        overlap=configuration.inference_overlap,
        mode=configuration.inference_blending,
    )
