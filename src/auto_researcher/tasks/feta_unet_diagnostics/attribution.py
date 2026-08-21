"""Optional attribution boundary; campaign and metric diagnostics do not depend on it."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AttributionBackend(Protocol):
    backend_id: str
    backend_version: str

    def attribute(
        self,
        *,
        model: Any,
        inputs: Any,
        target_label: int,
        target_region: Any,
        method: str,
        parameters: dict[str, Any],
    ) -> Any: ...


def captum_capability() -> dict[str, Any]:
    """Describe optional Captum availability without making it a core dependency."""

    try:
        import captum
    except ImportError:
        return {
            "available": False,
            "backend_id": "captum",
            "backend_version": None,
            "supported_methods": (),
        }
    return {
        "available": True,
        "backend_id": "captum",
        "backend_version": str(captum.__version__),
        "supported_methods": (
            "integrated_gradients",
            "layer_gradcam",
            "guided_gradcam",
            "occlusion",
            "infidelity",
            "sensitivity",
        ),
    }
