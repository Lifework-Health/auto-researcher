"""Development-only FeTA U-Net diagnostic sidecar."""

from auto_researcher.tasks.feta_unet_diagnostics.comparison import (
    compare_panel_metrics,
    summarise_learning_curve,
)
from auto_researcher.tasks.feta_unet_diagnostics.panel import (
    FeTADiagnosticPanel,
    FeTADiagnosticPanelCase,
    select_diagnostic_panel,
)

__all__ = [
    "FeTADiagnosticPanel",
    "FeTADiagnosticPanelCase",
    "compare_panel_metrics",
    "select_diagnostic_panel",
    "summarise_learning_curve",
]
