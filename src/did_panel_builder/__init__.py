"""did-panel-builder: Build panel datasets for difference-in-differences estimation."""

from ._types import PanelConfig
from .panels import MultiEventPanel, StackedPanel, StaggeredPanel
from .treatment import TreatmentAssigner

__all__ = [
    "PanelConfig",
    "TreatmentAssigner",
    "MultiEventPanel",
    "StaggeredPanel",
    "StackedPanel",
]

__version__ = "0.2.0"
