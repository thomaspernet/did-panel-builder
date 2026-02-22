"""did-panel-builder: Build panel datasets for difference-in-differences estimation."""

from ._types import PanelConfig
from .panels import MultiEventPanel, StackedPanel, StaggeredPanel

__all__ = [
    "PanelConfig",
    "MultiEventPanel",
    "StaggeredPanel",
    "StackedPanel",
]

__version__ = "0.1.0"
