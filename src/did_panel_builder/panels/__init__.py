"""Panel builders for different DiD estimation strategies."""

from .multi_event import MultiEventPanel
from .stacked import StackedPanel
from .staggered import StaggeredPanel

__all__ = ["MultiEventPanel", "StaggeredPanel", "StackedPanel"]
