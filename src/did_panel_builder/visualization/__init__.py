"""Visualization functions for DiD panel diagnostics and event studies."""

from .event_study import (
    plot_by_treatment_status,
    plot_event_time,
    plot_multi_outcome,
    plot_pre_post_comparison,
)
from .panel import (
    plot_coverage_summary,
    plot_observation_heatmap,
    plot_outcome_variation,
    plot_pre_post_coverage,
    plot_treatment_distribution,
)
from .stacked import plot_stacked_cohort

__all__ = [
    "plot_event_time",
    "plot_by_treatment_status",
    "plot_pre_post_comparison",
    "plot_multi_outcome",
    "plot_stacked_cohort",
    "plot_treatment_distribution",
    "plot_pre_post_coverage",
    "plot_outcome_variation",
    "plot_coverage_summary",
    "plot_observation_heatmap",
]
