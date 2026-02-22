"""Tests for event study visualization functions."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from did_panel_builder.visualization import (
    plot_by_treatment_status,
    plot_event_time,
    plot_multi_outcome,
    plot_pre_post_comparison,
)


@pytest.fixture
def event_panel():
    """Staggered panel with event_time and treatment_status columns."""
    rng = np.random.default_rng(42)
    rows = []
    for uid in range(1, 21):
        first_event = 2010 if uid <= 10 else 0  # 0 = never treated
        for year in range(2005, 2016):
            if first_event > 0:
                et = year - first_event
                status = "treated" if year >= first_event else "not_yet_treated"
            else:
                et = np.nan
                status = "never_treated"
            rows.append({
                "unit_id": str(uid),
                "year": year,
                "event_time": et,
                "treatment_status": status,
                "outcome": rng.normal(10 + (0.5 if status == "treated" else 0), 2),
                "outcome_b": rng.normal(5, 1),
            })
    return pd.DataFrame(rows)


class TestPlotEventTime:
    def teardown_method(self):
        plt.close("all")

    def test_returns_figure(self, event_panel):
        fig = plot_event_time(event_panel, outcome="outcome")
        assert isinstance(fig, Figure)

    def test_custom_window(self, event_panel):
        fig = plot_event_time(event_panel, outcome="outcome", event_window=(-3, 3))
        assert isinstance(fig, Figure)

    def test_custom_ci(self, event_panel):
        fig = plot_event_time(event_panel, outcome="outcome", ci=0.90)
        assert isinstance(fig, Figure)

    def test_with_ax(self, event_panel):
        _, ax = plt.subplots()
        fig = plot_event_time(event_panel, outcome="outcome", ax=ax)
        assert isinstance(fig, Figure)


class TestPlotByTreatmentStatus:
    def teardown_method(self):
        plt.close("all")

    def test_returns_figure(self, event_panel):
        fig = plot_by_treatment_status(event_panel, outcome="outcome")
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 3

    def test_custom_window(self, event_panel):
        fig = plot_by_treatment_status(
            event_panel, outcome="outcome", event_window=(-3, 3),
        )
        assert isinstance(fig, Figure)

    def test_empty_data(self, event_panel):
        # No matching treatment status
        df = event_panel.copy()
        df["treatment_status"] = "something_else"
        fig = plot_by_treatment_status(df, outcome="outcome")
        assert isinstance(fig, Figure)


class TestPlotPrePostComparison:
    def teardown_method(self):
        plt.close("all")

    def test_returns_figure(self, event_panel):
        fig = plot_pre_post_comparison(
            event_panel, outcomes=["outcome", "outcome_b"],
        )
        assert isinstance(fig, Figure)

    def test_single_outcome(self, event_panel):
        fig = plot_pre_post_comparison(event_panel, outcomes=["outcome"])
        assert isinstance(fig, Figure)


class TestPlotMultiOutcome:
    def teardown_method(self):
        plt.close("all")

    def test_returns_figure(self, event_panel):
        fig = plot_multi_outcome(
            event_panel, outcomes=["outcome", "outcome_b"],
        )
        assert isinstance(fig, Figure)

    def test_invalid_outcomes(self, event_panel):
        fig = plot_multi_outcome(event_panel, outcomes=["nonexistent"])
        assert isinstance(fig, Figure)

    def test_with_labels(self, event_panel):
        fig = plot_multi_outcome(
            event_panel, outcomes=["outcome", "outcome_b"],
            outcome_labels={"outcome": "Revenue", "outcome_b": "Profit"},
        )
        assert isinstance(fig, Figure)
