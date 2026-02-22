"""Tests for panel visualization functions."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from did_panel_builder.visualization import (
    plot_coverage_summary,
    plot_observation_heatmap,
    plot_outcome_variation,
    plot_pre_post_coverage,
    plot_treatment_distribution,
    plot_treatment_funnel,
    plot_treatment_summary,
)


class TestPlotTreatmentSummary:
    def teardown_method(self):
        plt.close("all")

    def test_returns_figure(self, simple_panel):
        fig = plot_treatment_summary(simple_panel, unit_col="unit_id", time_col="year")
        assert isinstance(fig, Figure)

    def test_two_panels(self, simple_panel):
        fig = plot_treatment_summary(simple_panel, unit_col="unit_id", time_col="year")
        assert len(fig.axes) == 2

    def test_time_range_filter(self, simple_panel):
        fig = plot_treatment_summary(
            simple_panel, unit_col="unit_id", time_col="year", time_range=(2008, 2012),
        )
        assert isinstance(fig, Figure)

    def test_never_treated(self, never_treated_only):
        fig = plot_treatment_summary(never_treated_only, unit_col="unit_id", time_col="year")
        assert isinstance(fig, Figure)

    def test_custom_event_col(self, simple_panel):
        simple_panel = simple_panel.copy()
        simple_panel["treated"] = simple_panel["has_event"]
        fig = plot_treatment_summary(
            simple_panel, unit_col="unit_id", time_col="year", event_col="treated",
        )
        assert isinstance(fig, Figure)


class TestPlotTreatmentFunnel:
    def teardown_method(self):
        plt.close("all")

    def test_returns_figure(self):
        steps = [("Raw data", 10000), ("Filtered", 5000), ("Final", 2000)]
        fig = plot_treatment_funnel(steps)
        assert isinstance(fig, Figure)

    def test_with_summary_stats(self):
        steps = [("Raw", 10000), ("Final", 2000)]
        stats = {"Unique firms": 500, "Treatment rate": 40.5, "Note": "test"}
        fig = plot_treatment_funnel(steps, summary_stats=stats)
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2

    def test_without_summary_stats(self):
        steps = [("Step A", 100), ("Step B", 50)]
        fig = plot_treatment_funnel(steps)
        assert isinstance(fig, Figure)

    def test_single_step(self):
        fig = plot_treatment_funnel([("Only step", 1000)])
        assert isinstance(fig, Figure)

    def test_many_steps(self):
        steps = [(f"Step {i}", 1000 - i * 100) for i in range(8)]
        fig = plot_treatment_funnel(steps)
        assert isinstance(fig, Figure)


@pytest.fixture
def staggered_panel():
    """Panel with first_event_time, cnt_pre/post, treatment_type columns."""
    rng = np.random.default_rng(42)
    rows = []
    for uid in range(1, 21):
        first_event = 2010 if uid <= 10 else -1000
        tt = "treated" if uid <= 10 else "never_treated"
        for year in range(2005, 2016):
            pre = max(0, first_event - 2005) if first_event > 0 else 0
            post = max(0, 2015 - first_event) if first_event > 0 else 0
            rows.append({
                "unit_id": str(uid),
                "year": year,
                "has_event": year == first_event,
                "first_event_time": first_event,
                "cnt_pre_periods": pre,
                "cnt_post_periods": post,
                "treatment_type": tt,
                "outcome_a": rng.normal(10, 2),
                "outcome_b": int(rng.random() > 0.5),
            })
    return pd.DataFrame(rows)


class TestPlotTreatmentDistribution:
    def teardown_method(self):
        plt.close("all")

    def test_returns_figure(self, staggered_panel):
        fig = plot_treatment_distribution(staggered_panel, unit_col="unit_id")
        assert isinstance(fig, Figure)

    def test_with_ax(self, staggered_panel):
        _, ax = plt.subplots()
        fig = plot_treatment_distribution(staggered_panel, unit_col="unit_id", ax=ax)
        assert isinstance(fig, Figure)


class TestPlotPrePostCoverage:
    def teardown_method(self):
        plt.close("all")

    def test_returns_figure(self, staggered_panel):
        fig = plot_pre_post_coverage(staggered_panel, unit_col="unit_id")
        assert isinstance(fig, Figure)
        assert len(fig.axes) >= 3


class TestPlotOutcomeVariation:
    def teardown_method(self):
        plt.close("all")

    def test_returns_figure(self, staggered_panel):
        fig = plot_outcome_variation(
            staggered_panel, outcome_col="outcome_b", unit_col="unit_id",
        )
        assert isinstance(fig, Figure)

    def test_without_treatment_type(self, staggered_panel):
        df = staggered_panel.drop(columns=["treatment_type"])
        fig = plot_outcome_variation(df, outcome_col="outcome_b", unit_col="unit_id")
        assert isinstance(fig, Figure)


class TestPlotCoverageSummary:
    def teardown_method(self):
        plt.close("all")

    def test_returns_figure(self, simple_panel):
        fig = plot_coverage_summary(simple_panel, unit_col="unit_id", time_col="year")
        assert isinstance(fig, Figure)
        assert len(fig.axes) >= 4


class TestPlotObservationHeatmap:
    def teardown_method(self):
        plt.close("all")

    def test_returns_figure(self, simple_panel):
        fig = plot_observation_heatmap(simple_panel, unit_col="unit_id", time_col="year")
        assert isinstance(fig, Figure)

    def test_sampling(self, simple_panel):
        fig = plot_observation_heatmap(
            simple_panel, unit_col="unit_id", time_col="year", sample_units=5,
        )
        assert isinstance(fig, Figure)
