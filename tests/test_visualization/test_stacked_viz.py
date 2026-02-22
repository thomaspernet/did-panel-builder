"""Tests for stacked cohort visualization."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from did_panel_builder.visualization import plot_stacked_cohort


@pytest.fixture
def stacked_panel():
    """Stacked panel with cohort, event_time, treated, and outcome columns."""
    rng = np.random.default_rng(42)
    rows = []
    for cohort in [2008, 2010, 2012]:
        for uid in range(1, 11):
            treated = 1 if uid <= 5 else 0
            for et in range(-3, 4):
                rows.append({
                    "cohort": cohort,
                    "unit_id": str(uid),
                    "event_time": et,
                    "treated": treated,
                    "outcome": rng.normal(10 + (0.5 if treated and et >= 0 else 0), 2),
                })
    return pd.DataFrame(rows)


class TestPlotStackedCohort:
    def teardown_method(self):
        plt.close("all")

    def test_returns_figure(self, stacked_panel):
        fig = plot_stacked_cohort(stacked_panel, outcome="outcome")
        assert isinstance(fig, Figure)
        assert len(fig.axes) >= 2

    def test_custom_cols(self, stacked_panel):
        df = stacked_panel.rename(columns={"cohort": "g", "treated": "tr"})
        fig = plot_stacked_cohort(df, outcome="outcome", cohort_col="g", treated_col="tr")
        assert isinstance(fig, Figure)
