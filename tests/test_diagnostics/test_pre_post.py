"""Tests for PrePostDiagnostics."""

import numpy as np
import pandas as pd
import pytest

from did_panel_builder.diagnostics import PrePostDiagnostics


class TestPrePostAnalyze:
    def test_analyze_returns_all_keys(self, staggered_panel, config):
        diag = PrePostDiagnostics(config=config)
        results = diag.analyze(
            staggered_panel,
            outcomes=["outcome_a", "outcome_b"],
            treatment_col="treatment_type",
            event_time_col="event_time",
        )
        assert "pre_post_means" in results
        assert "within_variation" in results
        assert "selection_gap" in results

    def test_raises_on_missing_outcomes(self, staggered_panel, config):
        diag = PrePostDiagnostics(config=config)
        with pytest.raises(ValueError, match="None of the outcomes"):
            diag.analyze(
                staggered_panel,
                outcomes=["nonexistent_col"],
                treatment_col="treatment_type",
                event_time_col="event_time",
            )

    def test_skips_missing_columns(self, staggered_panel, config):
        diag = PrePostDiagnostics(config=config)
        results = diag.analyze(
            staggered_panel,
            outcomes=["outcome_a", "nonexistent_col"],
            treatment_col="treatment_type",
            event_time_col="event_time",
        )
        assert len(results["pre_post_means"]) == 1
        assert results["pre_post_means"].iloc[0]["outcome"] == "outcome_a"

    def test_warns_on_missing_columns(self, staggered_panel, config):
        diag = PrePostDiagnostics(config=config)
        with pytest.warns(UserWarning, match="nonexistent_col"):
            diag.analyze(
                staggered_panel,
                outcomes=["outcome_a", "nonexistent_col"],
                treatment_col="treatment_type",
                event_time_col="event_time",
            )


class TestPrePostMeans:
    def test_columns(self, staggered_panel, config):
        diag = PrePostDiagnostics(config=config)
        results = diag.analyze(
            staggered_panel,
            outcomes=["outcome_a"],
            treatment_col="treatment_type",
            event_time_col="event_time",
        )
        df = results["pre_post_means"]
        expected = ["outcome", "pre_mean", "post_mean", "diff", "pre_obs", "post_obs"]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"

    def test_diff_is_post_minus_pre(self, staggered_panel, config):
        diag = PrePostDiagnostics(config=config)
        results = diag.analyze(
            staggered_panel,
            outcomes=["outcome_a"],
            treatment_col="treatment_type",
            event_time_col="event_time",
        )
        row = results["pre_post_means"].iloc[0]
        assert abs(row["diff"] - (row["post_mean"] - row["pre_mean"])) < 1e-10

    def test_obs_counts_positive(self, staggered_panel, config):
        diag = PrePostDiagnostics(config=config)
        results = diag.analyze(
            staggered_panel,
            outcomes=["outcome_a"],
            treatment_col="treatment_type",
            event_time_col="event_time",
        )
        row = results["pre_post_means"].iloc[0]
        assert row["pre_obs"] > 0
        assert row["post_obs"] > 0


class TestEventWindow:
    """Restricting pre/post to an event window matches the event study sample.

    Regression for issue #1: PrePostDiagnostics was pooling every treated-unit
    row, while event study plots filter to ``event_window``. On unbalanced
    panels with a trending outcome, the two views could disagree on the sign
    of the post-pre diff because distant event times outside the plot window
    dominated the PrePost means.
    """

    @staticmethod
    def _windowed_panel():
        """Treated unit 1 event in 2010, with outcome rising near the event
        and falling far post. Never-treated unit 2 as control."""
        from did_panel_builder import PanelConfig, StaggeredPanel

        rows = []
        for year in range(2000, 2021):
            if year < 2010:
                outcome = 0.0
            elif year <= 2012:
                outcome = 5.0
            else:
                outcome = -5.0
            rows.append({
                "unit_id": "1",
                "year": year,
                "has_event": year == 2010,
                "outcome_a": outcome,
            })
            rows.append({
                "unit_id": "2",
                "year": year,
                "has_event": False,
                "outcome_a": 0.0,
            })

        cfg = PanelConfig(unit_col="unit_id", time_col="year", event_col="has_event")
        panel = StaggeredPanel(pd.DataFrame(rows), config=cfg).build()
        return panel, cfg

    def test_full_panel_pools_distant_event_times(self):
        """Without event_window, distant negative post values flip the sign."""
        panel, cfg = self._windowed_panel()
        diag = PrePostDiagnostics(config=cfg)
        results = diag.analyze(panel, outcomes=["outcome_a"])
        row = results["pre_post_means"].iloc[0]
        assert row["diff"] < 0, (
            "Expected negative pooled diff because post years 2013-2020 "
            f"dominate, got diff={row['diff']}"
        )

    def test_event_window_matches_local_effect(self):
        """With event_window=(-2, 2), the post mean reflects the near-event rise."""
        panel, cfg = self._windowed_panel()
        diag = PrePostDiagnostics(config=cfg)
        results = diag.analyze(
            panel, outcomes=["outcome_a"], event_window=(-2, 2),
        )
        row = results["pre_post_means"].iloc[0]
        assert row["pre_mean"] == pytest.approx(0.0)
        assert row["post_mean"] == pytest.approx(5.0)
        assert row["diff"] == pytest.approx(5.0)
        assert row["pre_obs"] == 2
        assert row["post_obs"] == 3

    def test_event_window_filters_selection_gap(self):
        """Selection gap restricts post_treated to the event window."""
        panel, cfg = self._windowed_panel()
        diag = PrePostDiagnostics(config=cfg)
        results = diag.analyze(
            panel, outcomes=["outcome_a"], event_window=(-2, 2),
        )
        gap = results["selection_gap"]
        assert gap["n_treated"] == 3
        assert gap["rate_treated"] == pytest.approx(5.0)


class TestWithinVariation:
    def test_columns(self, staggered_panel, config):
        diag = PrePostDiagnostics(config=config)
        results = diag.analyze(
            staggered_panel,
            outcomes=["outcome_b"],
            treatment_col="treatment_type",
            event_time_col="event_time",
        )
        df = results["within_variation"]
        expected = ["outcome", "n_units", "always_0", "always_1", "varies", "pct_varies"]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"

    def test_counts_sum(self, staggered_panel, config):
        """always_0 + always_1 + varies == n_units."""
        diag = PrePostDiagnostics(config=config)
        results = diag.analyze(
            staggered_panel,
            outcomes=["outcome_b"],
            treatment_col="treatment_type",
            event_time_col="event_time",
        )
        row = results["within_variation"].iloc[0]
        assert row["always_0"] + row["always_1"] + row["varies"] == row["n_units"]

    def test_pct_varies_in_range(self, staggered_panel, config):
        diag = PrePostDiagnostics(config=config)
        results = diag.analyze(
            staggered_panel,
            outcomes=["outcome_b"],
            treatment_col="treatment_type",
            event_time_col="event_time",
        )
        row = results["within_variation"].iloc[0]
        assert 0.0 <= row["pct_varies"] <= 1.0

    def test_binary_outcome_has_some_variation(self, staggered_panel, config):
        """outcome_b is random binary, so some units should vary."""
        diag = PrePostDiagnostics(config=config)
        results = diag.analyze(
            staggered_panel,
            outcomes=["outcome_b"],
            treatment_col="treatment_type",
            event_time_col="event_time",
        )
        row = results["within_variation"].iloc[0]
        assert row["varies"] > 0


class TestSelectionGap:
    def test_keys(self, staggered_panel, config):
        diag = PrePostDiagnostics(config=config)
        results = diag.analyze(
            staggered_panel,
            outcomes=["outcome_b"],
            treatment_col="treatment_type",
            event_time_col="event_time",
        )
        gap = results["selection_gap"]
        assert "rate_treated" in gap
        assert "rate_control" in gap
        assert "excess_rate" in gap
        assert "n_treated" in gap
        assert "n_control" in gap
        assert "outcome" in gap

    def test_excess_is_diff(self, staggered_panel, config):
        diag = PrePostDiagnostics(config=config)
        results = diag.analyze(
            staggered_panel,
            outcomes=["outcome_b"],
            treatment_col="treatment_type",
            event_time_col="event_time",
        )
        gap = results["selection_gap"]
        expected = gap["rate_treated"] - gap["rate_control"]
        assert abs(gap["excess_rate"] - expected) < 1e-10

    def test_custom_selection_outcome(self, staggered_panel, config):
        diag = PrePostDiagnostics(config=config)
        results = diag.analyze(
            staggered_panel,
            outcomes=["outcome_a", "outcome_b"],
            treatment_col="treatment_type",
            event_time_col="event_time",
            selection_outcome="outcome_b",
        )
        assert results["selection_gap"]["outcome"] == "outcome_b"


class TestPrintSummary:
    def test_runs_without_error(self, staggered_panel, config, capsys):
        diag = PrePostDiagnostics(config=config)
        results = diag.analyze(
            staggered_panel,
            outcomes=["outcome_a", "outcome_b"],
            treatment_col="treatment_type",
            event_time_col="event_time",
        )
        diag.print_summary(results)
        captured = capsys.readouterr()
        assert "Pre/post means" in captured.out
        assert "Within-unit variation" in captured.out
        assert "Selection gap" in captured.out


class TestPlotSummary:
    def test_returns_figure(self, staggered_panel, config):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        diag = PrePostDiagnostics(config=config)
        results = diag.analyze(
            staggered_panel,
            outcomes=["outcome_a", "outcome_b"],
            treatment_col="treatment_type",
            event_time_col="event_time",
        )
        fig = diag.plot_summary(results)
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 3
        plt.close(fig)
