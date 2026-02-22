"""Tests for StackedPanel."""

import pandas as pd
import pytest

from did_panel_builder import StackedPanel


class TestStackedPanelBuild:
    def test_output_columns_exist(self, simple_panel, config):
        panel = StackedPanel(simple_panel, config=config, time_pre=2, time_post=2)
        df = panel.build()

        expected = [
            "cohort",
            "event_time",
            "treated",
            "post",
            "cohort_status",
            "control_type",
            "unit_cohort_id",
            "years_in_panel",
            "nb_years_in_panel",
        ]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"

    def test_stacked_has_more_rows(self, simple_panel, config):
        """Stacking duplicates observations across cohorts."""
        panel = StackedPanel(simple_panel, config=config, time_pre=2, time_post=2)
        df = panel.build()
        assert len(df) >= len(simple_panel)

    def test_event_time_balanced(self, simple_panel, config):
        panel = StackedPanel(simple_panel, config=config, time_pre=2, time_post=2)
        df = panel.build()

        # Event time should be within [-time_pre, time_post]
        assert df["event_time"].min() >= -2
        assert df["event_time"].max() <= 2

    def test_treated_indicator_binary(self, simple_panel, config):
        panel = StackedPanel(simple_panel, config=config, time_pre=2, time_post=2)
        df = panel.build()
        assert set(df["treated"].unique()) <= {0, 1}

    def test_post_indicator_correct(self, simple_panel, config):
        panel = StackedPanel(simple_panel, config=config, time_pre=2, time_post=2)
        df = panel.build()

        post_rows = df[df["post"] == 1]
        assert (post_rows["event_time"] >= 0).all()

        pre_rows = df[df["post"] == 0]
        assert (pre_rows["event_time"] < 0).all()

    def test_unit_cohort_id_format(self, simple_panel, config):
        panel = StackedPanel(simple_panel, config=config, time_pre=2, time_post=2)
        df = panel.build()

        # unit_cohort_id = unit_id + "_" + cohort
        row = df.iloc[0]
        expected = f"{row['unit_id']}_{row['cohort']}"
        assert row["unit_cohort_id"] == expected

    def test_control_types(self, simple_panel, config):
        panel = StackedPanel(simple_panel, config=config, time_pre=2, time_post=2)
        df = panel.build()

        valid_types = {"treated", "never_treated", "not_yet_treated", "already_treated"}
        assert set(df["control_type"].unique()) <= valid_types

    def test_explicit_cohort_times(self, simple_panel, config):
        panel = StackedPanel(
            simple_panel,
            config=config,
            time_pre=2,
            time_post=2,
            cohort_times=[2010],
        )
        df = panel.build()
        assert df["cohort"].unique().tolist() == [2010]

    def test_auto_derived_cohort_times(self, simple_panel, config):
        panel = StackedPanel(simple_panel, config=config, time_pre=2, time_post=2)
        # Should auto-derive cohort times from first-event times that fit the window
        assert len(panel.cohort_times) > 0

    def test_no_valid_cohorts_raises(self, config):
        # Panel too short for the event window
        df = pd.DataFrame({
            "unit_id": ["1", "1", "2", "2"],
            "year": [2000, 2001, 2000, 2001],
            "has_event": [False, True, False, False],
        })
        panel = StackedPanel(df, config=config, time_pre=3, time_post=3)
        with pytest.raises(ValueError, match="No valid cohorts"):
            panel.build()


class TestStackedCohortSummary:
    def test_summary_returns_dataframe(self, simple_panel, config):
        panel = StackedPanel(simple_panel, config=config, time_pre=2, time_post=2)
        panel.build()
        s = panel.cohort_summary()
        assert isinstance(s, pd.DataFrame)
        assert "n_firms" in s.columns
        assert "n_treated_units" in s.columns


class TestStackedMerge:
    def test_merge_duplicates_outcomes(self, simple_panel, outcomes_df, config):
        panel = StackedPanel(simple_panel, config=config, time_pre=2, time_post=2)
        merged = panel.merge_outcomes(outcomes_df)
        # Stacked panel has more rows due to duplication across cohorts
        assert "revenue" in merged.columns
        assert len(merged) > 0


class TestStackedNeverTreatedOnly:
    def test_never_treated_raises(self, never_treated_only, config):
        panel = StackedPanel(never_treated_only, config=config, time_pre=1, time_post=1)
        with pytest.raises(ValueError, match="No valid cohorts"):
            panel.build()
