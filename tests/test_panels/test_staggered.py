"""Tests for StaggeredPanel."""

import pandas as pd
import pytest

from did_panel_builder import PanelConfig, StaggeredPanel


class TestStaggeredPanelBuild:
    def test_output_columns_exist(self, simple_panel, config):
        panel = StaggeredPanel(simple_panel, config=config)
        df = panel.build()

        expected = [
            "first_event_time",
            "event_time",
            "treatment_type",
            "treatment_status",
            "cnt_pre_periods",
            "cnt_post_periods",
            "years_in_panel",
            "nb_years_in_panel",
        ]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"

    def test_row_count_preserved(self, simple_panel, config):
        panel = StaggeredPanel(simple_panel, config=config)
        df = panel.build()
        assert len(df) == len(simple_panel)

    def test_never_treated_classification(self, simple_panel, config):
        panel = StaggeredPanel(simple_panel, config=config)
        df = panel.build()

        never = df[df["unit_id"].isin(["6", "7", "8", "9", "10"])]
        assert (never["treatment_type"] == "never_treated").all()
        assert (never["treatment_status"] == "never_treated").all()
        assert never["event_time"].isna().all()
        assert (never["first_event_time"] == config.fill_value).all()

    def test_treated_classification(self, simple_panel, config):
        panel = StaggeredPanel(simple_panel, config=config)
        df = panel.build()

        # Unit 1: treated at 2008
        u1 = df[df["unit_id"] == "1"]
        assert (u1["first_event_time"] == 2008).all()

        # Before 2008: not_yet_treated
        pre = u1[u1["year"] < 2008]
        assert (pre["treatment_status"] == "not_yet_treated").all()

        # 2008 onwards: treated
        post = u1[u1["year"] >= 2008]
        assert (post["treatment_status"] == "treated").all()

    def test_event_time_arithmetic(self, simple_panel, config):
        panel = StaggeredPanel(simple_panel, config=config)
        df = panel.build()

        u2 = df[df["unit_id"] == "2"]  # treated at 2010
        row_2008 = u2[u2["year"] == 2008].iloc[0]
        assert row_2008["event_time"] == -2

        row_2010 = u2[u2["year"] == 2010].iloc[0]
        assert row_2010["event_time"] == 0

        row_2014 = u2[u2["year"] == 2014].iloc[0]
        assert row_2014["event_time"] == 4

    def test_pre_post_counts(self, simple_panel, config):
        panel = StaggeredPanel(simple_panel, config=config)
        df = panel.build()

        # Unit 1: treated 2008, panel 2005-2014 => 3 pre, 6 post
        u1 = df[df["unit_id"] == "1"].iloc[0]
        assert u1["cnt_pre_periods"] == 3
        assert u1["cnt_post_periods"] == 6

    def test_never_treated_only_panel(self, never_treated_only, config):
        panel = StaggeredPanel(never_treated_only, config=config)
        df = panel.build()
        assert (df["treatment_type"] == "never_treated").all()
        assert (df["first_event_time"] == config.fill_value).all()

    def test_panel_property_caches(self, simple_panel, config):
        panel = StaggeredPanel(simple_panel, config=config)
        df1 = panel.panel
        df2 = panel.panel
        assert df1 is df2


class TestStaggeredFilterSample:
    def test_filter_keeps_treated_and_never(self, simple_panel, config):
        panel = StaggeredPanel(simple_panel, config=config)
        df = panel.build()
        filtered = panel.filter_sample(df)

        types = filtered["treatment_type"].unique()
        assert set(types) <= {"treated", "never_treated"}

    def test_min_pre_filter(self, simple_panel, config):
        panel = StaggeredPanel(simple_panel, config=config)
        df = panel.build()
        filtered = panel.filter_sample(df, min_pre_periods=4)

        # Only unit 2 (treated 2010, 5 pre) and unit 3 (treated 2012, 7 pre) + never-treated
        ever_treated = filtered[filtered["treatment_type"] != "never_treated"]
        assert set(ever_treated["unit_id"].unique()) <= {"2", "3", "5"}

    def test_filter_preserves_never_treated(self, simple_panel, config):
        panel = StaggeredPanel(simple_panel, config=config)
        df = panel.build()
        filtered = panel.filter_sample(df, min_pre_periods=100)

        # Even with extreme filter, never-treated are kept
        never = filtered[filtered["treatment_type"] == "never_treated"]
        assert len(never) > 0

    def test_filter_without_df_uses_cache(self, simple_panel, config):
        panel = StaggeredPanel(simple_panel, config=config)
        panel.build()
        filtered = panel.filter_sample()  # No df argument
        assert len(filtered) > 0

    def test_min_event_time_trims_early_rows(self, simple_panel, config):
        panel = StaggeredPanel(simple_panel, config=config)
        df = panel.build()
        filtered = panel.filter_sample(df, min_event_time=-3)

        treated_rows = filtered[filtered["event_time"].notna()]
        assert (treated_rows["event_time"] >= -3).all()

    def test_max_event_time_trims_late_rows(self, simple_panel, config):
        panel = StaggeredPanel(simple_panel, config=config)
        df = panel.build()
        filtered = panel.filter_sample(df, max_event_time=5)

        treated_rows = filtered[filtered["event_time"].notna()]
        assert (treated_rows["event_time"] <= 5).all()

    def test_event_window_preserves_never_treated(self, simple_panel, config):
        panel = StaggeredPanel(simple_panel, config=config)
        df = panel.build()
        n_never_before = len(df[df["treatment_type"] == "never_treated"])

        filtered = panel.filter_sample(df, min_event_time=-2, max_event_time=2)

        n_never_after = len(filtered[filtered["treatment_type"] == "never_treated"])
        assert n_never_after == n_never_before

    def test_event_window_reduces_rows(self, simple_panel, config):
        panel = StaggeredPanel(simple_panel, config=config)
        df = panel.build()
        filtered = panel.filter_sample(df, min_event_time=-1, max_event_time=1)

        assert len(filtered) < len(df)

    def test_combined_filters(self, simple_panel, config):
        """Event window + min coverage + treatment types — mirrors R prepare_sunab."""
        panel = StaggeredPanel(simple_panel, config=config)
        df = panel.build()
        filtered = panel.filter_sample(
            df,
            keep_treatment_types=["treated", "never_treated"],
            min_event_time=-3,
            max_event_time=5,
            min_pre_periods=1,
            min_post_periods=2,
        )

        # Only treated and never_treated
        assert set(filtered["treatment_type"].unique()) <= {"treated", "never_treated"}
        # Event window respected
        treated_rows = filtered[filtered["event_time"].notna()]
        assert (treated_rows["event_time"] >= -3).all()
        assert (treated_rows["event_time"] <= 5).all()


class TestStaggeredVariationFlag:
    def test_adds_flag_column(self, simple_panel, config):
        panel = StaggeredPanel(simple_panel, config=config)
        df = panel.build()
        df_flagged = panel.add_variation_flag(df, "outcome_b")
        assert "has_variation_outcome_b" in df_flagged.columns

    def test_flag_values_binary(self, simple_panel, config):
        panel = StaggeredPanel(simple_panel, config=config)
        df = panel.build()
        df_flagged = panel.add_variation_flag(df, "outcome_b")
        assert set(df_flagged["has_variation_outcome_b"].unique()) <= {0, 1}


class TestStaggeredMergeOutcomes:
    def test_merge_preserves_rows(self, simple_panel, outcomes_df, config):
        panel = StaggeredPanel(simple_panel, config=config)
        merged = panel.merge_outcomes(outcomes_df)
        assert len(merged) == len(simple_panel)
        assert "revenue" in merged.columns
        assert "profit" in merged.columns

    def test_merge_selected_columns(self, simple_panel, outcomes_df, config):
        panel = StaggeredPanel(simple_panel, config=config)
        merged = panel.merge_outcomes(outcomes_df, outcome_cols=["revenue"])
        assert "revenue" in merged.columns
        assert "profit" not in merged.columns


class TestStaggeredSummary:
    def test_summary_returns_dataframe(self, simple_panel, config):
        panel = StaggeredPanel(simple_panel, config=config)
        s = panel.summary()
        assert isinstance(s, pd.DataFrame)
        assert "n_obs" in s.columns
        assert "n_units" in s.columns


class TestStaggeredValidation:
    def test_missing_column_raises(self, config):
        df = pd.DataFrame({"unit_id": [1], "year": [2000]})
        with pytest.raises(ValueError, match="Missing required columns"):
            StaggeredPanel(df, config=config)

    def test_custom_config(self):
        df = pd.DataFrame({
            "county": ["A", "A", "B", "B"],
            "yr": [2000, 2001, 2000, 2001],
            "shock": [False, True, False, False],
        })
        config = PanelConfig(unit_col="county", time_col="yr", event_col="shock")
        panel = StaggeredPanel(df, config=config)
        result = panel.build()
        assert "first_event_time" in result.columns
