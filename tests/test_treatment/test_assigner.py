"""Tests for TreatmentAssigner."""

import pandas as pd
import pytest

from did_panel_builder import StackedPanel, StaggeredPanel, TreatmentAssigner

EXPECTED_COLUMNS = [
    "first_event_time",
    "cnt_pre_periods",
    "cnt_post_periods",
    "early_treated",
    "insufficient_pre",
    "insufficient_post",
    "has_pre_data",
    "has_post_data",
    "treatment_category",
]

VALID_CATEGORIES = [
    "never_treated",
    "has_pre_post",
    "has_pre_only",
    "has_post_only",
    "has_neither",
]


class TestBuild:
    def test_output_columns_exist(self, simple_panel, config):
        assigner = TreatmentAssigner(simple_panel, config=config)
        result = assigner.build()
        for col in EXPECTED_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_row_count_preserved(self, simple_panel, config):
        assigner = TreatmentAssigner(simple_panel, config=config)
        result = assigner.build()
        assert len(result) == len(simple_panel)

    def test_never_treated_units(self, simple_panel, config):
        assigner = TreatmentAssigner(simple_panel, config=config)
        result = assigner.build()

        never = result[result["unit_id"].isin(["6", "7", "8", "9", "10"])]
        assert (never["first_event_time"] == config.fill_value).all()
        assert (never["cnt_pre_periods"] == 0).all()
        assert (never["cnt_post_periods"] == 0).all()
        assert (never["early_treated"] == 0).all()
        assert (never["has_pre_data"] == 0).all()
        assert (never["has_post_data"] == 0).all()
        assert (never["treatment_category"] == "never_treated").all()

    def test_pre_post_counts(self, simple_panel, config):
        """Unit '1' treated 2008, panel 2005-2014: pre=3 (2005,2006,2007), post=6."""
        assigner = TreatmentAssigner(simple_panel, config=config)
        result = assigner.build()

        u1 = result[result["unit_id"] == "1"].iloc[0]
        assert u1["first_event_time"] == 2008
        assert u1["cnt_pre_periods"] == 3
        assert u1["cnt_post_periods"] == 6

    def test_has_pre_post_flags(self, simple_panel, config):
        assigner = TreatmentAssigner(simple_panel, config=config)
        result = assigner.build()

        treated = result[result["first_event_time"] != config.fill_value]
        units = treated.drop_duplicates("unit_id")
        assert (units["has_pre_data"] == (units["cnt_pre_periods"] > 0).astype(int)).all()
        assert (units["has_post_data"] == (units["cnt_post_periods"] > 0).astype(int)).all()

    def test_category_values(self, simple_panel, config):
        assigner = TreatmentAssigner(simple_panel, config=config)
        result = assigner.build()
        assert set(result["treatment_category"].unique()).issubset(set(VALID_CATEGORIES))

    def test_treated_units_have_pre_post(self, simple_panel, config):
        """Units 1-5 have events in middle of panel → has_pre_post."""
        assigner = TreatmentAssigner(simple_panel, config=config)
        result = assigner.build()

        for uid in ["1", "2", "3", "4", "5"]:
            cat = result[result["unit_id"] == uid]["treatment_category"].iloc[0]
            assert cat == "has_pre_post", f"Unit {uid} expected has_pre_post, got {cat}"

    def test_result_property_caches(self, simple_panel, config):
        assigner = TreatmentAssigner(simple_panel, config=config)
        r1 = assigner.result
        r2 = assigner.result
        assert r1 is r2

    def test_never_treated_only_panel(self, never_treated_only, config):
        assigner = TreatmentAssigner(never_treated_only, config=config)
        result = assigner.build()
        assert (result["treatment_category"] == "never_treated").all()
        assert (result["early_treated"] == 0).all()


class TestStudyWindow:
    def test_early_treated_flag(self, config):
        """Unit treated in study_start year should be flagged."""
        rows = []
        for year in range(2005, 2015):
            rows.append({"unit_id": "1", "year": year, "has_event": year == 2005})
            rows.append({"unit_id": "2", "year": year, "has_event": year == 2010})
            rows.append({"unit_id": "3", "year": year, "has_event": False})
        df = pd.DataFrame(rows)

        assigner = TreatmentAssigner(df, config=config, study_start=2005)
        result = assigner.build()

        u1 = result[result["unit_id"] == "1"].iloc[0]
        u2 = result[result["unit_id"] == "2"].iloc[0]
        u3 = result[result["unit_id"] == "3"].iloc[0]

        assert u1["early_treated"] == 1
        assert u2["early_treated"] == 0
        assert u3["early_treated"] == 0

    def test_no_early_treated_without_study_start(self, simple_panel, config):
        assigner = TreatmentAssigner(simple_panel, config=config)
        result = assigner.build()
        assert (result["early_treated"] == 0).all()

    def test_post_count_capped_at_study_end(self, config):
        """With study_end=2010, post periods only count up to 2010."""
        rows = []
        for year in range(2005, 2015):
            rows.append({"unit_id": "1", "year": year, "has_event": year == 2008})
        df = pd.DataFrame(rows)

        # Without cap
        assigner_no_cap = TreatmentAssigner(df, config=config)
        r_no_cap = assigner_no_cap.build()
        post_no_cap = r_no_cap.iloc[0]["cnt_post_periods"]

        # With cap
        assigner_cap = TreatmentAssigner(df, config=config, study_end=2010)
        r_cap = assigner_cap.build()
        post_cap = r_cap.iloc[0]["cnt_post_periods"]

        assert post_no_cap == 6  # 2009-2014
        assert post_cap == 2  # 2009, 2010

    def test_no_cap_without_study_end(self, simple_panel, config):
        assigner = TreatmentAssigner(simple_panel, config=config)
        result = assigner.build()
        # Unit 1 treated 2008, panel 2005-2014 → post=6 (2009-2014)
        u1 = result[result["unit_id"] == "1"].iloc[0]
        assert u1["cnt_post_periods"] == 6


class TestThresholds:
    def test_insufficient_pre_flag(self, config):
        rows = []
        # Unit 1: 1 pre-period (2009), treated 2010
        for year in range(2009, 2015):
            rows.append({"unit_id": "1", "year": year, "has_event": year == 2010})
        # Unit 2: 3 pre-periods, treated 2010
        for year in range(2007, 2015):
            rows.append({"unit_id": "2", "year": year, "has_event": year == 2010})
        df = pd.DataFrame(rows)

        assigner = TreatmentAssigner(df, config=config, min_pre_periods=2)
        result = assigner.build()

        u1 = result[result["unit_id"] == "1"].iloc[0]
        u2 = result[result["unit_id"] == "2"].iloc[0]
        assert u1["insufficient_pre"] == 1
        assert u2["insufficient_pre"] == 0

    def test_insufficient_post_flag(self, config):
        rows = []
        # Unit 1: 1 post-period
        for year in range(2005, 2012):
            rows.append({"unit_id": "1", "year": year, "has_event": year == 2010})
        df = pd.DataFrame(rows)

        assigner = TreatmentAssigner(df, config=config, min_post_periods=3)
        result = assigner.build()
        u1 = result[result["unit_id"] == "1"].iloc[0]
        assert u1["insufficient_post"] == 1

    def test_insufficient_flags_zero_for_never_treated(self, never_treated_only, config):
        assigner = TreatmentAssigner(
            never_treated_only, config=config,
            min_pre_periods=5, min_post_periods=5,
        )
        result = assigner.build()
        assert (result["insufficient_pre"] == 0).all()
        assert (result["insufficient_post"] == 0).all()

    def test_zero_threshold_no_insufficient(self, simple_panel, config):
        assigner = TreatmentAssigner(simple_panel, config=config)
        result = assigner.build()
        # All treated units in simple_panel have pre AND post → never insufficient with threshold 0
        treated = result[result["first_event_time"] != config.fill_value]
        assert (treated["insufficient_pre"] == 0).all()
        assert (treated["insufficient_post"] == 0).all()


class TestFilter:
    def test_filter_drop_early_treated(self, config):
        rows = []
        for year in range(2005, 2015):
            rows.append({"unit_id": "1", "year": year, "has_event": year == 2005})
            rows.append({"unit_id": "2", "year": year, "has_event": year == 2010})
        df = pd.DataFrame(rows)

        assigner = TreatmentAssigner(df, config=config, study_start=2005)
        filtered = assigner.filter(drop_early_treated=True)
        assert "1" not in filtered["unit_id"].values
        assert "2" in filtered["unit_id"].values

    def test_filter_keep_categories(self, simple_panel, config):
        assigner = TreatmentAssigner(simple_panel, config=config)
        filtered = assigner.filter(keep_categories=["never_treated"])
        assert (filtered["treatment_category"] == "never_treated").all()

    def test_filter_combined(self, config):
        rows = []
        for year in range(2005, 2015):
            rows.append({"unit_id": "1", "year": year, "has_event": year == 2005})
            rows.append({"unit_id": "2", "year": year, "has_event": year == 2010})
            rows.append({"unit_id": "3", "year": year, "has_event": False})
        df = pd.DataFrame(rows)

        assigner = TreatmentAssigner(
            df, config=config, study_start=2005, min_pre_periods=2,
        )
        filtered = assigner.filter(
            drop_early_treated=True,
            drop_insufficient_pre=True,
            keep_categories=["never_treated", "has_pre_post"],
        )
        # Unit 1: early_treated → dropped
        # Unit 2: has_pre_post, not early, pre=5 ≥ 2 → kept
        # Unit 3: never_treated → kept
        remaining = set(filtered["unit_id"].unique())
        assert "1" not in remaining
        assert "2" in remaining
        assert "3" in remaining

    def test_filter_returns_copy(self, simple_panel, config):
        assigner = TreatmentAssigner(simple_panel, config=config)
        filtered = assigner.filter()
        filtered["new_col"] = 1
        assert "new_col" not in assigner.result.columns

    def test_filter_unit_level(self, simple_panel, config):
        """When a unit is excluded, ALL its rows are dropped."""
        assigner = TreatmentAssigner(simple_panel, config=config)
        filtered = assigner.filter(keep_categories=["never_treated"])
        treated_ids = {"1", "2", "3", "4", "5"}
        assert len(filtered[filtered["unit_id"].isin(treated_ids)]) == 0


class TestSummary:
    def test_summary_returns_dataframe(self, simple_panel, config):
        assigner = TreatmentAssigner(simple_panel, config=config)
        summary = assigner.summary()
        assert isinstance(summary, pd.DataFrame)

    def test_summary_columns(self, simple_panel, config):
        assigner = TreatmentAssigner(simple_panel, config=config)
        summary = assigner.summary()
        assert list(summary.columns) == ["treatment_category", "n_units", "pct_units"]

    def test_summary_all_categories(self, simple_panel, config):
        assigner = TreatmentAssigner(simple_panel, config=config)
        summary = assigner.summary()
        assert set(summary["treatment_category"]) == set(VALID_CATEGORIES)

    def test_summary_percentages_sum(self, simple_panel, config):
        assigner = TreatmentAssigner(simple_panel, config=config)
        summary = assigner.summary()
        assert abs(summary["pct_units"].sum() - 100.0) < 0.5  # rounding tolerance


class TestValidation:
    def test_missing_column_raises(self, config):
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="Missing required columns"):
            TreatmentAssigner(df, config=config)

    def test_invalid_study_window(self, simple_panel, config):
        with pytest.raises(ValueError, match="study_start.*must be.*study_end"):
            TreatmentAssigner(simple_panel, config=config, study_start=2020, study_end=2005)

    def test_negative_min_periods(self, simple_panel, config):
        with pytest.raises(ValueError, match="must be >= 0"):
            TreatmentAssigner(simple_panel, config=config, min_pre_periods=-1)


class TestIntegration:
    def test_feeds_staggered_panel(self, simple_panel, config):
        assigner = TreatmentAssigner(simple_panel, config=config)
        df_clean = assigner.filter(keep_categories=["never_treated", "has_pre_post"])
        panel = StaggeredPanel(df_clean, config=config)
        result = panel.build()
        assert len(result) > 0
        assert "event_time" in result.columns

    def test_feeds_stacked_panel(self, simple_panel, config):
        assigner = TreatmentAssigner(simple_panel, config=config)
        df_clean = assigner.filter(keep_categories=["never_treated", "has_pre_post"])
        panel = StackedPanel(df_clean, config=config, time_pre=2, time_post=2)
        result = panel.build()
        assert len(result) > 0
        assert "cohort" in result.columns
