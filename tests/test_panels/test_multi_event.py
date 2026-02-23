"""Tests for MultiEventPanel."""

import pandas as pd

from did_panel_builder import MultiEventPanel


class TestMultiEventPanelBuild:
    def test_output_columns_exist(self, simple_panel, config):
        panel = MultiEventPanel(simple_panel, config=config)
        df = panel.build()

        expected = [
            "first_event_time",
            "years_treated",
            "event_time",
            "treatment_timing",
            "D",
            "treatment_type",
            "cnt_pre_periods",
            "cnt_post_periods",
            "years_in_panel",
            "nb_years_in_panel",
        ]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"

    def test_row_count_preserved(self, simple_panel, config):
        panel = MultiEventPanel(simple_panel, config=config)
        df = panel.build()
        assert len(df) == len(simple_panel)

    def test_multiple_events_tracked(self, simple_panel, config):
        panel = MultiEventPanel(simple_panel, config=config)
        df = panel.build()

        # Unit 4: events at 2008 and 2011
        u4 = df[df["unit_id"] == "4"].iloc[0]
        assert u4["years_treated"] == [2008, 2011]
        assert u4["first_event_time"] == 2008

    def test_single_event_tracked(self, simple_panel, config):
        panel = MultiEventPanel(simple_panel, config=config)
        df = panel.build()

        u1 = df[df["unit_id"] == "1"].iloc[0]
        assert u1["years_treated"] == [2008]
        assert u1["first_event_time"] == 2008

    def test_never_treated_empty_list(self, simple_panel, config):
        panel = MultiEventPanel(simple_panel, config=config)
        df = panel.build()

        u6 = df[df["unit_id"] == "6"].iloc[0]
        assert u6["years_treated"] == []
        assert u6["first_event_time"] == config.fill_value

    def test_treatment_timing_at_event_only(self, simple_panel, config):
        panel = MultiEventPanel(simple_panel, config=config)
        df = panel.build()

        # Unit 1: treated at 2008
        u1 = df[df["unit_id"] == "1"]

        # At event year: treatment_timing == first_event_time
        at_event = u1[u1["year"] == 2008].iloc[0]
        assert at_event["treatment_timing"] == 2008

        # Before event: treatment_timing == fill_value
        before = u1[u1["year"] == 2007].iloc[0]
        assert before["treatment_timing"] == config.fill_value

        # After event: treatment_timing == fill_value
        after = u1[u1["year"] == 2009].iloc[0]
        assert after["treatment_timing"] == config.fill_value

    def test_event_time_arithmetic(self, simple_panel, config):
        panel = MultiEventPanel(simple_panel, config=config)
        df = panel.build()

        u1 = df[df["unit_id"] == "1"]
        row_2005 = u1[u1["year"] == 2005].iloc[0]
        assert row_2005["event_time"] == -3  # 2005 - 2008

        row_2008 = u1[u1["year"] == 2008].iloc[0]
        assert row_2008["event_time"] == 0

    def test_D_treated_unit(self, simple_panel, config):
        """D=1 for treated unit at and after first event, D=0 before."""
        panel = MultiEventPanel(simple_panel, config=config)
        df = panel.build()

        # Unit 1: treated at 2008
        u1 = df[df["unit_id"] == "1"]
        pre = u1[u1["year"] < 2008]
        post = u1[u1["year"] >= 2008]
        assert (pre["D"] == 0).all()
        assert (post["D"] == 1).all()

    def test_D_never_treated(self, simple_panel, config):
        """D=0 for all periods of never-treated units."""
        panel = MultiEventPanel(simple_panel, config=config)
        df = panel.build()

        u6 = df[df["unit_id"] == "6"]
        assert (u6["D"] == 0).all()

    def test_D_multi_event_unit(self, simple_panel, config):
        """D=1 from first event onward, even with multiple events."""
        panel = MultiEventPanel(simple_panel, config=config)
        df = panel.build()

        # Unit 4: events at 2008 and 2011 → D=1 from 2008 onward
        u4 = df[df["unit_id"] == "4"]
        pre = u4[u4["year"] < 2008]
        post = u4[u4["year"] >= 2008]
        assert (pre["D"] == 0).all()
        assert (post["D"] == 1).all()

    def test_D_never_treated_only(self, never_treated_only, config):
        """D=0 for entire panel when no units are treated."""
        panel = MultiEventPanel(never_treated_only, config=config)
        df = panel.build()
        assert (df["D"] == 0).all()

    def test_D_all_treated_same_time(self, all_treated_same_time, config):
        """D transitions from 0 to 1 at the common treatment year."""
        panel = MultiEventPanel(all_treated_same_time, config=config)
        df = panel.build()
        pre = df[df["year"] < 2007]
        post = df[df["year"] >= 2007]
        assert (pre["D"] == 0).all()
        assert (post["D"] == 1).all()

    def test_deduplication(self, simple_panel, config):
        """Duplicate (unit, time) rows are dropped, keeping first."""
        # Create a panel with duplicates
        dup = pd.concat([simple_panel, simple_panel.head(5)], ignore_index=True)
        panel = MultiEventPanel(dup, config=config)
        df = panel.build()
        # Should have same number of rows as original (no dups)
        assert len(df) == len(simple_panel)

    def test_never_treated_only(self, never_treated_only, config):
        panel = MultiEventPanel(never_treated_only, config=config)
        df = panel.build()
        assert (df["treatment_type"] == "never_treated").all()
        assert all(df["years_treated"].apply(lambda x: x == []))


class TestMultiEventMerge:
    def test_merge_preserves_rows(self, simple_panel, outcomes_df, config):
        panel = MultiEventPanel(simple_panel, config=config)
        merged = panel.merge_outcomes(outcomes_df)
        assert len(merged) == len(simple_panel)
        assert "revenue" in merged.columns
