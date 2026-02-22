"""Tests for VariationAnalyzer."""

import pandas as pd

from did_panel_builder.diagnostics import VariationAnalyzer


class TestVariationAnalyzer:
    def test_analyze_returns_expected_columns(self, simple_panel, config):
        analyzer = VariationAnalyzer(config=config)
        result = analyzer.analyze(simple_panel, "outcome_b")
        expected = ["unit_id", "n_obs", "mean", "std", "min", "max", "n_unique",
                    "has_variation", "all_zeros", "all_ones"]
        for col in expected:
            assert col in result.columns

    def test_analyze_one_row_per_unit(self, simple_panel, config):
        analyzer = VariationAnalyzer(config=config)
        result = analyzer.analyze(simple_panel, "outcome_b")
        assert len(result) == simple_panel["unit_id"].nunique()

    def test_usable_sample_filters(self, config):
        df = pd.DataFrame({
            "unit_id": ["A"] * 3 + ["B"] * 3,
            "year": [1, 2, 3, 1, 2, 3],
            "has_event": [False] * 6,
            "val": [0, 0, 0, 0, 1, 0],  # A has no variation, B does
        })
        analyzer = VariationAnalyzer(config=config)
        usable = analyzer.usable_sample(df, "val")
        assert set(usable["unit_id"].unique()) == {"B"}

    def test_by_cohort(self, simple_panel, config):
        from did_panel_builder import StaggeredPanel

        panel = StaggeredPanel(simple_panel, config=config)
        df = panel.build()

        analyzer = VariationAnalyzer(config=config)
        result = analyzer.by_cohort(df, "outcome_b")
        assert "n_units" in result.columns
        assert "pct_with_variation" in result.columns
