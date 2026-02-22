"""Tests for CoverageAnalyzer."""

import pandas as pd

from did_panel_builder.diagnostics import CoverageAnalyzer


class TestCoverageAnalyzer:
    def test_compute_returns_expected_columns(self, simple_panel, config):
        analyzer = CoverageAnalyzer(config=config)
        result = analyzer.compute(simple_panel)
        expected = ["unit_id", "n_periods", "time_span", "n_consecutive",
                    "n_gaps", "coverage_rate", "min_time", "max_time"]
        for col in expected:
            assert col in result.columns

    def test_compute_one_row_per_unit(self, simple_panel, config):
        analyzer = CoverageAnalyzer(config=config)
        result = analyzer.compute(simple_panel)
        assert len(result) == simple_panel["unit_id"].nunique()

    def test_balanced_panel_coverage_1(self, simple_panel, config):
        analyzer = CoverageAnalyzer(config=config)
        result = analyzer.compute(simple_panel)
        # simple_panel is balanced (all units have all years)
        assert (result["coverage_rate"] == 1.0).all()
        assert (result["n_gaps"] == 0).all()

    def test_unbalanced_panel(self, config):
        df = pd.DataFrame({
            "unit_id": ["A", "A", "A", "B", "B"],
            "year": [2000, 2002, 2004, 2000, 2001],  # A has gaps
            "has_event": [False] * 5,
        })
        analyzer = CoverageAnalyzer(config=config)
        result = analyzer.compute(df)
        a_row = result[result["unit_id"] == "A"].iloc[0]
        assert a_row["n_gaps"] == 2
        assert a_row["coverage_rate"] < 1.0

    def test_summary(self, simple_panel, config):
        analyzer = CoverageAnalyzer(config=config)
        result = analyzer.summary(simple_panel)
        assert isinstance(result, pd.DataFrame)
        assert "n_units" in result.columns
        assert "pct_balanced" in result.columns
