"""Tests for IO export functions."""

import tempfile

import pandas as pd
import pytest

from did_panel_builder import StaggeredPanel
from did_panel_builder.io import to_csv, to_parquet

pyarrow = pytest.importorskip("pyarrow", reason="pyarrow not installed")


class TestExport:
    def _build_panel(self, simple_panel, config):
        panel = StaggeredPanel(simple_panel, config=config)
        return panel.build()

    def test_to_csv(self, simple_panel, config):
        df = self._build_panel(simple_panel, config)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            to_csv(df, f.name)
            result = pd.read_csv(f.name)
            assert len(result) == len(df)

    def test_to_csv_list_columns_converted(self, simple_panel, config):
        df = self._build_panel(simple_panel, config)
        assert "years_in_panel" in df.columns

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            to_csv(df, f.name)
            result = pd.read_csv(f.name)
            assert "years_in_panel" in result.columns
            val = result["years_in_panel"].iloc[0]
            assert isinstance(val, str)
            assert "|" in val

    def test_to_parquet_drops_list_columns(self, simple_panel, config):
        df = self._build_panel(simple_panel, config)

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            to_parquet(df, f.name)
            result = pd.read_parquet(f.name)
            assert "years_in_panel" not in result.columns
            assert "first_event_time" in result.columns
