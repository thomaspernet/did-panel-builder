"""Tests for visualization style utilities."""

import matplotlib
matplotlib.use("Agg")

import pytest

from did_panel_builder.visualization._style import COLORS, Z_VALUES, apply_style, get_z


class TestGetZ:
    def test_known_values(self):
        assert get_z(0.95) == 1.960
        assert get_z(0.90) == 1.645
        assert get_z(0.99) == 2.576

    def test_invalid_ci(self):
        with pytest.raises(ValueError, match="Unsupported CI"):
            get_z(0.50)


class TestApplyStyle:
    def test_runs_without_error(self):
        apply_style()


class TestColors:
    def test_has_required_keys(self):
        for key in ["treated", "control", "pre", "post", "highlight"]:
            assert key in COLORS
