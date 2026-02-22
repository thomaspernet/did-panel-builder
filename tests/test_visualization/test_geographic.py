"""Tests for geographic visualization (plot_location_events)."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

try:
    import geopandas as gpd
    from shapely.geometry import box

    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

pytestmark = pytest.mark.skipif(not HAS_GEOPANDAS, reason="geopandas not installed")


@pytest.fixture
def boundaries():
    """Simple rectangular boundaries."""
    data = {
        "name": ["A", "B", "C", "D", "E"],
        "geometry": [box(i, 0, i + 1, 1) for i in range(5)],
    }
    return gpd.GeoDataFrame(data, crs="EPSG:4326")


@pytest.fixture
def location_events():
    """Location-level data with coordinates and event indicator."""
    rng = np.random.default_rng(99)
    rows = []
    for year in [2008, 2010, 2012]:
        for _ in range(50):
            rows.append({
                "unit_id": str(rng.integers(1, 20)),
                "year": year,
                "latitude": rng.uniform(25, 48),
                "longitude": rng.uniform(-125, -70),
                "has_event": int(rng.random() > 0.6),
            })
    return pd.DataFrame(rows)


class TestPlotLocationEvents:
    def teardown_method(self):
        plt.close("all")

    def test_returns_figure(self, location_events, boundaries):
        from did_panel_builder.visualization import plot_location_events

        fig = plot_location_events(
            location_events, lat_col="latitude", lon_col="longitude",
            time_col="year", boundaries=boundaries,
            clip_to_boundaries=False, crs_plot=None,
        )
        assert isinstance(fig, Figure)

    def test_specific_periods(self, location_events, boundaries):
        from did_panel_builder.visualization import plot_location_events

        fig = plot_location_events(
            location_events, lat_col="latitude", lon_col="longitude",
            time_col="year", time_periods=[2008, 2012],
            boundaries=boundaries, clip_to_boundaries=False,
            crs_plot=None,
        )
        assert isinstance(fig, Figure)
        visible = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible) >= 2

    def test_without_boundaries(self, location_events):
        from did_panel_builder.visualization import plot_location_events

        fig = plot_location_events(
            location_events, lat_col="latitude", lon_col="longitude",
            time_col="year", boundaries=None, crs_plot=None,
        )
        assert isinstance(fig, Figure)

    def test_custom_colours(self, location_events):
        from did_panel_builder.visualization import plot_location_events

        fig = plot_location_events(
            location_events, lat_col="latitude", lon_col="longitude",
            time_col="year", color_event="green", color_no_event="orange",
            crs_plot=None,
        )
        assert isinstance(fig, Figure)
