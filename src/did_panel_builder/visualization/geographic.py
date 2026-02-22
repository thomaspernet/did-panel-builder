"""Geographic treatment visualization functions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def plot_location_events(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    event_col: str = "has_event",
    time_col: str = "time",
    time_periods: list | None = None,
    boundaries: "gpd.GeoDataFrame | None" = None,
    clip_to_boundaries: bool = True,
    crs_plot: str | None = "EPSG:5070",
    ncols: int = 4,
    figsize: tuple[int, int] | None = None,
    markersize_event: int = 12,
    markersize_no_event: int = 8,
    color_event: str = "#d62728",
    color_no_event: str = "#1f77b4",
    title: str = "Location Events (Red) vs No Event (Blue)",
):
    """Scatter map of individual locations coloured by event status.

    Plots each location as a dot on a base map, red for treated / blue
    for untreated, with one subplot per time period.

    Parameters
    ----------
    df : pd.DataFrame
        Location-level data with coordinates and event indicator.
    lat_col, lon_col : str
        Latitude / longitude column names.
    event_col : str
        Binary column (1 = event, 0 = no event).
    time_col : str
        Time column used to split into subplots.
    time_periods : list, optional
        Values of *time_col* to plot.  ``None`` uses all unique values.
    boundaries : geopandas.GeoDataFrame, optional
        Region polygons drawn as the basemap.  When ``None``, points are
        plotted without a basemap.
    clip_to_boundaries : bool
        If ``True`` and *boundaries* is given, drop points outside the
        dissolved boundary polygon.
    crs_plot : str or None
        CRS used for plotting (default Albers Equal Area for CONUS).
        ``None`` skips reprojection and plots in the original CRS.
    ncols : int
        Columns in the subplot grid.
    figsize : tuple, optional
        Figure size.  Auto-computed when ``None``.
    markersize_event, markersize_no_event : int
        Marker sizes for event / non-event dots.
    color_event, color_no_event : str
        Marker colours.
    title : str
        Figure suptitle.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import geopandas as gpd  # noqa: F811
    except ImportError:
        raise ImportError(
            "geopandas is required for geographic plots. "
            "Install with: pip install did-panel-builder[geo]"
        )
    import matplotlib.pyplot as plt

    if time_periods is None:
        time_periods = sorted(df[time_col].unique())

    nrows = int(np.ceil(len(time_periods) / ncols))
    if figsize is None:
        figsize = (4 * ncols, 3.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.atleast_1d(axes).flatten()

    # Prepare basemap
    if boundaries is not None:
        bounds_plot = boundaries.to_crs(crs_plot) if crs_plot else boundaries
        if clip_to_boundaries:
            clip_geom = bounds_plot.dissolve().geometry.iloc[0]

    for i, period in enumerate(time_periods):
        ax = axes_flat[i]
        df_period = df[df[time_col] == period].dropna(subset=[lat_col, lon_col])

        if len(df_period) == 0:
            ax.text(0.5, 0.5, f"No data for {period}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        gdf = gpd.GeoDataFrame(
            df_period,
            geometry=gpd.points_from_xy(df_period[lon_col], df_period[lat_col]),
            crs="EPSG:4326",
        )
        if crs_plot:
            gdf = gdf.to_crs(crs_plot)

        if boundaries is not None and clip_to_boundaries:
            gdf = gdf[gdf.within(clip_geom)]

        # Basemap
        if boundaries is not None:
            bounds_plot.plot(ax=ax, facecolor="white", edgecolor="gray", linewidth=0.5)

        # No-event first (underneath), then events on top
        gdf_no = gdf[gdf[event_col] == 0]
        gdf_ev = gdf[gdf[event_col] == 1]

        gdf_no.plot(ax=ax, color=color_no_event, markersize=markersize_no_event, alpha=0.5)
        gdf_ev.plot(ax=ax, color=color_event, markersize=markersize_event, alpha=0.9)

        ax.set_title(f"{period} (n={len(gdf):,}, events={len(gdf_ev):,})", fontsize=10)
        ax.set_axis_off()

    for i in range(len(time_periods), len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig
