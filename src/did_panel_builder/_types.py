"""Shared types and configuration for did-panel-builder."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PanelConfig:
    """Column name mapping for panel data.

    Every panel builder takes this as its configuration argument.
    Create one and pass it to all builders and diagnostics.

    Parameters
    ----------
    unit_col : str
        Column name for the unit identifier (e.g., firm, county, individual).
    time_col : str
        Column name for the time period (e.g., year, quarter, month).
    event_col : str
        Column name for the binary event indicator (1/True = event occurred).
    fill_value : int
        Sentinel value for never-treated units' first_event_time. Must be
        a value that cannot appear as a real time period.

    Example
    -------
    >>> config = PanelConfig(unit_col="county_id", time_col="year", event_col="has_flood")
    """

    unit_col: str = "unit_id"
    time_col: str = "time"
    event_col: str = "has_event"
    fill_value: int = -1000
