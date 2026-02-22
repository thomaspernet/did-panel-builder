"""Shared fixtures for did-panel-builder tests."""

import numpy as np
import pandas as pd
import pytest

from did_panel_builder import PanelConfig


@pytest.fixture
def config() -> PanelConfig:
    return PanelConfig(unit_col="unit_id", time_col="year", event_col="has_event")


@pytest.fixture
def simple_panel() -> pd.DataFrame:
    """Panel with 10 units, 10 years (2005-2014).

    - Units 1-3: treated at different times (2008, 2010, 2012)
    - Units 4-5: multi-event (events in 2008+2011, 2009+2013)
    - Units 6-10: never treated
    """
    rng = np.random.default_rng(42)
    rows = []

    treatment_map = {
        "1": [2008],
        "2": [2010],
        "3": [2012],
        "4": [2008, 2011],
        "5": [2009, 2013],
    }

    for unit_id in range(1, 11):
        uid = str(unit_id)
        event_years = treatment_map.get(uid, [])
        for year in range(2005, 2015):
            rows.append({
                "unit_id": uid,
                "year": year,
                "has_event": year in event_years,
                "outcome_a": rng.normal(10, 2),
                "outcome_b": int(rng.random() > 0.5),
            })

    return pd.DataFrame(rows)


@pytest.fixture
def never_treated_only() -> pd.DataFrame:
    """Panel where no unit is ever treated."""
    rows = []
    for uid in range(1, 6):
        for year in range(2005, 2010):
            rows.append({
                "unit_id": str(uid),
                "year": year,
                "has_event": False,
                "outcome_a": 1.0,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def all_treated_same_time() -> pd.DataFrame:
    """Panel where all units are treated in the same year."""
    rows = []
    for uid in range(1, 6):
        for year in range(2005, 2010):
            rows.append({
                "unit_id": str(uid),
                "year": year,
                "has_event": year == 2007,
                "outcome_a": 1.0,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def outcomes_df() -> pd.DataFrame:
    """Separate outcome DataFrame for merge testing."""
    rng = np.random.default_rng(99)
    rows = []
    for uid in range(1, 11):
        for year in range(2005, 2015):
            rows.append({
                "unit_id": str(uid),
                "year": year,
                "revenue": rng.normal(100, 20),
                "profit": rng.normal(10, 5),
            })
    return pd.DataFrame(rows)
