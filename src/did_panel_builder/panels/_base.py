"""Base class for all DiD panel builders."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .._types import PanelConfig

logger = logging.getLogger(__name__)


class BasePanelBuilder(ABC):
    """Abstract base for all DiD panel construction strategies.

    Subclasses implement ``build()`` to produce a panel DataFrame with
    design-specific treatment timing variables. Shared logic for input
    validation, outcome merging, and summary statistics lives here.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data with at least ``unit_col``, ``time_col``, and ``event_col``.
    config : PanelConfig, optional
        Column name mapping. Uses defaults if not provided.
    """

    def __init__(self, df: pd.DataFrame, config: PanelConfig | None = None):
        self.config = config or PanelConfig()
        self._df = df.copy()
        self._validate_input()
        self._panel: pd.DataFrame | None = None

    def _validate_input(self) -> None:
        """Check required columns exist and coerce types."""
        c = self.config
        required = [c.unit_col, c.time_col, c.event_col]
        missing = [col for col in required if col not in self._df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. "
                f"Available: {sorted(self._df.columns.tolist())}"
            )

        self._df[c.unit_col] = self._df[c.unit_col].astype(str)
        self._df[c.time_col] = pd.to_numeric(self._df[c.time_col], errors="coerce")
        self._df[c.event_col] = self._df[c.event_col].astype(bool)

        n_obs = len(self._df)
        n_units = self._df[c.unit_col].nunique()
        logger.info(
            "%s initialized: %s observations, %s units",
            type(self).__name__,
            f"{n_obs:,}",
            f"{n_units:,}",
        )

    @abstractmethod
    def build(self) -> pd.DataFrame:
        """Construct the panel. Returns a DataFrame with design-specific columns."""
        ...

    @property
    def panel(self) -> pd.DataFrame:
        """Lazily build and cache the panel."""
        if self._panel is None:
            self._panel = self.build()
        return self._panel

    def merge_outcomes(
        self,
        df_outcomes: pd.DataFrame,
        outcome_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Merge outcome data onto the built panel.

        Parameters
        ----------
        df_outcomes : pd.DataFrame
            Must contain ``unit_col`` and ``time_col`` plus outcome columns.
        outcome_cols : list[str], optional
            If provided, only these outcome columns are kept.

        Returns
        -------
        pd.DataFrame
            Panel with outcome columns appended via left join.
        """
        c = self.config
        df_panel = self.panel.copy()

        df_out = df_outcomes.copy()
        df_out[c.unit_col] = df_out[c.unit_col].astype(str)
        df_out[c.time_col] = pd.to_numeric(df_out[c.time_col], errors="coerce")

        if outcome_cols:
            keep = [c.unit_col, c.time_col] + outcome_cols
            df_out = df_out[[col for col in keep if col in df_out.columns]]

        merged = df_panel.merge(df_out, on=[c.unit_col, c.time_col], how="left")
        logger.info("Merged with outcomes: %s rows", f"{len(merged):,}")
        return merged

    def summary(self) -> pd.DataFrame:
        """Return panel summary statistics.

        Returns
        -------
        pd.DataFrame
            One-row DataFrame with n_obs, n_units, time_range, and
            treatment_type counts.
        """
        c = self.config
        df = self.panel

        stats = {
            "n_obs": len(df),
            "n_units": df[c.unit_col].nunique(),
            "time_min": df[c.time_col].min(),
            "time_max": df[c.time_col].max(),
        }

        if "treatment_type" in df.columns:
            unit_types = df.drop_duplicates(c.unit_col)["treatment_type"].value_counts()
            for tt, cnt in unit_types.items():
                stats[f"n_{tt}"] = cnt

        return pd.DataFrame([stats])

    def _compute_first_event(self) -> dict:
        """Compute first event time per unit. Returns {unit_id: first_time}."""
        c = self.config
        return (
            self._df[self._df[c.event_col]]
            .groupby(c.unit_col)[c.time_col]
            .min()
            .to_dict()
        )

    def _compute_years_in_panel(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add years_in_panel (list) and nb_years_in_panel (count) columns."""
        c = self.config
        years_map = df.groupby(c.unit_col)[c.time_col].apply(sorted).to_dict()
        df = df.copy()
        df["years_in_panel"] = df[c.unit_col].map(years_map)
        df["nb_years_in_panel"] = df["years_in_panel"].apply(len)
        return df

    def _compute_pre_post(
        self, df: pd.DataFrame, first_event: dict
    ) -> pd.DataFrame:
        """Add pre/post period counts based on first event timing.

        Counts actual observed periods before/after first event (not time range),
        so unbalanced panels with gaps are handled correctly.
        """
        c = self.config
        df = df.copy()

        pre_counts: dict[str, int] = {}
        post_counts: dict[str, int] = {}

        for unit, grp in df.groupby(c.unit_col):
            fe = first_event.get(unit)
            if fe is None:
                pre_counts[unit] = 0
                post_counts[unit] = 0
            else:
                times = grp[c.time_col].unique()
                pre_counts[unit] = int((times < fe).sum())
                post_counts[unit] = int((times > fe).sum())

        df["cnt_pre_periods"] = df[c.unit_col].map(pre_counts).astype(int)
        df["cnt_post_periods"] = df[c.unit_col].map(post_counts).astype(int)

        return df

    def _assign_treatment_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add treatment_type column: never_treated, treated, treated_no_pre, treated_no_post."""
        c = self.config
        df = df.copy()

        conditions = [
            df["first_event_time"] == c.fill_value,
            df["cnt_pre_periods"] == 0,
            df["cnt_post_periods"] == 0,
        ]
        choices = ["never_treated", "treated_no_pre", "treated_no_post"]
        df["treatment_type"] = np.select(conditions, choices, default="treated")

        return df
