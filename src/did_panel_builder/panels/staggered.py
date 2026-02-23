"""Staggered Panel Builder -- Sun & Abraham (2021).

Uses first-treatment timing only, creating clean separation between
treated, not-yet-treated, and never-treated groups.

Reference:
    Sun, L., & Abraham, S. (2021). Estimating dynamic treatment effects in
    event studies with heterogeneous treatment effects. Journal of Econometrics.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ._base import BasePanelBuilder

logger = logging.getLogger(__name__)


class StaggeredPanel(BasePanelBuilder):
    """Build staggered DiD panel for Sun & Abraham (2021) estimation.

    Uses first-treatment timing only. Each unit-period observation is
    classified as treated, not-yet-treated, or never-treated based on
    whether the current period is before or after the unit's first event.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data with ``unit_col``, ``time_col``, and ``event_col``.
    config : PanelConfig, optional
        Column name mapping.

    Example
    -------
    >>> config = PanelConfig(unit_col="firm_id", time_col="year", event_col="has_shock")
    >>> panel = StaggeredPanel(df, config=config)
    >>> df_staggered = panel.build()
    """

    def build(self) -> pd.DataFrame:
        """Build the staggered panel with treatment timing variables.

        Returns
        -------
        pd.DataFrame
            Panel with added columns:

            - ``first_event_time``: first treatment time (fill_value for never-treated)
            - ``event_time``: periods relative to first event (NaN for never-treated)
            - ``treatment_type``: unit-level classification
            - ``treatment_status``: time-varying status (treated/not_yet_treated/never_treated)
            - ``cnt_pre_periods`` / ``cnt_post_periods``: pre/post period counts
            - ``years_in_panel`` / ``nb_years_in_panel``: panel coverage
        """
        c = self.config
        df = self._df.copy()

        # First event time per unit
        first_event = self._compute_first_event()
        df["first_event_time"] = (
            df[c.unit_col].map(first_event).fillna(c.fill_value).astype(int)
        )

        # Panel coverage
        df = self._compute_years_in_panel(df)

        # Pre/post period counts
        df = self._compute_pre_post(df, first_event)

        # Unit-level treatment type
        df = self._assign_treatment_type(df)

        # Time-varying treatment status (for Sun & Abraham)
        conditions = [
            df["first_event_time"] == c.fill_value,
            df[c.time_col] >= df["first_event_time"],
        ]
        choices = ["never_treated", "treated"]
        df["treatment_status"] = np.select(conditions, choices, default="not_yet_treated")

        # Event time (relative to first event)
        df["event_time"] = np.where(
            df["first_event_time"] != c.fill_value,
            df[c.time_col] - df["first_event_time"],
            np.nan,
        )

        self._panel = df
        self._log_summary(df)
        return df

    def filter_sample(
        self,
        df: pd.DataFrame | None = None,
        keep_treatment_types: list[str] | None = None,
        min_pre_periods: int = 0,
        min_post_periods: int = 0,
        min_event_time: int | None = None,
        max_event_time: int | None = None,
    ) -> pd.DataFrame:
        """Filter panel to specific treatment types and minimum coverage.

        Parameters
        ----------
        df : pd.DataFrame, optional
            Panel from ``build()``. If None, uses cached panel.
        keep_treatment_types : list[str], optional
            Treatment types to keep (default: ``["treated", "never_treated"]``).
        min_pre_periods : int
            Minimum pre-treatment periods required for ever-treated units.
        min_post_periods : int
            Minimum post-treatment periods required for ever-treated units.
        min_event_time : int, optional
            Drop observations with ``event_time < min_event_time``.
            Never-treated rows (NaN event_time) are always kept.
        max_event_time : int, optional
            Drop observations with ``event_time > max_event_time``.
            Never-treated rows (NaN event_time) are always kept.

        Returns
        -------
        pd.DataFrame
            Filtered panel.
        """
        if df is None:
            df = self.panel

        if keep_treatment_types is None:
            keep_treatment_types = ["treated", "never_treated"]

        filtered = df[df["treatment_type"].isin(keep_treatment_types)].copy()

        # Trim event window (keep NaN rows = never-treated)
        if min_event_time is not None:
            filtered = filtered[
                filtered["event_time"].isna() | (filtered["event_time"] >= min_event_time)
            ]
        if max_event_time is not None:
            filtered = filtered[
                filtered["event_time"].isna() | (filtered["event_time"] <= max_event_time)
            ]

        # Apply min coverage filters ONLY to ever-treated (not never-treated)
        if min_pre_periods > 0 or min_post_periods > 0:
            never = filtered[filtered["treatment_type"] == "never_treated"]
            ever = filtered[filtered["treatment_type"] != "never_treated"]

            if min_pre_periods > 0:
                ever = ever[ever["cnt_pre_periods"] >= min_pre_periods]
            if min_post_periods > 0:
                ever = ever[ever["cnt_post_periods"] >= min_post_periods]

            filtered = pd.concat([never, ever], ignore_index=True)

        logger.info("Filtered: %s -> %s rows", f"{len(df):,}", f"{len(filtered):,}")
        return filtered

    def add_variation_flag(
        self,
        df: pd.DataFrame,
        outcome_col: str,
    ) -> pd.DataFrame:
        """Add flag for within-unit variation in an outcome.

        Useful for Poisson/logit fixed effects that require variation.

        Parameters
        ----------
        df : pd.DataFrame
            Panel data.
        outcome_col : str
            Column to check for within-unit variation.

        Returns
        -------
        pd.DataFrame
            Panel with ``has_variation_{outcome_col}`` column (0/1).
        """
        c = self.config
        variation = (
            df.groupby(c.unit_col)[outcome_col]
            .apply(lambda x: int(x.nunique() > 1))
            .to_dict()
        )
        df = df.copy()
        df[f"has_variation_{outcome_col}"] = df[c.unit_col].map(variation).astype(int)

        n_with = sum(variation.values())
        n_total = len(variation)
        pct = 100 * n_with / n_total if n_total > 0 else 0
        logger.info(
            "Variation flag: %s/%s units (%.1f%%) have variation in %s",
            f"{n_with:,}",
            f"{n_total:,}",
            pct,
            outcome_col,
        )
        return df

    def _log_summary(self, df: pd.DataFrame) -> None:
        c = self.config
        logger.info("Staggered panel built")
        unit_types = df.drop_duplicates(c.unit_col)["treatment_type"].value_counts()
        for tt, cnt in unit_types.items():
            logger.info("  %s: %s units", tt, f"{cnt:,}")
        for ts, cnt in df["treatment_status"].value_counts().items():
            logger.info("  %s: %s obs", ts, f"{cnt:,}")
