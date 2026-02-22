"""Stacked Cohort Panel Builder -- Roth et al. (2022).

Creates separate clean 2x2 experiments for each treatment cohort,
with balanced event-time windows and appropriate control groups.

Reference:
    Roth, J., Sant'Anna, P. H., Bilinski, A., & Poe, J. (2022).
    What's trending in difference-in-differences? A synthesis of the
    recent econometrics literature. Journal of Econometrics.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .._types import PanelConfig
from ._base import BasePanelBuilder

logger = logging.getLogger(__name__)


class StackedPanel(BasePanelBuilder):
    """Build stacked cohort panel for Roth et al. (2022) estimation.

    Creates separate sub-experiments for each treatment cohort (units
    first treated at time g), with each cohort matched to appropriate
    controls within a balanced event window.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data with ``unit_col``, ``time_col``, and ``event_col``.
    config : PanelConfig, optional
        Column name mapping.
    time_pre : int
        Number of pre-treatment periods in the event window (default: 3).
    time_post : int
        Number of post-treatment periods in the event window (default: 3).
    cohort_times : list[int], optional
        Specific cohort times to use. If None, derived from the unique
        first-event times in the data, keeping only those with enough
        room for the full event window.

    Example
    -------
    >>> config = PanelConfig(unit_col="firm_id", time_col="year", event_col="has_shock")
    >>> panel = StackedPanel(df, config=config, time_pre=3, time_post=3)
    >>> df_stacked = panel.build()
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: PanelConfig | None = None,
        time_pre: int = 3,
        time_post: int = 3,
        cohort_times: list[int] | None = None,
    ):
        super().__init__(df, config)
        self.time_pre = time_pre
        self.time_post = time_post

        c = self.config

        # Compute first_event_time if not present
        if "first_event_time" not in self._df.columns:
            first_event = self._compute_first_event()
            self._df["first_event_time"] = (
                self._df[c.unit_col]
                .map(first_event)
                .fillna(c.fill_value)
            )

        self._df["first_event_time"] = (
            pd.to_numeric(self._df["first_event_time"], errors="coerce")
            .fillna(c.fill_value)
            .astype(int)
        )

        # Derive cohort times from data if not provided
        if cohort_times is None:
            time_min = int(self._df[c.time_col].min())
            time_max = int(self._df[c.time_col].max())

            # All unique first-event times (excluding never-treated)
            treated_times = self._df.loc[
                self._df["first_event_time"] != c.fill_value, "first_event_time"
            ].unique()

            # Keep cohorts with enough room for the event window
            self.cohort_times = sorted(
                int(t)
                for t in treated_times
                if t - time_pre >= time_min and t + time_post <= time_max
            )
        else:
            self.cohort_times = cohort_times

        logger.info(
            "Event window: -%s to +%s, cohort times: %s",
            time_pre,
            time_post,
            self.cohort_times,
        )

    def build(self) -> pd.DataFrame:
        """Build the stacked cohort panel.

        Returns
        -------
        pd.DataFrame
            Stacked panel with added columns:

            - ``cohort``: treatment cohort time
            - ``event_time``: periods relative to cohort treatment time
            - ``treated``: 1 if unit treated in this cohort, 0 if control
            - ``post``: 1 if event_time >= 0
            - ``cohort_status``: 'treated' or 'control'
            - ``control_type``: 'treated', 'never_treated', 'not_yet_treated', 'already_treated'
            - ``unit_cohort_id``: unit + "_" + cohort (for cohort-specific FE)
            - ``years_in_panel`` / ``nb_years_in_panel``: panel coverage
        """
        cohort_dfs = []

        for g in self.cohort_times:
            df_g = self._build_cohort(g)
            if len(df_g) > 0:
                cohort_dfs.append(df_g)

        if not cohort_dfs:
            raise ValueError(
                f"No valid cohorts found. cohort_times={self.cohort_times}. "
                "Check that your data has treated units in these periods."
            )

        df_stacked = pd.concat(cohort_dfs, ignore_index=True)

        c = self.config

        # Cohort-specific unit ID for fixed effects
        df_stacked["unit_cohort_id"] = (
            df_stacked[c.unit_col].astype(str)
            + "_"
            + df_stacked["cohort"].astype(str)
        )

        # Panel coverage (across all cohorts, at the unit level)
        df_stacked = self._compute_years_in_panel(df_stacked)

        self._panel = df_stacked

        logger.info("Stacked panel built")
        logger.info("  Total observations: %s", f"{len(df_stacked):,}")
        logger.info("  Cohorts: %s", df_stacked["cohort"].nunique())
        logger.info("  Unique units: %s", f"{df_stacked[c.unit_col].nunique():,}")
        for ct, cnt in df_stacked["control_type"].value_counts().items():
            logger.info("  %s: %s", ct, f"{cnt:,}")

        return df_stacked

    def cohort_summary(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Generate cohort-level summary statistics.

        Parameters
        ----------
        df : pd.DataFrame, optional
            Stacked panel. If None, uses cached panel.

        Returns
        -------
        pd.DataFrame
            Summary by cohort with unit counts, treated counts, and event window.
        """
        if df is None:
            df = self.panel

        c = self.config

        summary = (
            df.groupby("cohort")
            .agg(
                n_firms=(c.unit_col, "nunique"),
                n_treated_obs=("treated", "sum"),
                event_window=("event_time", lambda x: f"[{int(x.min())}, {int(x.max())}]"),
            )
        )

        treated_units = (
            df[df["treated"] == 1]
            .groupby("cohort")[c.unit_col]
            .nunique()
            .rename("n_treated_units")
        )

        return summary.join(treated_units)

    def _build_cohort(self, g: int) -> pd.DataFrame:
        """Build dataset for a single cohort g."""
        c = self.config

        df_treated = self._get_treated_units(g)
        df_controls = self._get_control_units(g)

        if len(df_treated) == 0:
            return pd.DataFrame()

        # Tag treated
        df_treated = df_treated.copy()
        df_treated["cohort"] = g
        df_treated["treated"] = 1
        df_treated["cohort_status"] = "treated"
        df_treated["control_type"] = "treated"

        # Tag controls
        df_controls = df_controls.copy()
        df_controls["cohort"] = g
        df_controls["treated"] = 0
        df_controls["cohort_status"] = "control"

        # Combine
        df_cohort = pd.concat([df_treated, df_controls], ignore_index=True)

        # Event time relative to cohort time g
        df_cohort["event_time"] = (df_cohort[c.time_col] - g).astype(int)

        # Trim to balanced event window
        df_cohort = df_cohort[
            (df_cohort["event_time"] >= -self.time_pre)
            & (df_cohort["event_time"] <= self.time_post)
        ]

        # Post indicator
        df_cohort["post"] = (df_cohort["event_time"] >= 0).astype(int)

        return df_cohort

    def _get_treated_units(self, g: int) -> pd.DataFrame:
        """Get units first treated at time g."""
        return self._df[self._df["first_event_time"] == g]

    def _get_control_units(self, g: int) -> pd.DataFrame:
        """Get control units for cohort g.

        Controls include:
        1. Never-treated (first_event_time == fill_value)
        2. Not-yet-treated (first_event_time > g + time_post)
        3. Already-treated far past (first_event_time < g - time_pre)
        """
        c = self.config
        df = self._df

        mask_never = df["first_event_time"] == c.fill_value
        mask_not_yet = df["first_event_time"] > g + self.time_post
        mask_already = df["first_event_time"] < g - self.time_pre

        df_controls = df[mask_never | mask_not_yet | mask_already].copy()

        # Label control type
        df_controls["control_type"] = np.select(
            [
                df_controls["first_event_time"] == c.fill_value,
                df_controls["first_event_time"] > g + self.time_post,
            ],
            ["never_treated", "not_yet_treated"],
            default="already_treated",
        )

        return df_controls
