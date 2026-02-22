"""Multi-Event Panel Builder -- de Chaisemartin & D'Haultfoeuille (2020).

Allows units to experience multiple treatment events over time,
appropriate for estimators that handle repeated treatments.

Reference:
    de Chaisemartin, C., & D'Haultfoeuille, X. (2020). Two-way fixed effects
    estimators with heterogeneous treatment effects. American Economic Review.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ._base import BasePanelBuilder

logger = logging.getLogger(__name__)


class MultiEventPanel(BasePanelBuilder):
    """Build multi-event panel for de Chaisemartin & D'Haultfoeuille estimation.

    Allows units to experience multiple events over time. Each event time
    is recorded, enabling analysis of dynamic treatment effects with
    repeated shocks.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data with ``unit_col``, ``time_col``, and ``event_col``.
    config : PanelConfig, optional
        Column name mapping.

    Example
    -------
    >>> config = PanelConfig(unit_col="firm_id", time_col="year", event_col="has_shock")
    >>> panel = MultiEventPanel(df, config=config)
    >>> df_multi = panel.build()
    """

    def build(self) -> pd.DataFrame:
        """Build the multi-event panel with treatment timing variables.

        Returns
        -------
        pd.DataFrame
            Panel with added columns:

            - ``first_event_time``: first treatment time (fill_value for never-treated)
            - ``years_treated``: list of ALL event times per unit
            - ``event_time``: periods relative to first event (NaN for never-treated)
            - ``treatment_timing``: first_event_time only at the event period, else fill_value
              (used by dCDH estimator)
            - ``treatment_type``: unit-level classification
            - ``cnt_pre_periods`` / ``cnt_post_periods``: pre/post period counts
            - ``years_in_panel`` / ``nb_years_in_panel``: panel coverage
        """
        c = self.config
        df = self._df.copy()

        # Panel coverage
        df = self._compute_years_in_panel(df)

        # All event times per unit
        all_events = (
            df[df[c.event_col]]
            .groupby(c.unit_col)[c.time_col]
            .apply(lambda x: sorted(x.unique().tolist()))
            .to_dict()
        )
        df["years_treated"] = df[c.unit_col].map(all_events).apply(
            lambda x: x if isinstance(x, list) else []
        )

        # First event time
        first_event = self._compute_first_event()
        df["first_event_time"] = (
            df[c.unit_col].map(first_event).fillna(c.fill_value).astype(int)
        )

        # Pre/post period counts
        df = self._compute_pre_post(df, first_event)

        # Event time (relative to first event)
        df["event_time"] = np.where(
            df["first_event_time"] != c.fill_value,
            df[c.time_col] - df["first_event_time"],
            np.nan,
        )

        # Treatment timing: first_event_time ONLY at the event period, else fill_value
        # Used by de Chaisemartin-D'Haultfoeuille estimator
        df["treatment_timing"] = np.where(
            (df["first_event_time"] > 0) & (df[c.time_col] == df["first_event_time"]),
            df["first_event_time"],
            c.fill_value,
        )

        # Treatment type classification
        df = self._assign_treatment_type(df)

        self._panel = df
        self._log_summary(df)
        return df

    def _log_summary(self, df: pd.DataFrame) -> None:
        logger.info("Multi-event panel built")
        for tt, cnt in df["treatment_type"].value_counts().items():
            logger.info("  %s: %s obs", tt, f"{cnt:,}")
