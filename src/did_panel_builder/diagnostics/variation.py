"""Within-unit outcome variation analysis for FE estimation."""

from __future__ import annotations

import pandas as pd

from .._types import PanelConfig


class VariationAnalyzer:
    """Analyze within-unit outcome variation for fixed-effects estimation.

    Fixed-effects estimators (Poisson, logit) require within-unit variation
    in the outcome. This class identifies units without variation and
    computes usable-sample statistics.

    Parameters
    ----------
    config : PanelConfig, optional
        Column name mapping.
    """

    def __init__(self, config: PanelConfig | None = None):
        self.config = config or PanelConfig()

    def analyze(self, df: pd.DataFrame, outcome: str) -> pd.DataFrame:
        """Compute unit-level variation statistics for an outcome.

        Parameters
        ----------
        df : pd.DataFrame
            Panel data.
        outcome : str
            Outcome column to analyze.

        Returns
        -------
        pd.DataFrame
            One row per unit with columns: n_obs, mean, std, min, max,
            n_unique, has_variation, all_zeros, all_ones.
        """
        c = self.config
        grouped = df.groupby(c.unit_col)[outcome]

        stats = grouped.agg(["count", "mean", "std", "min", "max", "nunique"])
        stats.columns = ["n_obs", "mean", "std", "min", "max", "n_unique"]
        stats["has_variation"] = (stats["n_unique"] > 1).astype(int)
        stats["all_zeros"] = (stats["max"] == 0).astype(int)
        stats["all_ones"] = (stats["min"] == 1).astype(int)

        return stats.reset_index()

    def usable_sample(self, df: pd.DataFrame, outcome: str) -> pd.DataFrame:
        """Filter to units with within-unit variation in the outcome.

        Parameters
        ----------
        df : pd.DataFrame
            Panel data.
        outcome : str
            Outcome column.

        Returns
        -------
        pd.DataFrame
            Filtered panel with only units that have variation.
        """
        c = self.config
        var_stats = self.analyze(df, outcome)
        units_with_var = var_stats[var_stats["has_variation"] == 1][c.unit_col]
        return df[df[c.unit_col].isin(units_with_var)].copy()

    def by_cohort(
        self,
        df: pd.DataFrame,
        outcome: str,
        first_event_col: str = "first_event_time",
    ) -> pd.DataFrame:
        """Variation breakdown by treatment cohort.

        Parameters
        ----------
        df : pd.DataFrame
            Panel data with a first-event column.
        outcome : str
            Outcome column.
        first_event_col : str
            Column with first event time.

        Returns
        -------
        pd.DataFrame
            Aggregated variation stats by cohort.
        """
        c = self.config

        # Get unit-level variation
        var_stats = self.analyze(df, outcome)

        # Map first_event to each unit
        unit_cohort = (
            df.drop_duplicates(c.unit_col)[[c.unit_col, first_event_col]]
        )
        merged = var_stats.merge(unit_cohort, on=c.unit_col)

        return (
            merged.groupby(first_event_col)
            .agg(
                n_units=(c.unit_col, "count"),
                n_with_variation=("has_variation", "sum"),
                pct_with_variation=("has_variation", "mean"),
                n_all_zeros=("all_zeros", "sum"),
            )
            .reset_index()
        )
