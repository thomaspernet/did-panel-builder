"""Panel coverage analysis: gaps, consecutive observations, balance."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._types import PanelConfig


class CoverageAnalyzer:
    """Analyze panel coverage: gaps, consecutive observations, attrition.

    Parameters
    ----------
    config : PanelConfig, optional
        Column name mapping.
    """

    def __init__(self, config: PanelConfig | None = None):
        self.config = config or PanelConfig()

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute unit-level coverage statistics.

        Returns
        -------
        pd.DataFrame
            One row per unit with columns: n_periods, time_span,
            n_consecutive, n_gaps, coverage_rate, min_time, max_time.
        """
        c = self.config

        def _unit_stats(group: pd.DataFrame) -> pd.Series:
            times = sorted(group[c.time_col].dropna().unique())
            if len(times) == 0:
                return pd.Series({
                    "n_periods": 0,
                    "time_span": 0,
                    "n_consecutive": 0,
                    "n_gaps": 0,
                    "coverage_rate": 0.0,
                    "min_time": np.nan,
                    "max_time": np.nan,
                })

            n_periods = len(times)
            time_span = int(times[-1] - times[0]) + 1
            diffs = np.diff(times)
            n_consecutive = int(np.sum(diffs == 1)) + 1 if len(diffs) > 0 else 1
            n_gaps = int(np.sum(diffs > 1))
            coverage_rate = n_periods / time_span if time_span > 0 else 1.0

            return pd.Series({
                "n_periods": n_periods,
                "time_span": time_span,
                "n_consecutive": n_consecutive,
                "n_gaps": n_gaps,
                "coverage_rate": round(coverage_rate, 4),
                "min_time": times[0],
                "max_time": times[-1],
            })

        return (
            df.groupby(c.unit_col)
            .apply(_unit_stats, include_groups=False)
            .reset_index()
        )

    def summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate coverage statistics across all units.

        Parameters
        ----------
        df : pd.DataFrame
            Panel data (not pre-computed coverage stats).

        Returns
        -------
        pd.DataFrame
            One-row DataFrame with mean, median, min, max for each stat.
        """
        coverage = self.compute(df)
        numeric_cols = ["n_periods", "time_span", "n_consecutive", "n_gaps", "coverage_rate"]

        stats = {}
        for col in numeric_cols:
            stats[f"{col}_mean"] = coverage[col].mean()
            stats[f"{col}_median"] = coverage[col].median()
            stats[f"{col}_min"] = coverage[col].min()
            stats[f"{col}_max"] = coverage[col].max()

        stats["n_units"] = len(coverage)
        stats["n_balanced"] = int((coverage["coverage_rate"] == 1.0).sum())
        stats["pct_balanced"] = round(stats["n_balanced"] / stats["n_units"] * 100, 1)

        return pd.DataFrame([stats])
