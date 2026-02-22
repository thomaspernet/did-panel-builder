"""Treatment assignment for DiD panel studies."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .._types import PanelConfig

logger = logging.getLogger(__name__)


class TreatmentAssigner:
    """Compute treatment metadata and classify units before panel construction.

    Takes a unit-time panel with a binary event indicator and produces
    an enriched DataFrame with first-event timing, pre/post period counts,
    treatment categories, and diagnostic flags. The output feeds directly
    into ``StaggeredPanel``, ``MultiEventPanel``, or ``StackedPanel``.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data with at least ``unit_col``, ``time_col``, and ``event_col``.
    config : PanelConfig, optional
        Column name mapping. Uses defaults if not provided.
    study_start : int, optional
        First period of the study window. Units treated here are flagged
        as ``early_treated``. If None, ``early_treated`` is always 0.
    study_end : int, optional
        Last period of the study window. Post-period counts are capped
        at observations through this period. If None, no cap is applied.
    min_pre_periods : int
        Threshold for ``insufficient_pre`` flag (default 0).
    min_post_periods : int
        Threshold for ``insufficient_post`` flag (default 0).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: PanelConfig | None = None,
        study_start: int | None = None,
        study_end: int | None = None,
        min_pre_periods: int = 0,
        min_post_periods: int = 0,
    ) -> None:
        self.config = config or PanelConfig()
        self.study_start = study_start
        self.study_end = study_end
        self.min_pre_periods = min_pre_periods
        self.min_post_periods = min_post_periods

        self._df = df.copy()
        self._validate()
        self._result: pd.DataFrame | None = None

    def _validate(self) -> None:
        """Check inputs and coerce types."""
        c = self.config
        required = [c.unit_col, c.time_col, c.event_col]
        missing = [col for col in required if col not in self._df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. "
                f"Available: {sorted(self._df.columns.tolist())}"
            )

        if (
            self.study_start is not None
            and self.study_end is not None
            and self.study_start > self.study_end
        ):
            raise ValueError(
                f"study_start ({self.study_start}) must be <= study_end ({self.study_end})"
            )

        if self.min_pre_periods < 0 or self.min_post_periods < 0:
            raise ValueError("min_pre_periods and min_post_periods must be >= 0")

        self._df[c.unit_col] = self._df[c.unit_col].astype(str)
        self._df[c.time_col] = pd.to_numeric(self._df[c.time_col], errors="coerce")
        self._df[c.event_col] = self._df[c.event_col].astype(bool)

        n_obs = len(self._df)
        n_units = self._df[c.unit_col].nunique()
        logger.info(
            "TreatmentAssigner initialized: %s observations, %s units",
            f"{n_obs:,}",
            f"{n_units:,}",
        )

    def build(self) -> pd.DataFrame:
        """Compute treatment assignment metadata.

        Returns
        -------
        pd.DataFrame
            Input data with added columns: ``first_event_time``,
            ``cnt_pre_periods``, ``cnt_post_periods``, ``early_treated``,
            ``insufficient_pre``, ``insufficient_post``, ``has_pre_data``,
            ``has_post_data``, ``treatment_category``.
        """
        c = self.config
        df = self._df.copy()

        # First event time per unit
        first_event = (
            df[df[c.event_col]]
            .groupby(c.unit_col)[c.time_col]
            .min()
            .to_dict()
        )
        df["first_event_time"] = (
            df[c.unit_col].map(first_event).fillna(c.fill_value).astype(int)
        )

        ever_treated = df["first_event_time"] != c.fill_value

        # Pre/post period counts (unique observed periods)
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
                if self.study_end is not None:
                    post_counts[unit] = int(
                        ((times > fe) & (times <= self.study_end)).sum()
                    )
                else:
                    post_counts[unit] = int((times > fe).sum())

        df["cnt_pre_periods"] = df[c.unit_col].map(pre_counts).astype(int)
        df["cnt_post_periods"] = df[c.unit_col].map(post_counts).astype(int)

        # Early treated flag
        if self.study_start is not None:
            df["early_treated"] = (
                ever_treated & (df["first_event_time"] == self.study_start)
            ).astype(int)
        else:
            df["early_treated"] = 0

        # Insufficient pre/post flags
        df["insufficient_pre"] = (
            ever_treated & (df["cnt_pre_periods"] < self.min_pre_periods)
        ).astype(int)
        df["insufficient_post"] = (
            ever_treated & (df["cnt_post_periods"] < self.min_post_periods)
        ).astype(int)

        # Has pre/post data flags
        df["has_pre_data"] = (ever_treated & (df["cnt_pre_periods"] > 0)).astype(int)
        df["has_post_data"] = (ever_treated & (df["cnt_post_periods"] > 0)).astype(int)

        # Treatment category
        conditions = [
            ~ever_treated,
            (df["has_pre_data"] == 1) & (df["has_post_data"] == 1),
            (df["has_pre_data"] == 1) & (df["has_post_data"] == 0),
            (df["has_pre_data"] == 0) & (df["has_post_data"] == 1),
        ]
        choices = ["never_treated", "has_pre_post", "has_pre_only", "has_post_only"]
        df["treatment_category"] = np.select(conditions, choices, default="has_neither")

        self._result = df
        logger.info(
            "Treatment assigned: %s ever-treated, %s never-treated",
            f"{ever_treated.drop_duplicates().sum():,}" if False else
            f"{df.loc[ever_treated, c.unit_col].nunique():,}",
            f"{df.loc[~ever_treated, c.unit_col].nunique():,}",
        )
        return df

    @property
    def result(self) -> pd.DataFrame:
        """Lazily build and cache the treatment result."""
        if self._result is None:
            self._result = self.build()
        return self._result

    def summary(self) -> pd.DataFrame:
        """Return treatment category breakdown.

        Returns
        -------
        pd.DataFrame
            One row per treatment category with unit counts and percentages.
        """
        c = self.config
        df = self.result

        unit_cats = df.drop_duplicates(c.unit_col)["treatment_category"].value_counts()

        all_categories = [
            "never_treated", "has_pre_post", "has_pre_only",
            "has_post_only", "has_neither",
        ]
        rows = []
        total = unit_cats.sum()
        for cat in all_categories:
            n = unit_cats.get(cat, 0)
            rows.append({
                "treatment_category": cat,
                "n_units": int(n),
                "pct_units": round(100.0 * n / total, 1) if total > 0 else 0.0,
            })

        return pd.DataFrame(rows)

    def filter(
        self,
        drop_early_treated: bool = False,
        drop_insufficient_pre: bool = False,
        drop_insufficient_post: bool = False,
        keep_categories: list[str] | None = None,
    ) -> pd.DataFrame:
        """Filter the treatment DataFrame at the unit level.

        Parameters
        ----------
        drop_early_treated : bool
            Drop units with ``early_treated == 1``.
        drop_insufficient_pre : bool
            Drop units with ``insufficient_pre == 1``.
        drop_insufficient_post : bool
            Drop units with ``insufficient_post == 1``.
        keep_categories : list[str], optional
            Keep only units in these treatment categories.

        Returns
        -------
        pd.DataFrame
            Filtered copy of the treatment DataFrame.
        """
        c = self.config
        df = self.result.copy()

        # Collect units to exclude
        exclude_units: set[str] = set()

        if drop_early_treated:
            flagged = df.loc[df["early_treated"] == 1, c.unit_col].unique()
            exclude_units.update(flagged)

        if drop_insufficient_pre:
            flagged = df.loc[df["insufficient_pre"] == 1, c.unit_col].unique()
            exclude_units.update(flagged)

        if drop_insufficient_post:
            flagged = df.loc[df["insufficient_post"] == 1, c.unit_col].unique()
            exclude_units.update(flagged)

        if keep_categories is not None:
            not_in_cats = df.loc[
                ~df["treatment_category"].isin(keep_categories), c.unit_col
            ].unique()
            exclude_units.update(not_in_cats)

        if exclude_units:
            df = df[~df[c.unit_col].isin(exclude_units)]

        n_before = self.result[c.unit_col].nunique()
        n_after = df[c.unit_col].nunique()
        logger.info(
            "Filtered: %s -> %s units (%s dropped)",
            f"{n_before:,}",
            f"{n_after:,}",
            f"{n_before - n_after:,}",
        )
        return df
