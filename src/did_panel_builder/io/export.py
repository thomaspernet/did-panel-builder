"""Export utilities for panel data."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _drop_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns containing list values (not serializable to most formats)."""
    list_cols = [
        col for col in df.columns
        if df[col].apply(type).eq(list).any()
    ]
    if list_cols:
        logger.info("Dropping list columns for export: %s", list_cols)
        df = df.drop(columns=list_cols)
    return df


def to_parquet(df: pd.DataFrame, path: str | Path, **kwargs) -> None:
    """Export panel to parquet, dropping list columns.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data.
    path : str or Path
        Output file path.
    **kwargs
        Passed to ``DataFrame.to_parquet()``.
    """
    df_clean = _drop_list_columns(df.copy())
    df_clean.to_parquet(path, index=False, **kwargs)
    logger.info("Exported %s rows to %s", f"{len(df_clean):,}", path)


def to_csv(df: pd.DataFrame, path: str | Path, **kwargs) -> None:
    """Export panel to CSV, converting list columns to pipe-separated strings.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data.
    path : str or Path
        Output file path.
    **kwargs
        Passed to ``DataFrame.to_csv()``.
    """
    df_out = df.copy()
    for col in df_out.columns:
        if df_out[col].apply(type).eq(list).any():
            df_out[col] = df_out[col].apply(
                lambda x: "|".join(str(v) for v in x) if isinstance(x, list) else x
            )
    df_out.to_csv(path, index=False, **kwargs)
    logger.info("Exported %s rows to %s", f"{len(df_out):,}", path)


def to_stata(df: pd.DataFrame, path: str | Path, **kwargs) -> None:
    """Export panel to Stata .dta, dropping unsupported column types.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data.
    path : str or Path
        Output file path.
    **kwargs
        Passed to ``DataFrame.to_stata()``.
    """
    df_clean = _drop_list_columns(df.copy())

    # Stata column names max 32 chars
    rename = {}
    for col in df_clean.columns:
        if len(col) > 32:
            rename[col] = col[:32]
    if rename:
        logger.info("Truncating column names for Stata: %s", rename)
        df_clean = df_clean.rename(columns=rename)

    df_clean.to_stata(path, write_index=False, **kwargs)
    logger.info("Exported %s rows to %s", f"{len(df_clean):,}", path)
