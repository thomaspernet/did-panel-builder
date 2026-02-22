"""Panel diagnostic visualization functions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._style import COLORS


def plot_treatment_distribution(
    df: pd.DataFrame,
    treatment_time_col: str = "first_event_time",
    unit_col: str = "unit_id",
    fill_value: int = -1000,
    ax=None,
):
    """Cohort size histogram showing when units are first treated.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data.
    treatment_time_col : str
        Column with first treatment time.
    unit_col : str
        Column with unit identifier.
    fill_value : int
        Sentinel value for never-treated units.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.get_figure()

    units = df.drop_duplicates(unit_col)
    treated = units[units[treatment_time_col] != fill_value]
    n_never = len(units) - len(treated)

    if len(treated) > 0:
        cohort_counts = treated[treatment_time_col].value_counts().sort_index()
        ax.bar(cohort_counts.index, cohort_counts.values, color=COLORS["treated"], alpha=0.8, edgecolor="black")

        # Cumulative line
        ax2 = ax.twinx()
        cumulative = cohort_counts.cumsum()
        ax2.plot(cumulative.index, cumulative.values, "o-", color=COLORS["control"], linewidth=2.5)
        ax2.set_ylabel("Cumulative treated units", fontsize=11, fontweight="bold")

    ax.set_xlabel("First Treatment Time", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Units", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Treatment Distribution (N treated={len(treated):,}, N never-treated={n_never:,})",
        fontsize=13, fontweight="bold",
    )

    plt.tight_layout()
    return fig


def plot_pre_post_coverage(
    df: pd.DataFrame,
    unit_col: str = "unit_id",
    cnt_pre_col: str = "cnt_pre_periods",
    cnt_post_col: str = "cnt_post_periods",
):
    """Pre/post period distribution and cross-tabulation heatmap.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data with pre/post count columns.
    unit_col : str
        Unit identifier column.
    cnt_pre_col : str
        Column with pre-treatment period count.
    cnt_post_col : str
        Column with post-treatment period count.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    units = df.drop_duplicates(unit_col)

    # Pre distribution
    axes[0].hist(units[cnt_pre_col], bins=20, color=COLORS["pre"], edgecolor="black", alpha=0.8)
    axes[0].set_xlabel("Pre-treatment Periods", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Count", fontsize=11, fontweight="bold")
    axes[0].set_title("Pre-treatment Coverage", fontsize=12, fontweight="bold")

    # Post distribution
    axes[1].hist(units[cnt_post_col], bins=20, color=COLORS["post"], edgecolor="black", alpha=0.8)
    axes[1].set_xlabel("Post-treatment Periods", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Count", fontsize=11, fontweight="bold")
    axes[1].set_title("Post-treatment Coverage", fontsize=12, fontweight="bold")

    # Cross-tab heatmap
    crosstab = pd.crosstab(units[cnt_pre_col], units[cnt_post_col])
    sns.heatmap(crosstab, cmap="YlGnBu", annot=True, fmt="d", ax=axes[2])
    axes[2].set_xlabel("Post-treatment Periods", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Pre-treatment Periods", fontsize=11, fontweight="bold")
    axes[2].set_title("Pre x Post Coverage", fontsize=12, fontweight="bold")

    plt.tight_layout()
    return fig


def plot_outcome_variation(
    df: pd.DataFrame,
    outcome_col: str,
    unit_col: str = "unit_id",
    treatment_type_col: str = "treatment_type",
):
    """Within-unit variation patterns for an outcome.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data.
    outcome_col : str
        Outcome column to analyze.
    unit_col : str
        Unit identifier column.
    treatment_type_col : str
        Treatment type column.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Compute variation per unit
    variation = (
        df.groupby(unit_col)
        .agg(
            n_unique=(outcome_col, "nunique"),
            min_val=(outcome_col, "min"),
            max_val=(outcome_col, "max"),
        )
    )
    variation["pattern"] = "Variable"
    variation.loc[variation["max_val"] == 0, "pattern"] = "Always 0"
    variation.loc[variation["min_val"] == 1, "pattern"] = "Always 1"

    # Pie chart
    pattern_counts = variation["pattern"].value_counts()
    pattern_colors = {
        "Variable": COLORS["success"],
        "Always 0": COLORS["danger"],
        "Always 1": COLORS["info"],
    }
    colors = [pattern_colors.get(p, "gray") for p in pattern_counts.index]
    axes[0].pie(pattern_counts.values, labels=pattern_counts.index, colors=colors,
                autopct="%1.1f%%", startangle=90)
    axes[0].set_title(f"Variation in {outcome_col}", fontsize=12, fontweight="bold")

    # By treatment type
    if treatment_type_col in df.columns:
        unit_types = df.drop_duplicates(unit_col)[[unit_col, treatment_type_col]].set_index(unit_col)
        merged = variation.join(unit_types)
        crosstab = pd.crosstab(merged[treatment_type_col], merged["pattern"])
        crosstab.plot(kind="bar", stacked=True, ax=axes[1],
                      color=[pattern_colors.get(c, "gray") for c in crosstab.columns])
        axes[1].set_title("Variation by Treatment Type", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("")
        axes[1].legend(title="Pattern")
    else:
        axes[1].set_visible(False)

    plt.tight_layout()
    return fig


def plot_coverage_summary(
    df: pd.DataFrame,
    unit_col: str = "unit_id",
    time_col: str = "time",
):
    """2x2 grid: panel length, consecutive periods, gaps, coverage rate.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data.
    unit_col : str
        Unit identifier column.
    time_col : str
        Time column.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    from .._types import PanelConfig
    from ..diagnostics.coverage import CoverageAnalyzer

    config = PanelConfig(unit_col=unit_col, time_col=time_col)
    analyzer = CoverageAnalyzer(config=config)
    coverage = analyzer.compute(df)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = [
        ("n_periods", "Panel Length (Periods)", COLORS["control"]),
        ("n_consecutive", "Max Consecutive Periods", COLORS["pre"]),
        ("n_gaps", "Number of Gaps", COLORS["danger"]),
        ("coverage_rate", "Coverage Rate", COLORS["success"]),
    ]

    for ax, (col, title, color) in zip(axes.flatten(), metrics):
        ax.hist(coverage[col], bins=20, color=color, edgecolor="black", alpha=0.8)
        ax.axvline(coverage[col].mean(), color="black", linestyle="--", linewidth=2,
                   label=f"Mean: {coverage[col].mean():.2f}")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend()

    plt.suptitle("Panel Coverage Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_observation_heatmap(
    df: pd.DataFrame,
    unit_col: str = "unit_id",
    time_col: str = "time",
    sample_units: int = 100,
    random_state: int = 42,
):
    """Presence/absence heatmap for sampled units over time.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data.
    unit_col : str
        Unit identifier column.
    time_col : str
        Time column.
    sample_units : int
        Number of units to sample for the heatmap.
    random_state : int
        Random seed for unit sampling.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(14, 8))

    units = df[unit_col].unique()
    if len(units) > sample_units:
        rng = np.random.default_rng(random_state)
        units = rng.choice(units, sample_units, replace=False)

    subset = df[df[unit_col].isin(units)]
    presence = subset.pivot_table(index=unit_col, columns=time_col, values=unit_col,
                                  aggfunc="count", fill_value=0)
    presence = (presence > 0).astype(int)

    sns.heatmap(presence, cmap="Blues", cbar_kws={"label": "Observed"}, ax=ax, yticklabels=False)
    ax.set_xlabel("Time", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"Units (n={len(units)})", fontsize=12, fontweight="bold")
    ax.set_title("Panel Observation Heatmap", fontsize=13, fontweight="bold")

    plt.tight_layout()
    return fig
