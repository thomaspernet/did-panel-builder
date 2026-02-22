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

    subset = df[df[unit_col].isin(units)].assign(_obs=1)
    presence = subset.pivot_table(index=unit_col, columns=time_col, values="_obs",
                                  aggfunc="sum", fill_value=0)
    presence = (presence > 0).astype(int)

    sns.heatmap(presence, cmap="Blues", cbar_kws={"label": "Observed"}, ax=ax, yticklabels=False)
    ax.set_xlabel("Time", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"Units (n={len(units)})", fontsize=12, fontweight="bold")
    ax.set_title("Panel Observation Heatmap", fontsize=13, fontweight="bold")

    plt.tight_layout()
    return fig


def plot_treatment_summary(
    df: pd.DataFrame,
    unit_col: str = "unit_id",
    time_col: str = "time",
    event_col: str = "has_event",
    time_range: tuple[int, int] | None = None,
    figsize: tuple[int, int] = (14, 5),
):
    """Event rate and event count over time.

    Left panel shows the treatment exposure rate per period with a mean
    reference line.  Right panel shows absolute event counts.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data.
    unit_col : str
        Unit identifier column.
    time_col : str
        Time column.
    event_col : str
        Binary event column.
    time_range : tuple, optional
        ``(min, max)`` time range to display.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    summary = (
        df.groupby(time_col)
        .agg(n_units=(unit_col, "nunique"), n_events=(event_col, "sum"))
        .reset_index()
    )
    summary["event_rate"] = summary["n_events"] / summary["n_units"] * 100

    if time_range is not None:
        summary = summary[
            (summary[time_col] >= time_range[0])
            & (summary[time_col] <= time_range[1])
        ]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: event rate over time
    ax1 = axes[0]
    ax1.fill_between(summary[time_col], summary["event_rate"], alpha=0.3, color=COLORS["treated"])
    ax1.plot(summary[time_col], summary["event_rate"], "s-", color=COLORS["treated"], linewidth=2.5)
    mean_rate = summary["event_rate"].mean()
    ax1.axhline(mean_rate, linestyle="--", color="gray", linewidth=1.5,
                label=f"Mean: {mean_rate:.1f}%")
    ax1.set_xlabel("Time", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Event Rate (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Treatment Exposure Rate", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)

    # Right: event count per period
    ax2 = axes[1]
    ax2.bar(summary[time_col], summary["n_events"], color=COLORS["control"], alpha=0.7)
    ax2.set_xlabel("Time", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Number of Events", fontsize=11, fontweight="bold")
    ax2.set_title("Event Count by Period", fontsize=12, fontweight="bold")
    ax2.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


def plot_treatment_funnel(
    steps: list[tuple[str, int]],
    summary_stats: dict[str, str | int | float] | None = None,
    figsize: tuple[int, int] = (14, 5),
):
    """Horizontal bar funnel showing data retention through pipeline steps.

    Parameters
    ----------
    steps : list of (label, count)
        Ordered pipeline steps from top to bottom.
    summary_stats : dict, optional
        Key-value pairs shown in a text panel beside the funnel.
        If ``None``, the funnel spans the full width.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    palette = [COLORS["info"], COLORS["success"], COLORS["danger"],
               COLORS["muted"], COLORS["treated"]]

    ncols = 2 if summary_stats else 1
    width_ratios = [2, 1] if summary_stats else [1]
    fig, axes = plt.subplots(1, ncols, figsize=figsize,
                             gridspec_kw={"width_ratios": width_ratios})
    if ncols == 1:
        axes = [axes]

    labels = [s[0] for s in steps]
    values = [s[1] for s in steps]
    colors = [palette[i % len(palette)] for i in range(len(steps))]

    ax1 = axes[0]
    y_pos = np.arange(len(labels))
    bars = ax1.barh(y_pos, values, color=colors, alpha=0.8, edgecolor="white")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels)
    ax1.invert_yaxis()
    ax1.set_xlabel("Observations", fontsize=11, fontweight="bold")
    ax1.set_title("Treatment Building Process", fontsize=12, fontweight="bold")
    ax1.grid(alpha=0.3, axis="x")

    max_val = max(values) if values else 1
    for bar, val in zip(bars, values):
        ax1.text(bar.get_width() + max_val * 0.02, bar.get_y() + bar.get_height() / 2,
                 f"{val:,.0f}", va="center", ha="left", fontsize=10)

    # Retention percentages between consecutive steps
    for i in range(1, len(values)):
        if values[i - 1] > 0:
            pct = values[i] / values[i - 1] * 100
            mid_y = (y_pos[i - 1] + y_pos[i]) / 2
            ax1.annotate(
                f"{pct:.0f}%", xy=(max_val * 0.85, mid_y),
                fontsize=9, ha="center", va="center", color="gray",
            )

    ax1.set_xlim(0, max_val * 1.25)

    # Summary text panel
    if summary_stats:
        ax2 = axes[1]
        ax2.axis("off")
        lines = ["SUMMARY", "=" * 30]
        for key, val in summary_stats.items():
            if isinstance(val, float):
                lines.append(f"{key}: {val:,.1f}")
            elif isinstance(val, int):
                lines.append(f"{key}: {val:,}")
            else:
                lines.append(f"{key}: {val}")
        ax2.text(
            0.1, 0.9, "\n".join(lines), transform=ax2.transAxes,
            fontsize=11, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.8),
        )

    plt.tight_layout()
    return fig
