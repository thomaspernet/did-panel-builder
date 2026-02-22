"""Event study visualization functions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._style import COLORS, get_z


def plot_event_time(
    df: pd.DataFrame,
    outcome: str,
    event_time_col: str = "event_time",
    event_window: tuple[int, int] = (-5, 5),
    ci: float = 0.95,
    title: str | None = None,
    ax=None,
    **kwargs,
):
    """Plot mean outcome by event time with confidence interval band.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data with event_time column.
    outcome : str
        Outcome column to plot.
    event_time_col : str
        Column with relative event time.
    event_window : tuple
        Event time range to plot.
    ci : float
        Confidence level (0.90, 0.95, or 0.99).
    title : str, optional
        Plot title.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 6)))
    else:
        fig = ax.get_figure()

    z = get_z(ci)

    # Filter to event window
    df_plot = df[
        (df[event_time_col] >= event_window[0])
        & (df[event_time_col] <= event_window[1])
    ]

    # Stats by event time
    stats = (
        df_plot.groupby(event_time_col)[outcome]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    stats["se"] = stats["std"] / np.sqrt(stats["count"])

    # CI band
    ax.fill_between(
        stats[event_time_col],
        stats["mean"] - z * stats["se"],
        stats["mean"] + z * stats["se"],
        alpha=0.3,
        color=COLORS["control"],
    )
    ax.plot(
        stats[event_time_col],
        stats["mean"],
        "o-",
        color=COLORS["control"],
        linewidth=2.5,
        markersize=8,
    )

    # Event line
    ax.axvline(0, color=COLORS["highlight"], linestyle="--", linewidth=2, alpha=0.7, label="Event")

    # Pre-event mean
    pre_mean = stats[stats[event_time_col] < 0]["mean"].mean()
    ax.axhline(pre_mean, color="gray", linestyle=":", alpha=0.5, label=f"Pre-event mean: {pre_mean:.3f}")

    ax.set_xlabel("Event Time", fontsize=12, fontweight="bold")
    ax.set_ylabel(outcome, fontsize=12, fontweight="bold")
    ax.set_title(title or f"{outcome} Around Event", fontsize=13, fontweight="bold")
    ax.legend()
    ax.set_xticks(range(event_window[0], event_window[1] + 1))

    plt.tight_layout()
    return fig


def plot_by_treatment_status(
    df: pd.DataFrame,
    outcome: str,
    event_time_col: str = "event_time",
    treatment_status_col: str = "treatment_status",
    event_window: tuple[int, int] = (-5, 5),
    figsize: tuple[int, int] = (16, 5),
):
    """Three-panel plot: event-time trend, treatment effect, pre/post comparison.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data.
    outcome : str
        Outcome column.
    event_time_col : str
        Column with relative event time.
    treatment_status_col : str
        Column with treatment status.
    event_window : tuple
        Event time range.
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Filter to ever-treated (treated or not_yet_treated)
    df_plot = df[
        (df[treatment_status_col].isin(["treated", "not_yet_treated"]))
        & (df[event_time_col].notna())
        & (df[outcome].notna())
        & (df[event_time_col] >= event_window[0])
        & (df[event_time_col] <= event_window[1])
    ].copy()

    if len(df_plot) == 0:
        for ax in axes:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        plt.tight_layout()
        return fig

    # Panel 1: outcome by event time
    stats = df_plot.groupby(event_time_col)[outcome].agg(["mean", "std", "count"]).reset_index()
    stats["se"] = stats["std"] / np.sqrt(stats["count"])

    axes[0].fill_between(
        stats[event_time_col],
        stats["mean"] - 1.96 * stats["se"],
        stats["mean"] + 1.96 * stats["se"],
        alpha=0.3, color=COLORS["control"],
    )
    axes[0].plot(stats[event_time_col], stats["mean"], "o-", color=COLORS["control"], linewidth=2.5)
    axes[0].axvline(0, linestyle="--", color=COLORS["highlight"], linewidth=2, alpha=0.7, label="Event")
    pre_mean = stats[stats[event_time_col] < 0]["mean"].mean()
    axes[0].axhline(pre_mean, linestyle=":", color="gray", alpha=0.7, label=f"Pre-event mean: {pre_mean:.3f}")
    axes[0].set_xlabel("Event Time", fontsize=11, fontweight="bold")
    axes[0].set_ylabel(f"Mean {outcome}", fontsize=11, fontweight="bold")
    axes[0].set_title(f"{outcome} by Event Time", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=9)

    # Panel 2: treatment effect (difference from pre-mean)
    diff_from_pre = stats["mean"] - pre_mean
    colors = [COLORS["post"] if et >= 0 else COLORS["pre"] for et in stats[event_time_col]]
    axes[1].bar(stats[event_time_col], diff_from_pre, color=colors, alpha=0.7, edgecolor="black")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].axvline(-0.5, linestyle="--", color=COLORS["highlight"], linewidth=2, alpha=0.7)

    post_mean = stats[stats[event_time_col] >= 0]["mean"].mean()
    att = post_mean - pre_mean
    axes[1].annotate(
        f"ATT = {att:+.3f}",
        xy=(0.95, 0.95), xycoords="axes fraction", ha="right", va="top",
        fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )
    axes[1].set_xlabel("Event Time", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Delta from Pre-Event Mean", fontsize=11, fontweight="bold")
    axes[1].set_title("Treatment Effect Estimate", fontsize=12, fontweight="bold")

    # Panel 3: pre vs post bar chart
    pre_vals = df_plot[df_plot[event_time_col] < 0][outcome]
    post_vals = df_plot[df_plot[event_time_col] >= 0][outcome]
    pre_m, post_m = pre_vals.mean(), post_vals.mean()
    pre_se = pre_vals.std() / np.sqrt(len(pre_vals)) if len(pre_vals) > 0 else 0
    post_se = post_vals.std() / np.sqrt(len(post_vals)) if len(post_vals) > 0 else 0

    bars = axes[2].bar(
        ["Pre-Event", "Post-Event"], [pre_m, post_m],
        yerr=[1.96 * pre_se, 1.96 * post_se],
        color=[COLORS["pre"], COLORS["post"]], edgecolor="black", alpha=0.8, capsize=5,
    )
    for bar, val in zip(bars, [pre_m, post_m]):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    diff = post_m - pre_m
    diff_pct = (diff / pre_m * 100) if pre_m != 0 else 0
    axes[2].set_title(f"Pre vs Post\n(Delta = {diff:+.3f}, {diff_pct:+.1f}%)", fontsize=12, fontweight="bold")
    axes[2].set_ylabel(f"Mean {outcome}", fontsize=11, fontweight="bold")

    plt.tight_layout()
    return fig


def plot_pre_post_comparison(
    df: pd.DataFrame,
    outcomes: list[str],
    treatment_status_col: str = "treatment_status",
    ncols: int = 2,
    figsize: tuple[int, int] | None = None,
):
    """Grid of pre vs post bar charts for multiple outcomes.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data.
    outcomes : list[str]
        Outcome columns to compare.
    treatment_status_col : str
        Column with treatment status.
    ncols : int
        Number of columns in grid.
    figsize : tuple, optional
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    n = len(outcomes)
    nrows = (n + ncols - 1) // ncols
    if figsize is None:
        figsize = (7 * ncols, 5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()

    pre = df[df[treatment_status_col] == "not_yet_treated"]
    post = df[df[treatment_status_col] == "treated"]

    for idx, outcome in enumerate(outcomes):
        ax = axes[idx]
        pre_vals = pre[outcome].dropna()
        post_vals = post[outcome].dropna()

        if len(pre_vals) == 0 or len(post_vals) == 0:
            ax.text(0.5, 0.5, f"No valid data for {outcome}", ha="center", va="center", transform=ax.transAxes)
            continue

        pre_mean, post_mean = pre_vals.mean(), post_vals.mean()
        bars = ax.bar(
            ["Pre-Event", "Post-Event"], [pre_mean, post_mean],
            color=[COLORS["pre"], COLORS["post"]], edgecolor="black", alpha=0.8,
        )
        for bar, val in zip(bars, [pre_mean, post_mean]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        diff = post_mean - pre_mean
        diff_pct = (diff / pre_mean * 100) if pre_mean != 0 else 0
        ax.set_title(f"{outcome}\n(Delta = {diff:+.3f}, {diff_pct:+.1f}%)", fontsize=11, fontweight="bold")

    for idx in range(len(outcomes), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Pre vs Post Event Comparison", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def plot_multi_outcome(
    df: pd.DataFrame,
    outcomes: list[str],
    event_time_col: str = "event_time",
    treatment_status_col: str = "treatment_status",
    event_window: tuple[int, int] = (-5, 5),
    ncols: int = 3,
    figsize: tuple[int, int] | None = None,
    outcome_labels: dict[str, str] | None = None,
):
    """Grid of event-time plots for multiple outcomes.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data.
    outcomes : list[str]
        Outcome columns to plot.
    event_time_col : str
        Column with relative event time.
    treatment_status_col : str
        Column with treatment status.
    event_window : tuple
        Event time range.
    ncols : int
        Number of columns in grid.
    figsize : tuple, optional
        Figure size.
    outcome_labels : dict, optional
        Pretty labels for outcomes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    # Filter to ever-treated and event window
    df_plot = df[
        (df[treatment_status_col].isin(["treated", "not_yet_treated"]))
        & (df[event_time_col].notna())
        & (df[event_time_col] >= event_window[0])
        & (df[event_time_col] <= event_window[1])
    ].copy()

    valid_outcomes = [o for o in outcomes if o in df_plot.columns]
    if not valid_outcomes:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"None of the outcomes found in data.\nRequested: {outcomes}",
                ha="center", va="center")
        return fig

    n = len(valid_outcomes)
    nrows = (n + ncols - 1) // ncols
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten() if n > 1 else [axes]

    if outcome_labels is None:
        outcome_labels = {}

    for idx, outcome in enumerate(valid_outcomes):
        ax = axes[idx]
        data = df_plot[[event_time_col, outcome]].dropna()

        if len(data) == 0:
            ax.text(0.5, 0.5, f"No data for {outcome}", ha="center", va="center")
            continue

        stats = data.groupby(event_time_col)[outcome].agg(["mean", "std", "count"]).reset_index()
        stats["se"] = stats["std"] / np.sqrt(stats["count"])

        ax.fill_between(
            stats[event_time_col],
            stats["mean"] - 1.96 * stats["se"],
            stats["mean"] + 1.96 * stats["se"],
            alpha=0.3, color=COLORS["control"],
        )
        ax.plot(stats[event_time_col], stats["mean"], "o-", color=COLORS["control"], linewidth=2, markersize=6)
        ax.axvline(0, linestyle="--", color=COLORS["highlight"], linewidth=1.5, alpha=0.7)

        pre_stats = stats[stats[event_time_col] < 0]
        if len(pre_stats) > 0:
            pre_mean = pre_stats["mean"].mean()
            ax.axhline(pre_mean, linestyle=":", color="gray", alpha=0.5)
            post_stats = stats[stats[event_time_col] >= 0]
            if len(post_stats) > 0:
                att = post_stats["mean"].mean() - pre_mean
                att_pct = (att / pre_mean * 100) if pre_mean != 0 else 0
                ax.annotate(
                    f"ATT: {att:+.3f} ({att_pct:+.1f}%)",
                    xy=(0.98, 0.02), xycoords="axes fraction", ha="right", va="bottom",
                    fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7),
                )

        label = outcome_labels.get(outcome, outcome.replace("_", " ").title())
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Event Time", fontsize=9)
        ax.set_ylabel("Mean", fontsize=9)
        ax.set_xlim(event_window[0] - 0.5, event_window[1] + 0.5)

    for idx in range(len(valid_outcomes), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Event Study: Multiple Outcomes", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig
