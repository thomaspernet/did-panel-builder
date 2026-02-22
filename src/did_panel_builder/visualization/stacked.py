"""Stacked cohort-specific visualization functions."""

from __future__ import annotations

import pandas as pd

from ._style import COLORS


def plot_stacked_cohort(
    df: pd.DataFrame,
    outcome: str,
    cohort_col: str = "cohort",
    event_time_col: str = "event_time",
    treated_col: str = "treated",
    figsize: tuple[int, int] = (14, 6),
):
    """Pooled event study + cohort heatmap for stacked designs.

    Left panel: treated vs control outcome evolution (pooled across cohorts).
    Right panel: heatmap of treated-unit outcomes by cohort x event time.

    Parameters
    ----------
    df : pd.DataFrame
        Stacked panel data.
    outcome : str
        Outcome column.
    cohort_col : str
        Column with cohort identifier.
    event_time_col : str
        Column with event time.
    treated_col : str
        Column with treated indicator (0/1).
    figsize : tuple
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Panel 1: pooled event study (treated vs control)
    stats = (
        df.groupby([event_time_col, treated_col])[outcome]
        .agg(["mean", "sem"])
        .reset_index()
    )

    for treated_val in [0, 1]:
        data = stats[stats[treated_col] == treated_val]
        label = "Treated" if treated_val == 1 else "Control"
        color = COLORS["treated"] if treated_val == 1 else COLORS["control"]

        axes[0].plot(data[event_time_col], data["mean"], marker="o", linewidth=2, color=color, label=label)
        axes[0].fill_between(
            data[event_time_col],
            data["mean"] - 1.96 * data["sem"],
            data["mean"] + 1.96 * data["sem"],
            alpha=0.2, color=color,
        )

    axes[0].axvline(0, linestyle="--", color="black", linewidth=2, alpha=0.7)
    axes[0].set_xlabel("Event Time", fontsize=12, fontweight="bold")
    axes[0].set_ylabel(f"Mean {outcome}", fontsize=12, fontweight="bold")
    axes[0].set_title("Pooled Event Study (Stacked)", fontsize=13, fontweight="bold")
    axes[0].legend()

    # Panel 2: cohort heatmap (treated only)
    cohort_stats = (
        df[df[treated_col] == 1]
        .groupby([cohort_col, event_time_col])[outcome]
        .mean()
        .unstack(event_time_col)
    )

    if len(cohort_stats) > 0:
        sns.heatmap(
            cohort_stats, cmap="RdYlBu_r",
            center=cohort_stats.mean().mean(),
            ax=axes[1], cbar_kws={"label": f"Mean {outcome}"},
        )
        axes[1].set_xlabel("Event Time", fontsize=12, fontweight="bold")
        axes[1].set_ylabel("Cohort", fontsize=12, fontweight="bold")
        axes[1].set_title("Outcome by Cohort x Event Time", fontsize=13, fontweight="bold")
    else:
        axes[1].text(0.5, 0.5, "No treated cohort data", ha="center", va="center", transform=axes[1].transAxes)

    plt.tight_layout()
    return fig
