"""Pre/post treatment diagnostics for difference-in-differences panels.

Answers three questions any DiD researcher asks before estimation:

1. **Pre/post means** — Do outcomes shift around treatment? (raw, not causal)
2. **Within-variation** — Which binary outcomes vary within treated units?
   (required for fixed-effects identification)
3. **Selection gap** — Is there differential selection into the outcome
   between treated and control? (Lee bounds motivation)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .._types import PanelConfig


class PrePostDiagnostics:
    """Run pre/post treatment diagnostics on a built DiD panel.

    Parameters
    ----------
    config : PanelConfig, optional
        Column name mapping.

    Example
    -------
    >>> from did_panel_builder.diagnostics import PrePostDiagnostics
    >>> diag = PrePostDiagnostics(config=config)
    >>> results = diag.analyze(
    ...     df_panel,
    ...     outcomes=["outcome_a", "outcome_b"],
    ...     treatment_col="treatment_type",
    ...     event_time_col="event_time",
    ... )
    >>> diag.print_summary(results)
    >>> fig = diag.plot_summary(results)
    """

    def __init__(self, config: PanelConfig | None = None):
        self.config = config or PanelConfig()

    def analyze(
        self,
        df: pd.DataFrame,
        outcomes: list[str],
        treatment_col: str = "treatment_type",
        event_time_col: str = "event_time",
        treated_value: str = "treated",
        control_value: str = "never_treated",
        threshold: int = 0,
        selection_outcome: str | None = None,
    ) -> dict[str, Any]:
        """Run all three diagnostics and return results.

        Parameters
        ----------
        df : pd.DataFrame
            Built panel with treatment_type and event_time columns.
        outcomes : list[str]
            Outcome columns to analyze.
        treatment_col : str
            Column identifying treatment groups.
        event_time_col : str
            Column with time relative to first event.
        treated_value : str
            Value in ``treatment_col`` for treated units.
        control_value : str
            Value in ``treatment_col`` for control units.
        threshold : int
            Event time threshold: ``< threshold`` is pre, ``>= threshold``
            is post. Default 0.
        selection_outcome : str, optional
            Binary outcome for selection gap analysis. If None, uses the
            first outcome in ``outcomes``.

        Returns
        -------
        dict
            Keys: ``pre_post_means`` (DataFrame), ``within_variation``
            (DataFrame), ``selection_gap`` (dict).
        """
        available = [o for o in outcomes if o in df.columns]
        if not available:
            raise ValueError(
                f"None of the outcomes {outcomes} found in DataFrame. "
                f"Available columns: {sorted(df.columns.tolist())}"
            )

        results: dict[str, Any] = {}

        results["pre_post_means"] = self._compute_pre_post_means(
            df,
            outcomes=available,
            treatment_col=treatment_col,
            event_time_col=event_time_col,
            treated_value=treated_value,
            threshold=threshold,
        )

        results["within_variation"] = self._compute_within_variation(
            df,
            outcomes=available,
            treatment_col=treatment_col,
            treated_value=treated_value,
        )

        sel = selection_outcome or available[0]
        results["selection_gap"] = self._compute_selection_gap(
            df,
            outcome=sel,
            treatment_col=treatment_col,
            event_time_col=event_time_col,
            treated_value=treated_value,
            control_value=control_value,
            threshold=threshold,
        )

        return results

    # ------------------------------------------------------------------
    # Private computation methods
    # ------------------------------------------------------------------

    def _compute_pre_post_means(
        self,
        df: pd.DataFrame,
        outcomes: list[str],
        treatment_col: str,
        event_time_col: str,
        treated_value: str,
        threshold: int,
    ) -> pd.DataFrame:
        """Compare outcome means before and after treatment for treated units."""
        c = self.config
        treated = df[df[treatment_col] == treated_value]
        pre = treated[treated[event_time_col] < threshold]
        post = treated[treated[event_time_col] >= threshold]

        rows = []
        for col in outcomes:
            if col not in df.columns:
                continue
            pre_mean = pre[col].mean() if len(pre) > 0 else np.nan
            post_mean = post[col].mean() if len(post) > 0 else np.nan
            rows.append({
                "outcome": col,
                "pre_mean": pre_mean,
                "post_mean": post_mean,
                "diff": post_mean - pre_mean if not (np.isnan(pre_mean) or np.isnan(post_mean)) else np.nan,
                "pre_obs": pre[col].notna().sum(),
                "post_obs": post[col].notna().sum(),
            })

        return pd.DataFrame(rows)

    def _compute_within_variation(
        self,
        df: pd.DataFrame,
        outcomes: list[str],
        treatment_col: str,
        treated_value: str,
    ) -> pd.DataFrame:
        """Identify treated units with within-unit variation in each outcome.

        For fixed-effects identification, units need at least one 0-to-1
        or 1-to-0 transition. This is most relevant for binary outcomes.
        """
        c = self.config
        treated = df[df[treatment_col] == treated_value]

        rows = []
        for col in outcomes:
            if col not in df.columns:
                continue
            stats = treated.groupby(c.unit_col)[col].agg(["min", "max"])
            n_units = len(stats)
            always_0 = int((stats["max"] == 0).sum())
            always_1 = int((stats["min"] == 1).sum())
            varies = n_units - always_0 - always_1
            rows.append({
                "outcome": col,
                "n_units": n_units,
                "always_0": always_0,
                "always_1": always_1,
                "varies": varies,
                "pct_varies": varies / n_units if n_units > 0 else 0.0,
            })

        return pd.DataFrame(rows)

    def _compute_selection_gap(
        self,
        df: pd.DataFrame,
        outcome: str,
        treatment_col: str,
        event_time_col: str,
        treated_value: str,
        control_value: str,
        threshold: int,
    ) -> dict[str, Any]:
        """Compute excess outcome rate between treated (post) and control.

        Motivated by Lee (2009) bounds: if treatment shifts selection
        into the outcome, the excess rate tells you how many observations
        are "marginal" (would not exist absent treatment).
        """
        post_treated = df[
            (df[treatment_col] == treated_value)
            & (df[event_time_col] >= threshold)
        ]
        control = df[df[treatment_col] == control_value]

        rate_treated = post_treated[outcome].mean() if len(post_treated) > 0 else np.nan
        rate_control = control[outcome].mean() if len(control) > 0 else np.nan

        excess = (
            rate_treated - rate_control
            if not (np.isnan(rate_treated) or np.isnan(rate_control))
            else np.nan
        )

        return {
            "outcome": outcome,
            "rate_treated": rate_treated,
            "rate_control": rate_control,
            "excess_rate": excess,
            "n_treated": len(post_treated),
            "n_control": len(control),
        }

    # ------------------------------------------------------------------
    # Output: print
    # ------------------------------------------------------------------

    def print_summary(self, results: dict[str, Any]) -> None:
        """Print formatted diagnostic summary.

        Parameters
        ----------
        results : dict
            Output of :meth:`analyze`.
        """
        # --- Pre/post means ---
        df_pp = results.get("pre_post_means")
        if df_pp is not None and len(df_pp) > 0:
            print("Pre/post means (treated units)")
            print("=" * 72)
            max_name = max(len(r["outcome"]) for _, r in df_pp.iterrows())
            header = (
                f"  {'Outcome':<{max_name}s}  {'Pre':>8s}  {'Post':>8s}  "
                f"{'Diff':>8s}  {'Pre N':>7s}  {'Post N':>7s}"
            )
            print(header)
            print("  " + "-" * (len(header) - 2))
            for _, row in df_pp.iterrows():
                diff_str = f"{row['diff']:+.4f}" if not np.isnan(row["diff"]) else "N/A"
                print(
                    f"  {row['outcome']:<{max_name}s}  {row['pre_mean']:>8.4f}  "
                    f"{row['post_mean']:>8.4f}  {diff_str:>8s}  "
                    f"{row['pre_obs']:>7,.0f}  {row['post_obs']:>7,.0f}"
                )
            print()

        # --- Within-variation ---
        df_wv = results.get("within_variation")
        if df_wv is not None and len(df_wv) > 0:
            print("Within-unit variation (treated units)")
            print("=" * 72)
            max_name = max(len(r["outcome"]) for _, r in df_wv.iterrows())
            header = (
                f"  {'Outcome':<{max_name}s}  {'Units':>6s}  {'Always 0':>9s}  "
                f"{'Always 1':>9s}  {'Varies':>7s}  {'% Varies':>9s}"
            )
            print(header)
            print("  " + "-" * (len(header) - 2))
            for _, row in df_wv.iterrows():
                print(
                    f"  {row['outcome']:<{max_name}s}  {row['n_units']:>6,}  "
                    f"{row['always_0']:>9,}  {row['always_1']:>9,}  "
                    f"{row['varies']:>7,}  {row['pct_varies']:>8.1%}"
                )
            print()
            print("  Note: 'Varies' = units with at least one transition.")
            print("  Only these units identify the effect under unit FE.")
            print()

        # --- Selection gap ---
        gap = results.get("selection_gap")
        if gap is not None:
            print("Selection gap (Lee bounds motivation)")
            print("=" * 72)
            outcome = gap.get("outcome", "outcome")
            print(f"  Outcome: {outcome}")
            print(f"  Treated (post):  {gap['rate_treated']:.4f}  (n={gap['n_treated']:,})")
            print(f"  Control:         {gap['rate_control']:.4f}  (n={gap['n_control']:,})")
            excess = gap["excess_rate"]
            if not np.isnan(excess):
                print(f"  Excess rate:     {excess:+.4f}")
                if abs(excess) < 0.02:
                    print("  -> Excess rate near zero: selection concern is limited.")
                else:
                    print(
                        f"  -> {abs(excess):.1%} of treated observations are "
                        f"potentially 'marginal' (Lee bounds would trim by this proportion)."
                    )
            print()

    # ------------------------------------------------------------------
    # Output: plots
    # ------------------------------------------------------------------

    def plot_summary(
        self,
        results: dict[str, Any],
        figsize: tuple[int, int] = (16, 5),
    ):
        """Plot 1x3 diagnostic summary.

        Parameters
        ----------
        results : dict
            Output of :meth:`analyze`.
        figsize : tuple
            Figure size.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        from ..visualization._style import COLORS, apply_style

        apply_style()

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        self._plot_pre_post_means(results.get("pre_post_means"), ax=axes[0])
        self._plot_within_variation(results.get("within_variation"), ax=axes[1])
        self._plot_selection_gap(results.get("selection_gap"), ax=axes[2])

        plt.tight_layout()
        return fig

    def _plot_pre_post_means(
        self, df_pp: pd.DataFrame | None, ax
    ) -> None:
        """Horizontal paired bars: pre (blue) vs post (red) per outcome."""
        from ..visualization._style import COLORS

        if df_pp is None or len(df_pp) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title("Pre/Post Means")
            return

        outcomes = df_pp["outcome"].tolist()
        y = np.arange(len(outcomes))
        bar_h = 0.35

        ax.barh(y - bar_h / 2, df_pp["pre_mean"], bar_h,
                label="Pre", color=COLORS["pre"], edgecolor="white")
        ax.barh(y + bar_h / 2, df_pp["post_mean"], bar_h,
                label="Post", color=COLORS["post"], edgecolor="white")

        # Annotate diff
        for i, (_, row) in enumerate(df_pp.iterrows()):
            if not np.isnan(row["diff"]):
                x_pos = max(row["pre_mean"], row["post_mean"])
                ax.text(
                    x_pos * 1.02 + 0.001, i,
                    f"{row['diff']:+.3f}",
                    va="center", fontsize=9, color=COLORS["highlight"],
                )

        ax.set_yticks(y)
        ax.set_yticklabels(outcomes)
        ax.set_xlabel("Mean")
        ax.set_title("Pre/Post Means (Treated)")
        ax.legend(loc="lower right", fontsize=9)
        ax.invert_yaxis()

    def _plot_within_variation(
        self, df_wv: pd.DataFrame | None, ax
    ) -> None:
        """Stacked horizontal bar: always_0 / varies / always_1 per outcome."""
        from ..visualization._style import COLORS

        if df_wv is None or len(df_wv) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title("Within-Unit Variation")
            return

        outcomes = df_wv["outcome"].tolist()
        y = np.arange(len(outcomes))

        n_units = df_wv["n_units"].values
        pct_0 = df_wv["always_0"].values / np.maximum(n_units, 1) * 100
        pct_v = df_wv["varies"].values / np.maximum(n_units, 1) * 100
        pct_1 = df_wv["always_1"].values / np.maximum(n_units, 1) * 100

        ax.barh(y, pct_0, label="Always 0", color=COLORS["muted"],
                edgecolor="white")
        ax.barh(y, pct_v, left=pct_0, label="Varies",
                color=COLORS["success"], edgecolor="white")
        ax.barh(y, pct_1, left=pct_0 + pct_v, label="Always 1",
                color=COLORS["info"], edgecolor="white")

        ax.set_yticks(y)
        ax.set_yticklabels(outcomes)
        ax.set_xlabel("% of Treated Units")
        ax.set_title("Within-Unit Variation (Treated)")
        ax.legend(loc="lower right", fontsize=9)
        ax.set_xlim(0, 100)
        ax.invert_yaxis()

    def _plot_selection_gap(self, gap: dict | None, ax) -> None:
        """Two bars: treated post rate vs control rate."""
        from ..visualization._style import COLORS

        if gap is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title("Selection Gap")
            return

        labels = ["Treated\n(post)", "Control"]
        values = [gap["rate_treated"], gap["rate_control"]]
        colors = [COLORS["treated"], COLORS["control"]]

        bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.5)

        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=10,
                )

        excess = gap.get("excess_rate", np.nan)
        if not np.isnan(excess):
            ax.annotate(
                f"Excess: {excess:+.4f}",
                xy=(0.5, 0.95), xycoords="axes fraction",
                ha="center", fontsize=11, color=COLORS["highlight"],
                fontweight="bold",
            )

        outcome = gap.get("outcome", "outcome")
        ax.set_ylabel(f"Rate ({outcome})")
        ax.set_title("Selection Gap (Lee Bounds)")
        ax.set_ylim(0, max(v for v in values if not np.isnan(v)) * 1.25 + 0.01)
