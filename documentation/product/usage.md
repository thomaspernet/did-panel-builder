# Product Usage

## Getting Started

Install from PyPI. Pick the extras that match what you need:

```bash
pip install did-panel-builder            # core (pandas + numpy)
pip install "did-panel-builder[viz]"     # + matplotlib / seaborn plots
pip install "did-panel-builder[geo]"     # + geopandas maps
pip install "did-panel-builder[io]"      # + parquet export
pip install "did-panel-builder[all]"     # everything
```

Your input is a DataFrame with three columns:

| Role | Type | Example |
|---|---|---|
| Unit ID | str / int | `county_id`, `firm_id` |
| Time | int / float | year, quarter, month |
| Event | bool / 0-1 | whether treatment occurred in this unit-period |

Tell the library which columns are which via `PanelConfig`:

```python
from did_panel_builder import PanelConfig, StaggeredPanel

config = PanelConfig(unit_col="county_id", time_col="year", event_col="has_flood")
panel = StaggeredPanel(df, config=config).build()
```

## Common Workflows

### 1. Classify the sample before building the panel

`TreatmentAssigner` is the step that decides which units make it into your estimation sample.

```python
from did_panel_builder import TreatmentAssigner

assigner = TreatmentAssigner(
    df, config=config,
    study_start=2005, study_end=2020,
    min_pre_periods=2, min_post_periods=1,
)
assigner.build()
assigner.summary()                    # counts by treatment_category
df_clean = assigner.filter(
    drop_early_treated=True,
    drop_insufficient_pre=True,
    keep_categories=["never_treated", "has_pre_post"],
)
```

### 2. Pick a panel design

| You want... | Use | Reference |
|---|---|---|
| First-treatment timing only | `StaggeredPanel` | Sun & Abraham (2021) |
| Multiple treatments per unit | `MultiEventPanel` | de Chaisemartin & D'Haultfoeuille (2020) |
| Stacked 2×2 per cohort | `StackedPanel` | Roth et al. (2022) |

All three take the same `PanelConfig` and expose `.build()`. `StaggeredPanel` and `MultiEventPanel` add a `filter_sample(...)` for event-window and coverage trimming; `StackedPanel` adds `cohort_summary()`.

### 3. Merge outcomes

```python
df_outcomes = pd.read_csv("outcomes.csv")
df_merged = panel.merge_outcomes(df_outcomes, outcome_cols=["gdp", "population"])
```

### 4. Run diagnostics

Before estimating, check sample health:

```python
from did_panel_builder.diagnostics import (
    VariationAnalyzer,   # within-unit outcome variation (needed for FE)
    CoverageAnalyzer,    # gaps, consecutive periods, balance
    PrePostDiagnostics,  # pre/post balance, means, plot summary
)
```

### 5. Visualize

`did_panel_builder.visualization` returns `matplotlib.Figure` objects — so you can save, tweak, or drop them into a notebook directly. Standard plots: `plot_event_time`, `plot_by_treatment_status`, `plot_multi_outcome`, `plot_stacked_cohort`, `plot_treatment_funnel`, plus geographic `plot_location_events`.

### 6. Export

```python
from did_panel_builder.io import to_parquet, to_csv, to_stata
to_parquet(df_panel, "panel.parquet")   # drops list columns
to_csv(df_panel, "panel.csv")           # flattens list cols to pipe-separated
to_stata(df_panel, "panel.dta")         # drops unsupported types
```

## Examples

The `README.md` at the repo root has a full end-to-end walkthrough for each of the three panel designs, plus diagnostics, visualization, and export. Start there.

## Troubleshooting

- **`ImportError` when plotting or exporting.** The base install is pandas + numpy only. Plots need `pip install "did-panel-builder[viz]"`; geographic plots need `[geo]`; parquet export needs `[io]`.
- **"My outcome column has no variation"** — within-unit variation matters for fixed-effects estimation. Use `VariationAnalyzer` to see which units carry enough variation to stay in the estimation sample; use `panel.add_variation_flag(...)` to mark them on the panel.
- **Too few pre/post periods after filtering.** Relax `min_pre_periods` / `min_post_periods` in `TreatmentAssigner`, or widen `study_start` / `study_end`. `assigner.summary()` shows exactly how many units each rule drops.
- **`StackedPanel` cohort times.** If you don't pass `cohort_times` explicitly, they are auto-derived from the data — check `panel.cohort_summary()` to see which cohorts were built.
