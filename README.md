# did-panel-builder

![Tests](https://img.shields.io/badge/tests-151%20passed-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

Build panel datasets for difference-in-differences estimation.

Three panel construction strategies, covering the main modern DiD designs:

| Builder | Design | Reference |
|---|---|---|
| `StaggeredPanel` | First-treatment timing | Sun & Abraham (2021) |
| `MultiEventPanel` | Multiple treatments per unit | de Chaisemartin & D'Haultfoeuille (2020) |
| `StackedPanel` | Stacked cohort 2x2 experiments | Roth et al. (2022) |

## Installation

```bash
pip install did-panel-builder

# With visualization support
pip install "did-panel-builder[viz]"

# With geographic plots (geopandas)
pip install "did-panel-builder[geo]"

# With parquet export
pip install "did-panel-builder[io]"

# Everything
pip install "did-panel-builder[all]"
```

## Quick Start

```python
import pandas as pd
from did_panel_builder import PanelConfig, StaggeredPanel

# Your data: a DataFrame with unit, time, and event columns
df = pd.read_csv("my_data.csv")

# Tell the library which columns are which
config = PanelConfig(
    unit_col="county_id",
    time_col="year",
    event_col="has_flood",
)

# Build the panel
panel = StaggeredPanel(df, config=config)
df_panel = panel.build()

# Filter to estimation sample
df_clean = panel.filter_sample(
    min_event_time=-3, max_event_time=5,
    min_pre_periods=2, min_post_periods=1,
)

# Merge outcomes from another DataFrame
df_outcomes = pd.read_csv("outcomes.csv")
df_merged = panel.merge_outcomes(df_outcomes, outcome_cols=["gdp", "population"])
```

## Treatment Assignment

Before building panels, use `TreatmentAssigner` to classify units and prepare the sample. This computes first-event timing, pre/post period counts, treatment categories, and diagnostic flags.

```python
from did_panel_builder import TreatmentAssigner, PanelConfig, StaggeredPanel

config = PanelConfig(unit_col="firm_id", time_col="year", event_col="has_shock")

# Step 1: Assign treatment metadata
assigner = TreatmentAssigner(
    df, config=config,
    study_start=2005, study_end=2020,
    min_pre_periods=2, min_post_periods=1,
)
df_treatment = assigner.build()

# Step 2: Inspect the sample
assigner.summary()
#   treatment_category  n_units  pct_units
#   never_treated          150       30.0
#   has_pre_post           300       60.0
#   has_pre_only            25        5.0
#   has_post_only           15        3.0
#   has_neither             10        2.0

# Step 3: Filter to estimation sample
df_clean = assigner.filter(
    drop_early_treated=True,
    drop_insufficient_pre=True,
    keep_categories=["never_treated", "has_pre_post"],
)

# Step 4: Build panel
panel = StaggeredPanel(df_clean, config=config)
df_panel = panel.build()
```

### Output columns

| Column | Type | Description |
|---|---|---|
| `first_event_time` | int | First treatment time; `fill_value` for never-treated |
| `cnt_pre_periods` | int | Unique observed periods before first event |
| `cnt_post_periods` | int | Unique observed periods after first event (capped at `study_end`) |
| `early_treated` | 0/1 | 1 if treated in the `study_start` period |
| `insufficient_pre` | 0/1 | 1 if pre-periods < `min_pre_periods` |
| `insufficient_post` | 0/1 | 1 if post-periods < `min_post_periods` |
| `has_pre_data` | 0/1 | 1 if any pre-treatment observations |
| `has_post_data` | 0/1 | 1 if any post-treatment observations |
| `treatment_category` | str | `never_treated`, `has_pre_post`, `has_pre_only`, `has_post_only`, `has_neither` |

## Panel Builders

### StaggeredPanel (Sun & Abraham 2021)

Uses first-treatment timing only. Classifies each observation as `treated`, `not_yet_treated`, or `never_treated`.

```python
from did_panel_builder import PanelConfig, StaggeredPanel

config = PanelConfig(unit_col="firm_id", time_col="year", event_col="has_shock")
panel = StaggeredPanel(df, config=config)
df_stag = panel.build()

# Output columns: first_event_time, event_time, treatment_type,
#   treatment_status, cnt_pre_periods, cnt_post_periods

# Filter to usable sample (mirrors R prepare_sunab pattern)
df_clean = panel.filter_sample(
    keep_treatment_types=["treated", "never_treated"],
    min_event_time=-3,       # trim event window
    max_event_time=5,
    min_pre_periods=2,       # minimum coverage
    min_post_periods=1,
)

# Check within-unit variation (needed for FE logit/Poisson)
df_flagged = panel.add_variation_flag(df_clean, "has_disclosure")
```

### MultiEventPanel (de Chaisemartin & D'Haultfoeuille 2020)

Allows multiple treatment events per unit. Tracks all event times.

```python
from did_panel_builder import MultiEventPanel

panel = MultiEventPanel(df, config=config)
df_multi = panel.build()

# Output columns: first_event_time, years_treated (list of all events),
#   event_time, treatment_timing (for dCDH estimator), treatment_type
```

### StackedPanel (Roth et al. 2022)

Creates separate clean 2x2 experiments per treatment cohort.

```python
from did_panel_builder import StackedPanel

panel = StackedPanel(
    df,
    config=config,
    time_pre=3,      # 3 pre-treatment periods
    time_post=3,     # 3 post-treatment periods
    # cohort_times auto-derived from data if not specified
)
df_stacked = panel.build()

# Output columns: cohort, event_time, treated, post,
#   cohort_status, control_type, unit_cohort_id

# Cohort-level summary
panel.cohort_summary()
```

## Diagnostics

```python
from did_panel_builder.diagnostics import (
    CoverageAnalyzer,
    PrePostDiagnostics,
    VariationAnalyzer,
)

# Within-unit outcome variation (needed for FE estimation)
var = VariationAnalyzer(config=config)
stats = var.analyze(df_panel, outcome="revenue")
usable = var.usable_sample(df_panel, outcome="revenue")

# Panel coverage: gaps, consecutive periods, balance
cov = CoverageAnalyzer(config=config)
coverage = cov.compute(df_panel)
summary = cov.summary(df_panel)

# Pre/post treatment diagnostics (print, DataFrame, plot)
diag = PrePostDiagnostics(config=config)
results = diag.analyze(
    df_panel,
    outcomes=["revenue", "profit", "has_disclosure"],
    treatment_col="treatment_type",
    event_time_col="event_time",
    selection_outcome="has_disclosure",  # for Lee bounds gap
)
diag.print_summary(results)       # formatted text
df_means = results["pre_post_means"]  # DataFrame
fig = diag.plot_summary(results)  # 1x3 matplotlib figure
```

## Visualization

All visualization functions return a `matplotlib.figure.Figure`.

```python
from did_panel_builder.visualization import (
    # Event study
    plot_event_time,
    plot_by_treatment_status,
    plot_multi_outcome,
    plot_pre_post_comparison,
    # Stacked cohort
    plot_stacked_cohort,
    # Panel diagnostics
    plot_treatment_distribution,
    plot_treatment_summary,
    plot_treatment_funnel,
    plot_pre_post_coverage,
    plot_outcome_variation,
    plot_coverage_summary,
    plot_observation_heatmap,
    # Geographic
    plot_location_events,
)

# Event-time plot with CI band
plot_event_time(df_panel, outcome="revenue", event_time_col="event_time")

# Three-panel diagnostic: trend + treatment effect + pre/post comparison
plot_by_treatment_status(df_panel, outcome="revenue")

# Grid of multiple outcomes
plot_multi_outcome(df_panel, outcomes=["revenue", "profit", "employment"])

# Stacked cohort: pooled event study + cohort heatmap
plot_stacked_cohort(df_stacked, outcome="revenue")

# Treatment distribution histogram
plot_treatment_distribution(df_panel)

# Treatment rate + event count over time
plot_treatment_summary(df_panel, unit_col="firm_id", time_col="year")

# Data pipeline funnel
plot_treatment_funnel([
    ("Raw data", 100000),
    ("Filtered", 50000),
    ("Final sample", 20000),
])

# Scatter map of event locations (works with any lat/lon worldwide)
import geopandas as gpd
boundaries = gpd.read_file("states.shp")  # optional basemap
plot_location_events(
    df_locations,
    lat_col="latitude",
    lon_col="longitude",
    time_periods=[2008, 2012, 2016, 2020],
    boundaries=boundaries,
)
```

## Export

```python
from did_panel_builder.io import to_parquet, to_csv, to_stata

to_parquet(df_panel, "panel.parquet")  # drops list columns
to_csv(df_panel, "panel.csv")          # converts list cols to pipe-separated
to_stata(df_panel, "panel.dta")        # drops unsupported types
```

## Input Requirements

Your data needs three columns (names are configurable via `PanelConfig`):

| Column | Type | Description |
|---|---|---|
| Unit ID | str/int | Unique identifier for each unit (firm, county, individual) |
| Time | int/float | Time period (year, quarter, month) |
| Event | bool/0-1 | Whether a treatment event occurred in this unit-period |

The panel should have one row per unit-period. The library handles the rest: computing first-event timing, classifying treatment status, building relative-time variables, and constructing the appropriate panel structure for your chosen estimator.

## Development

```bash
git clone https://github.com/thomaspernet/did-panel-builder.git
cd did-panel-builder
uv venv && uv pip install -e ".[dev]"
uv run pytest
uv run pytest --cov=did_panel_builder  # 95% coverage
uv run ruff check src/
```
