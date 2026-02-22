# did-panel-builder

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
df_clean = panel.filter_sample(min_pre_periods=2, min_post_periods=1)

# Merge outcomes from another DataFrame
df_outcomes = pd.read_csv("outcomes.csv")
df_merged = panel.merge_outcomes(df_outcomes, outcome_cols=["gdp", "population"])
```

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

# Filter to usable sample
df_clean = panel.filter_sample(
    keep_treatment_types=["treated", "never_treated"],
    min_pre_periods=2,
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
from did_panel_builder.diagnostics import VariationAnalyzer, CoverageAnalyzer

# Within-unit outcome variation (needed for FE estimation)
var = VariationAnalyzer(config=config)
stats = var.analyze(df_panel, outcome="revenue")
usable = var.usable_sample(df_panel, outcome="revenue")

# Panel coverage: gaps, consecutive periods, balance
cov = CoverageAnalyzer(config=config)
coverage = cov.compute(df_panel)
summary = cov.summary(df_panel)
```

## Visualization

All visualization functions accept an optional `ax` parameter for composability.

```python
from did_panel_builder.visualization import (
    plot_event_time,
    plot_by_treatment_status,
    plot_multi_outcome,
    plot_stacked_cohort,
    plot_treatment_distribution,
    plot_pre_post_coverage,
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
plot_treatment_distribution(df_panel, treatment_time_col="first_event_time")
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
uv run ruff check src/
```
