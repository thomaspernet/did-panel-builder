# Product Overview

## What It Does

`did-panel-builder` turns a raw unit-time-event DataFrame into an analysis-ready panel for difference-in-differences (DiD) estimation. It computes first-event timing, classifies units (treated / not-yet-treated / never-treated), builds relative-time variables, and prepares the panel shape each modern DiD estimator expects — plus a diagnostics and visualization layer for sample coverage, within-unit variation, and pre/post balance.

## Who It's For

Empirical researchers — economists, policy analysts, quantitative social scientists — who run DiD designs in Python and want the panel-construction step to be explicit, tested, and reproducible rather than a one-off notebook script.

Background assumed: comfortable with pandas and with the DiD literature (Sun & Abraham, de Chaisemartin & D'Haultfoeuille, Roth et al.). No software-engineering background required.

## Key Concepts

- **Panel builder** — a class that takes a raw DataFrame and returns a panel shaped for one DiD design.
  - `StaggeredPanel` — first-treatment timing only (Sun & Abraham 2021).
  - `MultiEventPanel` — multiple treatments per unit (de Chaisemartin & D'Haultfoeuille 2020).
  - `StackedPanel` — stacked cohort 2×2 experiments (Roth et al. 2022).
- **`TreatmentAssigner`** — a pre-step that classifies units, counts pre/post periods, and flags units that fail coverage rules before the panel is built.
- **Event time** — periods measured relative to each unit's first treatment (0 = treatment period; −3 = three periods before).
- **Treatment type** — `treated`, `not_yet_treated`, or `never_treated` in a given panel row.
- **`PanelConfig`** — the three column roles (unit, time, event) the whole library is parameterized on.

## Value

- One consistent API across the three main modern DiD designs — swap estimators without rewriting your panel-construction code.
- Diagnostics (`CoverageAnalyzer`, `PrePostDiagnostics`, `VariationAnalyzer`) surface sample problems — gaps, insufficient within-unit variation, imbalanced pre/post — before they silently distort estimates.
- Optional extras (`[viz]`, `[geo]`, `[io]`) keep the core dependency set to pandas + numpy, so the library runs in constrained environments and the visualization / geographic / parquet layers are opt-in.
- Tested (151 tests, 95% coverage) and typed, so the panel the library hands back is the panel you think it is.
