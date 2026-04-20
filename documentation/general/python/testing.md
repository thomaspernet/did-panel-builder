# Python Testing

## pytest Ecosystem

Use pytest as the test runner. Useful plugins for this project:

- **pytest-cov** for coverage reporting.
- **pytest-mock** for the `mocker` fixture when you need to stub out optional deps.

Minimal `pyproject.toml` config:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "slow: takes > 5s",
    "viz: requires matplotlib / seaborn (optional extra)",
    "geo: requires geopandas (optional extra)",
]

[tool.coverage.run]
source = ["src/did_panel_builder"]

[tool.coverage.report]
fail_under = 90
show_missing = true
```

## Test Organisation

Tests mirror the source tree:

```
src/did_panel_builder/panels/staggered.py
src/did_panel_builder/treatment/assigner.py
src/did_panel_builder/diagnostics/pre_post.py

tests/test_panels/test_staggered.py
tests/test_treatment/test_assigner.py
tests/test_diagnostics/test_pre_post.py
```

Shared DataFrame fixtures live in `tests/conftest.py`. Prefer small, hand-crafted DataFrames over generated data — they make assertion failures trivial to read.

## DataFrame Fixtures

Factory functions with `**overrides` keep tests concise:

```python
# tests/conftest.py
import pandas as pd
import pytest


@pytest.fixture
def staggered_df() -> pd.DataFrame:
    """Three units, years 2010–2015, two treated in 2012 and 2014."""
    return pd.DataFrame({
        "firm_id": ["A"] * 6 + ["B"] * 6 + ["C"] * 6,
        "year":    list(range(2010, 2016)) * 3,
        "has_shock": (
            [0, 0, 1, 0, 0, 0] +   # A treated in 2012
            [0, 0, 0, 0, 1, 0] +   # B treated in 2014
            [0] * 6                 # C never treated
        ),
    })


def make_panel_row(**overrides) -> dict:
    defaults = {
        "firm_id": "A",
        "year": 2010,
        "has_shock": 0,
        "revenue": 100.0,
    }
    return defaults | overrides
```

## Test Naming

Descriptive names that state the behaviour being tested:

```python
def test_staggered_panel_assigns_never_treated_when_no_events(staggered_df):
    ...

def test_filter_sample_trims_event_window_between_bounds(staggered_df):
    ...

def test_merge_outcomes_warns_when_outcome_column_missing(staggered_df, recwarn):
    ...
```

Not `test_build_1`, `test_build_2`.

## Grouping Tests

Group related tests in a class when they share setup or conceptually belong together:

```python
class TestStaggeredBuild:
    def test_adds_event_time_column(self, staggered_df):
        panel = StaggeredPanel(staggered_df, config=config)
        df = panel.build()
        assert "event_time" in df.columns

    def test_never_treated_event_time_uses_fill_value(self, staggered_df):
        ...

    def test_raises_when_unit_col_missing(self):
        with pytest.raises(ValueError, match="missing"):
            StaggeredPanel(pd.DataFrame({"year": [2010]}), config=config).build()
```

## Coverage

Target the package, not the environment:

```bash
uv run pytest --cov=did_panel_builder --cov-report=term-missing
```

Set `fail_under` in `pyproject.toml` so regressions are caught in CI.

## Markers for Optional Extras

Tests that need `matplotlib` / `geopandas` must be skippable in the base install:

```python
import pytest

matplotlib = pytest.importorskip("matplotlib")


@pytest.mark.viz
def test_plot_event_time_returns_figure(panel_df):
    from did_panel_builder.visualization import plot_event_time
    fig = plot_event_time(panel_df, outcome="revenue", event_time_col="event_time")
    assert fig is not None
```

Run subsets:

```bash
uv run pytest -m "not viz and not geo"   # core only
uv run pytest -m viz                     # only plot tests
```

## What NOT to Test

- Pandas / NumPy internals (groupby semantics, merge correctness). The library authors tested those.
- Trivial property accessors with no logic.
- Private helpers directly — test them through the public API that calls them.
- Everything with mocks. Real DataFrames with a handful of rows are almost always clearer than a mock pandas object.
