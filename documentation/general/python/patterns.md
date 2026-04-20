# Python Patterns

Reusable patterns for a sync, data-transformation library. Framework-agnostic.

---

## Error Handling

Minimal exception types. No deep custom hierarchies unless genuinely needed.

| Exception | Meaning |
|-----------|---------|
| `ValueError` | Bad input — unknown column, malformed config, invalid parameter combination |
| `RuntimeError` | Unexpected internal failure |

Warnings over silent skips: when input data is legitimate but incomplete (e.g., missing outcome column), `warnings.warn(...)` and continue with the observable subset rather than silently dropping rows.

### Exception chaining

Always preserve the original cause:

```python
try:
    data = parse_config(raw)
except KeyError as e:
    raise ValueError(f"Invalid panel config: missing {e.args[0]}") from e
```

---

## Guard Clauses / Early Returns

Validate at the top, fail fast. Flatten control flow with early returns.

```python
# Bad — deeply nested
def filter_sample(self, df, min_event_time, max_event_time):
    if "event_time" in df.columns:
        if min_event_time is not None:
            if max_event_time is not None:
                return df[df["event_time"].between(min_event_time, max_event_time)]
            else:
                raise ValueError("max_event_time required")
        else:
            raise ValueError("min_event_time required")
    else:
        raise ValueError("event_time column missing")

# Good — flat with early returns
def filter_sample(self, df, min_event_time, max_event_time):
    if "event_time" not in df.columns:
        raise ValueError("event_time column missing; call .build() first")
    if min_event_time is None or max_event_time is None:
        raise ValueError("min_event_time and max_event_time are both required")
    return df[df["event_time"].between(min_event_time, max_event_time)]
```

Short-circuit returns for optional work:

```python
def add_variation_flag(self, df, outcome):
    if outcome not in df.columns:
        warnings.warn(f"{outcome} not in DataFrame; skipping variation flag")
        return df
    return df.assign(**{f"has_variation_{outcome}": self._compute_variation(df, outcome)})
```

---

## Composition Over Inheritance

Prefer composition for reusable helpers. Panel builders here already compose: each has a `config: PanelConfig` and delegates to the `treatment` module for assignment logic rather than subclassing.

```python
# Good — composition (has-a)
class StaggeredPanel:
    def __init__(self, df: pd.DataFrame, config: PanelConfig) -> None:
        self._df = df
        self._config = config
        self._assigner = TreatmentAssigner(df, config=config)

    def build(self) -> pd.DataFrame:
        df_treat = self._assigner.build()
        return self._attach_event_time(df_treat)
```

Avoid multi-level class hierarchies. If two builders share logic, extract a module-level helper, not a base class.

---

## Design Patterns

### Factory — @classmethod for object creation

Useful for builders that accept many variants of input:

```python
class PanelConfig:
    def __init__(self, unit_col: str, time_col: str, event_col: str) -> None:
        self.unit_col = unit_col
        self.time_col = time_col
        self.event_col = event_col

    @classmethod
    def from_dict(cls, data: dict) -> "PanelConfig":
        return cls(
            unit_col=data["unit_col"],
            time_col=data["time_col"],
            event_col=data["event_col"],
        )
```

### Strategy — interchangeable algorithms

When multiple implementations of "the same shape" exist, separate them behind a common call signature rather than a giant `if` ladder. The three panel builders (`StaggeredPanel`, `MultiEventPanel`, `StackedPanel`) already follow this: same `.build()` contract, different internal logic.

---

## Modern Python Syntax

### match/case for complex branching

```python
def classify_treatment(first_event_time: int | None, t: int, study_end: int) -> str:
    match first_event_time:
        case None:
            return "never_treated"
        case fet if t < fet:
            return "not_yet_treated"
        case fet if t >= fet:
            return "treated"
```

### Walrus operator for assignment in conditionals

```python
if (event_col := self._config.event_col) not in df.columns:
    raise ValueError(f"event column {event_col!r} missing from DataFrame")
```

### Comprehensions

```python
# Transform
active_units = [uid for uid, row in df.iterrows() if row["treated"]]

# Aggregate
treated_count = sum(1 for _, row in df.iterrows() if row["treated"])
```

For DataFrame work, prefer vectorised pandas over row-level comprehensions whenever possible (`df.loc[df["treated"], "unit"].unique()` instead of the iterrows version).

---

## No Hardcoded Values

Thresholds, window widths, period counts, fill values — accept them as parameters with sensible defaults. Code defines the mechanism; callers decide the policy.

```python
# Bad — hardcoded
def filter_sample(self, df):
    return df[df["event_time"].between(-3, 5)]

# Good — parameterised
def filter_sample(
    self,
    df: pd.DataFrame,
    min_event_time: int = -3,
    max_event_time: int = 5,
    min_pre_periods: int = 2,
    min_post_periods: int = 1,
) -> pd.DataFrame:
    ...
```

The test: if a user wants a different estimation window, can they change it without editing the library? If yes, it's decoupled.

---

## Function Design

Keep functions short (5-50 lines). One responsibility per function.

### Before writing a new function, check what exists

Do not create a new helper if one already exists. Before writing a utility:

1. Check the current submodule (`panels/`, `treatment/`, `diagnostics/`, `io/`, `visualization/`).
2. Check `_types.py` for existing type aliases / constants.
3. Grep the codebase for the operation you need.

If similar logic exists in another module, extract it into a shared location rather than duplicating it.

### Where helpers belong

| Scope | Location | Example |
|---|---|---|
| Used within one class | Private method (`_attach_event_time`) | Internal step of one builder |
| Used within one module | Module-level function | Shared helpers within `panels/` |
| Used across submodules | Shared utility in the package root (`_utils.py`) | Column-existence checks, config validation |

Do not scatter helpers across random files.

### Extract private helpers

```python
class StaggeredPanel:
    def build(self) -> pd.DataFrame:
        df = self._validate_columns(self._df)
        df = self._attach_first_event_time(df)
        df = self._attach_event_time(df)
        return self._classify_treatment(df)

    def _validate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in (self._config.unit_col, self._config.time_col, self._config.event_col)
                   if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        return df
```

### Properties for computed attributes

```python
class PanelConfig:
    @property
    def required_columns(self) -> tuple[str, ...]:
        return (self.unit_col, self.time_col, self.event_col)
```

### @staticmethod for stateless utilities

```python
class StaggeredPanel:
    @staticmethod
    def _fill_value_for_first_event_time(series: pd.Series) -> int:
        return int(series.max()) + 1
```

---

## DataFrame Conventions

Specific to a pandas-centric library — these are the patterns that matter most here.

### Don't mutate input DataFrames

Every public method returns a new DataFrame. Use `.copy()`, `.assign()`, or new construction. Never `df["new_col"] = ...` on an input argument.

```python
# Good — new DataFrame
def build(self) -> pd.DataFrame:
    df = self._df.copy()
    df = df.assign(event_time=df[self._config.time_col] - df["first_event_time"])
    return df
```

### Vectorise over iterating

Prefer `groupby` / `merge` / boolean masks / `.assign()` over `iterrows`. `iterrows` is acceptable only in tests or one-off diagnostics.

### Lazy imports for optional extras

`matplotlib`, `seaborn`, `geopandas`, `pyarrow` are optional extras. Import them inside functions, not at module top, so the base install stays `pandas + numpy` only:

```python
def plot_event_time(df, outcome, event_time_col):
    import matplotlib.pyplot as plt  # lazy — only imported when plotting
    ...
```

### Column name constants

Column names that are written by panel builders and read by other modules (`event_time`, `first_event_time`, `treatment_type`, `treatment_status`) belong in `_types.py` or a module-level constant, so renames are localised.
