# Python Typing

## Strict Type Checking

Every public function must be fully annotated — parameters and return type. No exceptions for code users call.

Use `from __future__ import annotations` at the top of each module. This enables PEP 563 deferred evaluation, so forward references to types like `pd.DataFrame` and `PanelConfig` work without import-order pain.

```python
from __future__ import annotations

import pandas as pd


def build(df: pd.DataFrame, config: PanelConfig) -> pd.DataFrame:
    ...
```

Optional MyPy config (not enforced in this repo yet, but recommended if added):

```toml
[tool.mypy]
python_version = "3.10"
strict = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
no_implicit_optional = true
warn_return_any = true

[[tool.mypy.overrides]]
module = ["geopandas.*", "pyarrow.*", "seaborn.*"]
ignore_missing_imports = true
```

## Modern Type Syntax (Python 3.10+)

Use `X | None` instead of `Optional[X]`. Use built-in generics instead of `typing` wrappers.

```python
# Good
def first_event_time(df: pd.DataFrame, unit_col: str) -> dict[str, int | None]:
    ...

def event_times(events: list[int]) -> list[int]:
    ...

# Bad — legacy syntax
from typing import Optional, List, Dict

def first_event_time(df: pd.DataFrame, unit_col: str) -> Dict[str, Optional[int]]:
    ...
```

Use `collections.abc` for abstract container types:

```python
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
```

Use `Literal` for constrained string parameters — many DiD knobs are enums:

```python
from typing import Literal

TreatmentType = Literal["treated", "not_yet_treated", "never_treated"]
ControlType = Literal["never_treated", "not_yet_treated"]

def filter_sample(
    df: pd.DataFrame,
    keep_treatment_types: list[TreatmentType] | None = None,
) -> pd.DataFrame:
    ...
```

Use `TypeVar` and `Generic` for reusable containers when you actually have multiple element types. For single-type helpers, don't bother.

## Dataclasses for Internal Data

Use `@dataclass(frozen=True)` for immutable value objects like configuration and output schemas.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class PanelConfig:
    unit_col: str
    time_col: str
    event_col: str


@dataclass(frozen=True)
class PrePostResult:
    treated_pre_mean: float
    treated_post_mean: float
    control_pre_mean: float
    control_post_mean: float
```

Dataclasses are the right tool here — no Pydantic, no schema layer, no serialization concerns.

## TYPE_CHECKING Pattern

Import heavy or optional dependencies inside `if TYPE_CHECKING:` so they don't get imported at module load time. Especially useful for the optional extras (`matplotlib`, `geopandas`):

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.figure
    import geopandas as gpd


def plot_location_events(
    df,
    boundaries: gpd.GeoDataFrame | None = None,
) -> matplotlib.figure.Figure:
    import matplotlib.pyplot as plt  # actual import happens only when called
    ...
```

## No `Any` Abuse

`Any` is acceptable for genuinely heterogeneous data:

```python
# Acceptable — JSON-like config with varied types
def from_dict(data: dict[str, Any]) -> PanelConfig:
    ...
```

`Any` is never acceptable in these positions:

```python
# Bad — return type hides the actual contract
def build() -> Any:
    ...

# Bad — parameter type when a specific type exists
def filter_sample(df: Any) -> pd.DataFrame:
    ...

# Bad — generic fallback out of laziness
results: list[Any] = []
```

When you don't know the exact type, use `object` for "anything" or define a `Protocol` for the expected interface. For DataFrame inputs, always use `pd.DataFrame` — never `Any`.
