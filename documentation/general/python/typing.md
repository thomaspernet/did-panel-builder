---
mandatory_for:
  skills: [fix-issue, feat-issue]
  rules: [backend]
---

# Python Typing

## Strict Type Checking

Enable MyPy strict mode in `pyproject.toml`:

```toml
[tool.mypy]
python_version = "3.12"
strict = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
no_implicit_optional = true
warn_return_any = true
strict_equality = true
plugins = ["pydantic.mypy"]

[[tool.mypy.overrides]]
module = ["some_untyped_library.*"]
ignore_missing_imports = true
```

Every function must be fully annotated -- parameters and return type. No exceptions.

Use `from __future__ import annotations` at the top of every module. This enables PEP 563 deferred evaluation, allowing forward references and cleaner syntax without runtime overhead.

```python
from __future__ import annotations


def process(items: list[Item]) -> Result:
    ...
```

## Modern Type Syntax (Python 3.10+)

Use `X | None` instead of `Optional[X]`. Use built-in generics instead of `typing` wrappers.

```python
# Good
def find_user(user_id: str) -> User | None:
    ...

def get_scores() -> dict[str, float]:
    ...

def get_names() -> list[str]:
    ...

# Bad -- legacy syntax
from typing import Optional, List, Dict

def find_user(user_id: str) -> Optional[User]:
    ...
```

Use `collections.abc` for abstract container types:

```python
from collections.abc import Callable, AsyncIterator, AsyncGenerator, Awaitable

def retry(fn: Callable[[], Awaitable[T]]) -> T:
    ...

async def stream_events() -> AsyncIterator[Event]:
    ...
```

Use `Literal` for constrained string parameters:

```python
from typing import Literal

def set_log_level(level: Literal["debug", "info", "warning", "error"]) -> None:
    ...
```

Use `TypeVar` and `Generic` for reusable containers:

```python
from typing import TypeVar, Generic

T = TypeVar("T")

class Repository(Generic[T]):
    def get(self, entity_id: str) -> T | None:
        ...

    def save(self, entity: T) -> T:
        ...
```

## Pydantic Models

All data crossing layer boundaries must be a Pydantic `BaseModel`. Define separate models per operation to keep validation tight.

```python
from pydantic import BaseModel, Field, model_validator


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)


class ProjectUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = None


class ProjectResponse(BaseModel):
    uuid: str
    name: str
    description: str
    created_at: str
```

Cross-field validation with `model_validator`:

```python
class DateRange(BaseModel):
    start: date
    end: date

    @model_validator(mode="after")
    def validate_range(self) -> DateRange:
        if self.end < self.start:
            msg = "end must be after start"
            raise ValueError(msg)
        return self
```

Model inheritance for shared fields:

```python
class TimestampMixin(BaseModel):
    created_at: str
    updated_at: str


class ProjectResponse(TimestampMixin):
    uuid: str
    name: str
```

Generic response wrappers:

```python
from typing import Generic, TypeVar
from pydantic import BaseModel

T = TypeVar("T")

class DataResponse(BaseModel, Generic[T]):
    data: T
    count: int

    model_config = {"populate_by_name": True}
```

Use `populate_by_name = True` when accepting both camelCase (from frontends) and snake_case (from Python callers).

## Dataclasses for Internal Data

Use `@dataclass(frozen=True)` for immutable value objects that do not need Pydantic validation.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Coordinate:
    x: float
    y: float

    @property
    def magnitude(self) -> float:
        return (self.x ** 2 + self.y ** 2) ** 0.5


@dataclass(frozen=True)
class RetryConfig:
    max_attempts: int
    backoff_seconds: float
    timeout_seconds: float
```

Use dataclasses for internal DTOs, configuration objects, and value types where serialization or schema validation is not required.

## TYPE_CHECKING Pattern

Import expensive or circular dependencies inside `if TYPE_CHECKING:` to keep them available at type-check time but absent at runtime.

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from myapp.services.heavy_service import HeavyService
    from myapp.domain.models import ComplexModel


class Controller:
    def __init__(self, service: HeavyService) -> None:
        self.service = service

    def process(self, model: ComplexModel) -> str:
        return self.service.run(model)
```

This pattern breaks circular imports cleanly. No lazy import hacks, no runtime `importlib` gymnastics.

## No `Any` Abuse

`Any` is acceptable for genuinely heterogeneous data:

```python
# Acceptable -- JSON payload with unknown structure
def store_metadata(metadata: dict[str, Any]) -> None:
    ...
```

`Any` is never acceptable in these positions:

```python
# Bad -- return type hides the actual contract
def get_user() -> Any:
    ...

# Bad -- parameter type when a specific type exists
def process(data: Any) -> None:
    ...

# Bad -- generic fallback out of laziness
items: list[Any] = []
```

When you do not know the exact type, use `object` for "anything" or define a `Protocol` / `TypeVar` for the expected interface.

```python
from typing import Protocol


class Renderable(Protocol):
    def render(self) -> str:
        ...


def display(item: Renderable) -> None:
    print(item.render())
```
