# Python Coding Conventions

General-purpose conventions for Python projects. Use as a reference while writing code.

## Naming

| Element | Convention | Example |
|---|---|---|
| Classes | PascalCase | `UserAccount`, `HttpClient` |
| Functions / methods | snake_case | `get_user()`, `parse_response()` |
| Constants | UPPER_SNAKE_CASE | `MAX_RETRIES`, `DEFAULT_TIMEOUT` |
| Private members | Leading underscore | `_internal_cache`, `_validate()` |
| Files / modules | snake_case.py | `user_service.py`, `http_client.py` |
| Interfaces (ABCs) | I-prefix | `IRepository`, `IConnector`, `IStorage` |
| Type parameters | Single letter or bound | `T`, `K`, `V`, `ItemT = TypeVar("ItemT", bound=BaseModel)` |

```python
class IRepository(ABC):
    """Interface for data access."""

    @abstractmethod
    def find_by_id(self, id: str) -> Entity | None: ...

class UserRepository(IRepository):
    MAX_PAGE_SIZE = 100
    _connection_pool: ConnectionPool
```

## Import Organization

Order: `__future__` -> standard library -> third-party -> first-party. Alphabetical within each group. Blank line between groups.

```python
from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel
from sqlalchemy import select

from mypackage.domain.models import User
from mypackage.infrastructure.db import SessionFactory
```

Rules:

- Absolute imports only. Never `from . import something`.
- Explicit imports: `from module import ClassA, ClassB`. Never `import *`.
- Use `TYPE_CHECKING` blocks for type-only imports to break circular dependencies at runtime.
- Lazy imports inside function bodies as a last resort for circular dependencies.

```python
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mypackage.domain.models import HeavyModel

def process(item_id: str) -> None:
    # HeavyModel used only in type hints above, not at runtime
    ...
```

## Docstrings

Google-style format. Keep them short when the function is self-explanatory.

```python
"""User management utilities."""

class UserService:
    """Handles user lifecycle operations."""

    def create_user(self, name: str, email: str) -> User:
        """Create a new user and send a welcome email.

        Args:
            name: Display name.
            email: Must be unique across the system.

        Returns:
            The created user with a generated ID.
        """
        ...

    def delete_user(self, user_id: str) -> None:
        """Delete a user by ID."""
        ...
```

Guidelines:

- Module-level docstring always first line of the file.
- Class docstring: one-line summary.
- Method docstring: summary + Args + Returns when non-obvious. Single line for simple methods.
- Inline comments for non-obvious logic only.
- `# NOTE:` for code needing future attention.

```python
# NOTE: upstream API returns dates as epoch seconds, not milliseconds
timestamp = raw_value / 1000
```

## String Formatting

| Method | When to use |
|---|---|
| f-strings | Default for interpolation |
| `.format()` | Template strings with named placeholders |
| %-style | Logging calls only (deferred evaluation) |

```python
# f-string (default)
message = f"User {user.name} created at {user.created_at}"

# .format() for reusable templates
TEMPLATE = "Hello {name}, your order {order_id} is ready."
body = TEMPLATE.format(name=user.name, order_id=order.id)

# %-style for logging (arguments evaluated only if level is active)
logger.info("Created user %s with email %s", user.name, user.email)
```

## Logging

Per-module logger. Parameterized messages. No f-strings in log calls.

```python
import logging

logger = logging.getLogger(__name__)

def process_order(order_id: str) -> None:
    logger.info("Processing order %s", order_id)
    try:
        result = _execute(order_id)
        logger.info("Order %s completed in %dms", order_id, result.duration_ms)
    except ValidationError:
        logger.warning("Order %s failed validation, skipping", order_id)
    except Exception:
        logger.error("Order %s failed unexpectedly", order_id, exc_info=True)
```

Level guidelines:

| Level | Use for |
|---|---|
| DEBUG | Detailed diagnostic info (disabled in production) |
| INFO | Normal operations: started, completed, created |
| WARNING | Recoverable issues: retries, fallbacks, deprecations |
| ERROR | Failures requiring attention |

Suppress noisy third-party loggers at startup:

```python
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
```

## Linting and Formatting

Ruff as the single tool for linting and formatting.

```toml
# pyproject.toml
[tool.ruff]
line-length = 88

[tool.ruff.format]
quote-style = "double"

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "F",   # pyflakes
    "W",   # pycodestyle warnings
    "I",   # isort
    "UP",  # pyupgrade
    "B",   # bugbear
]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]  # re-exports
```

Document intentional rule exceptions inline:

```python
# Depends() calls evaluated at import time by the DI framework
app.include_router(router)  # noqa: B008
```

## `__init__.py` Pattern

Explicit `__all__` exports. Minimal re-exports for the public API only.

```python
"""Public API for the domain models package."""

from mypackage.domain.models.user import User
from mypackage.domain.models.order import Order, OrderStatus

__all__ = [
    "Order",
    "OrderStatus",
    "User",
]
```

Rules:

- Never use star imports in `__init__.py`.
- Only re-export symbols that external consumers need.
- Keep `__all__` sorted alphabetically.
- Internal helpers stay in their modules; do not surface them here.
