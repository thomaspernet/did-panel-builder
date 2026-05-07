---
mandatory_for:
  skills: [check-code-quality]
---

# Python Testing

## pytest Ecosystem

Use pytest as the test runner. Key plugins:

- **pytest-asyncio** with `asyncio_mode = "auto"` -- eliminates manual `@pytest.mark.asyncio` on every test.
- **pytest-cov** for coverage reporting.
- **pytest-mock** for the `mocker` fixture (thin wrapper around `unittest.mock`).

Minimal `pyproject.toml` config:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
markers = [
    "slow: takes > 5s",
    "integration: requires external services",
    "unit: fast, isolated",
]

[tool.coverage.run]
source = ["src/mypackage"]
omit = ["*/abc_*.py", "*/interfaces/*"]

[tool.coverage.report]
fail_under = 80
show_missing = true
```

## Async Test Patterns

With `asyncio_mode = "auto"`, any `async def test_*` runs automatically:

```python
async def test_fetch_user_returns_none_when_missing(user_service):
    result = await user_service.get_by_id("nonexistent")
    assert result is None
```

Use `AsyncMock` for async callables, `MagicMock` for sync objects:

```python
from unittest.mock import AsyncMock, MagicMock

repo = MagicMock()
repo.find_by_id = AsyncMock(return_value=None)

result = await service.get(repo, "abc")
repo.find_by_id.assert_awaited_once_with("abc")
```

Patch at the import site, not the definition site:

```python
# module: app/services/user.py imports send_email from app.email
# Correct -- patch where it's looked up:
with patch("app.services.user.send_email") as mock_send:
    ...

# Wrong -- patching the original module has no effect:
with patch("app.email.send_email") as mock_send:
    ...
```

## Mock Strategy

Use context manager syntax for clean scope:

```python
async def test_create_user_sends_welcome_email():
    mock_repo = AsyncMock()
    mock_repo.save.return_value = User(id="1", name="Alice")

    with patch("app.services.user.send_email") as mock_send:
        result = await create_user(mock_repo, name="Alice")

    mock_send.assert_called_once_with("Alice")
    assert result.id == "1"
```

Multiple patches without nesting:

```python
async def test_sync_fetches_and_stores():
    with (
        patch("app.services.sync.fetch_remote", new_callable=AsyncMock) as mock_fetch,
        patch("app.services.sync.store_local", new_callable=AsyncMock) as mock_store,
    ):
        mock_fetch.return_value = [{"id": "1"}]
        await sync_service.run()

    mock_fetch.assert_awaited_once()
    mock_store.assert_awaited_once_with([{"id": "1"}])
```

Set up return values before calling the code under test. Verify calls with `assert_called_once_with()` for sync and `assert_awaited_with()` for async.

## Test Markers

```python
import pytest

@pytest.mark.slow
async def test_full_pipeline_processes_large_file():
    ...

@pytest.mark.integration
async def test_database_round_trip():
    ...

@pytest.mark.unit
def test_parse_date_handles_iso_format():
    ...
```

Register all markers in `pyproject.toml` (shown above) to avoid warnings. Run subsets:

```bash
pytest -m unit           # fast feedback loop
pytest -m "not slow"     # skip slow tests
pytest -m integration    # only integration tests
```

## Coverage Configuration

Target specific source modules, not the entire environment:

```bash
pytest --cov=src/mypackage --cov-report=html --cov-report=term-missing
```

Omit abstract base classes -- their method bodies never execute. Set `fail_under` in `pyproject.toml` so CI catches regressions.

## Test Helpers

Factory functions with `**overrides` for test data:

```python
def make_user(**overrides) -> User:
    defaults = {
        "id": "user-1",
        "name": "Test User",
        "email": "test@example.com",
        "role": "member",
    }
    return User(**(defaults | overrides))

def test_admin_can_delete():
    admin = make_user(role="admin")
    assert admin.can_delete()
```

Reusable fixtures in `conftest.py`:

```python
# tests/conftest.py -- available to all tests
@pytest.fixture
def mock_db():
    db = MagicMock()
    db.execute = AsyncMock(return_value=[])
    return db

# tests/unit/conftest.py -- only for unit tests
@pytest.fixture
def user_service(mock_db):
    return UserService(db=mock_db)
```

## Test Organization

Mirror the source tree:

```
src/mypackage/services/user.py
src/mypackage/services/billing.py

tests/unit/services/test_user.py
tests/unit/services/test_billing.py
tests/integration/test_database.py
```

Group related tests in classes:

```python
class TestCreateUser:
    async def test_returns_created_user(self, user_service):
        result = await user_service.create(name="Alice")
        assert result.name == "Alice"

    async def test_raises_on_duplicate_email(self, user_service, mock_db):
        mock_db.execute.side_effect = DuplicateError("email exists")
        with pytest.raises(DuplicateError):
            await user_service.create(name="Alice")

    async def test_sends_welcome_email(self, user_service):
        ...
```

Use descriptive names: `test_returns_none_when_not_found`, not `test_get_user_3`.

## What NOT to Test

- Framework behavior (ORM saves, HTTP routing, serialization). The framework authors already tested this.
- Trivial getters and setters with no logic.
- Private methods directly. Test them through the public API that calls them.
- Everything in isolation. Integration tests catch what unit tests miss -- wiring errors, serialization mismatches, transaction boundaries.
