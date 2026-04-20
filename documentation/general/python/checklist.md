# Python Completion Checklist

Run the [general checklist](../checklist) first, then this one.

---

## Typing

- [ ] All public functions have full type annotations — parameters and return type. → [Strict Typing](typing#strict-type-checking)
- [ ] Union syntax: `X | None`, not `Optional[X]`. Built-in generics: `list[T]`, not `List[T]`. → [Modern Type Syntax](typing#modern-type-syntax-python-310)
- [ ] DataFrame inputs/returns typed as `pd.DataFrame`, never `Any`. → [No Any Abuse](typing#no-any-abuse)
- [ ] Frozen dataclasses used for config / result objects. → [Dataclasses for Internal Data](typing#dataclasses-for-internal-data)

## Patterns

- [ ] Error handling uses `ValueError` for bad input and `RuntimeError` for unexpected internal failures. No custom exception hierarchies unless genuinely needed. → [Error Handling](patterns#error-handling)
- [ ] Exception chaining: `raise NewError(...) from original`. Never re-raise without `from`. → [Error Handling](patterns#error-handling)
- [ ] Warnings (`warnings.warn`) used for recoverable issues — e.g., missing outcome column — rather than silent skips. → [Error Handling](patterns#error-handling)
- [ ] Guard clauses at the top of functions. Fail fast, early returns, no deep nesting. → [Guard Clauses](patterns#guard-clauses--early-returns)
- [ ] No hardcoded thresholds / window widths / fill values. Expose them as parameters with sensible defaults. → [No Hardcoded Values](patterns#no-hardcoded-values)
- [ ] Checked for existing helpers before writing new ones. → [Before Writing a New Function](patterns#before-writing-a-new-function-check-what-exists)

## DataFrame Conventions

- [ ] Public methods return a new DataFrame. Input DataFrames are never mutated in place. → [DataFrame Conventions](patterns#dataframe-conventions)
- [ ] Vectorised pandas (`groupby`, `merge`, boolean masks, `.assign()`) used instead of `iterrows` in production code.
- [ ] Optional extras (`matplotlib`, `seaborn`, `geopandas`, `pyarrow`) imported inside functions, not at module top.
- [ ] Column names written/read across submodules are defined as constants, not repeated as string literals.

## Conventions

- [ ] Naming: PascalCase classes, snake_case functions, UPPER_SNAKE_CASE constants. → [Naming](conventions#naming)
- [ ] Absolute imports only. No relative imports. → [Import Organization](conventions#import-organization)
- [ ] `__init__.py` exports are explicit (`__all__` or explicit re-exports). No star imports. → [\_\_init\_\_.py Pattern](conventions#__init__py-pattern)
- [ ] `ruff check .` passes.

## Observability

- [ ] No silent exception swallowing. Every `except` block either logs, re-raises, or has a comment explaining why it is safe to ignore.
- [ ] Warning messages include enough context to debug (column name, unit identifier, parameter name).

## Tests

- [ ] New behaviour has a test under the matching `tests/test_<submodule>/` directory. → [Test Organisation](testing#test-organisation)
- [ ] Test names describe the behaviour, not the implementation (`test_filter_sample_trims_event_window_between_bounds`). → [Test Naming](testing#test-naming)
- [ ] DataFrame fixtures are small and hand-crafted for readability.
- [ ] Tests that require `matplotlib` / `geopandas` are marked and skippable in the base install. → [Markers for Optional Extras](testing#markers-for-optional-extras)
- [ ] `uv run pytest` passes locally before pushing.
