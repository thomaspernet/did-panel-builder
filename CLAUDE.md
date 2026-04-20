# did-panel-builder

## Project & Tech

Python 3.10+ library for building panel datasets for difference-in-differences estimation (Staggered, MultiEvent, Stacked designs). Dependencies: pandas, numpy. Tooling: `uv`, `hatchling`, `pytest`, `ruff`.

## Top Rules

1. Fix root causes, not symptoms — DiD panel logic is subtle; a warning or silent skip often hides a data bug.
2. No backward-compatibility hacks. If an API is unused, delete it.
3. Every new panel behavior needs a test under `tests/test_panels/` or the relevant sibling.
4. Keep `pandas`/`numpy` as the only required deps. Visualization, geo, and parquet stay behind optional extras.
5. Run `pytest` and `ruff check` before pushing.

## Rules Loading

| Rule file | Loads when | Points to |
|---|---|---|
| `critical.md` | Every conversation | `documentation/general/clean-code.md`, checklists |
| `backend.md` | Reading `**/*.py` | `documentation/general/python/*` |

## Key Directories

- `src/did_panel_builder/panels/` — `StaggeredPanel`, `MultiEventPanel`, `StackedPanel` builders.
- `src/did_panel_builder/treatment/` — treatment assignment / event column helpers.
- `src/did_panel_builder/diagnostics/` — sample and pre/post diagnostics (e.g. `PrePostDiagnostics`).
- `src/did_panel_builder/io/` — panel I/O (CSV, parquet via `[io]` extra).
- `src/did_panel_builder/visualization/` — matplotlib/seaborn plots (gated by `[viz]`/`[geo]` extras).
- `tests/` — pytest suite mirroring the `src/` layout (`test_panels`, `test_treatment`, `test_diagnostics`, `test_io`, `test_visualization`).

## Dev Commands

```bash
# Install with all extras + dev tools
uv sync --extra dev

# Run the test suite
uv run pytest

# Lint
uv run ruff check .

# Format (ruff-compatible)
uv run ruff format .
```

## Reference

| When you need... | Read |
|---|---|
| Clean code principles | `documentation/general/clean-code.md` |
| General checklist | `documentation/general/checklist.md` |
| Documentation principles | `documentation/general/documentation.md` |
| Rules checklist | `documentation/general/rules-checklist.md` |
| Propagation scan | `documentation/general/propagation-scan.md` |
| Issue to rule | `documentation/general/issue-to-rule.md` |
| Python patterns | `documentation/general/python/patterns.md` |
| Python conventions | `documentation/general/python/conventions.md` |
| Python typing | `documentation/general/python/typing.md` |
| Python testing | `documentation/general/python/testing.md` |
| Project architecture | `documentation/project/architecture.md` |
| Project conventions | `documentation/project/conventions.md` |
| Product overview | `documentation/product/overview.md` |
| Product usage | `documentation/product/usage.md` |

## Documentation

This project contains a `documentation/` folder with three top-level areas:

- **`documentation/general/`** — cross-project rules and conventions (clean code, principles, Python guides, checklists). Reusable across any project.
- **`documentation/project/`** — technical / dev-facing docs for *this* codebase: how the code works, architecture, decisions and trade-offs.
- **`documentation/product/`** — what the library is and how to use it. Written for non-technical readers and as usage guides for end users.
