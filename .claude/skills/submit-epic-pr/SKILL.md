---
description: "Open the epic-level PR for an epic-rooted workflow."
---

Open the epic PR for an epic-rooted workflow (#1884). Wraps the same `gh pr create` + closes-trailer body + `workflows.pr_number` write that used to run in-process — moved into a skill so the dashboard's inline terminal opens like every other workflow / step action.

The readiness gate has already run on the server before this skill launched; if you got here, every child is closed on GitHub and present on the epic / shared branch. Do not re-prompt the user.

## Mandatory reads — do this first

Run:

    devwatch --repo "$REPO" doc-read --skill submit-epic-pr --display

The output contains every doc you must read; treat it as if you opened each file directly. Do not proceed with the skill body until done.

## Parse arguments

`$ARGUMENTS` shape:

```
<epic> --workflow-id <id> [--run <run_id>]
```

Examples:

- `185 --workflow-id 18 --run 42` → open the epic PR for epic #185 on workflow 18, attach to agent-run 42.

Extract `EPIC` (positional, integer), `WORKFLOW_ID` (`--workflow-id <id>`), and `RUN_ID` (`--run <id>`).

If `WORKFLOW_ID` is missing, **stop** — this skill is only for workflow-bound submissions launched from the dashboard. The legacy CLI path (no workflow id) is for direct ad-hoc use, not the dashboard chip.

## Detect repo

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
```

Pass `--repo "$REPO"` to every `devwatch` command.

## Execution

Hand the work to the existing CLI — it dispatches on workflow shape (EPIC_INTEGRATION vs SAME/batch), composes the closes-trailer body, runs `gh pr create`, writes `workflows.pr_number`, and updates the agent run row:

```bash
devwatch --repo "$REPO" submit-epic-pr "$EPIC" \
  --workflow-id "$WORKFLOW_ID" \
  --run-id "$RUN_ID"
```

Omit `--run-id` if no `RUN_ID` was parsed.

The CLI prints the PR URL, the shipped-child list, and the agent-run id. Surface those lines to the user verbatim — they are the only acknowledgement the user gets that the click did something.

### What the CLI does (so you can explain failures)

- **EPIC_INTEGRATION shape**: PR head = `epic/<N>-<slug>` (the epic integration branch), PR base = repo dev branch. Body lists `Closes #<root>` plus `Closes #<child>` per shipped child.
- **SAME/batch shape**: PR head = `workflow.base_branch` (the shared branch every step landed on), PR base = repo dev branch. Body lists `Closes` lines for the root epic and every `done` step.

Both shapes label the PR `epic` and write `workflows.pr_number` so the dashboard's `submit-workflow-pr` chip flips to `done` and **Merge PR** unblocks.

## Boundary

This skill opens the PR. It does not merge, does not run `/release`, and does not file follow-up issues. Tell the user the next step is to wait for CI green, then run `/merge-pr` (or click the workflow chip).
