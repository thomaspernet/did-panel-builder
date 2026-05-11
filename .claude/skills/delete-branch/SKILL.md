---
description: "Safely delete the feature branch for issue #$ARGUMENTS after verifying the PR is merged or issue is closed."
---

Safely delete the feature branch for a completed issue.

## Parse arguments

Extract issue number and optional run ID from `$ARGUMENTS`:
- `$ARGUMENTS` = `"42"` → ISSUE=42, RUN_ID=(none)
- `$ARGUMENTS` = `"42 --run 7"` → ISSUE=42, RUN_ID=7

If no issue number is provided, detect from current branch:

```bash
git branch --show-current
```

Extract issue number from branch name (e.g., `fix/42-broken` → ISSUE=42).

## Detect repo

Determine the target repository from the current working directory:

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
```

Pass `--repo "$REPO"` to every `devwatch` command to ensure the correct repo is targeted.

## Intelligence (what you decide)

1. Check issue history for context (run `devwatch issue-history --help` for all options):
   ```bash
   devwatch --repo "$REPO" issue-history <ISSUE>
   ```
   This shows the branch name, PR status, and whether cleanup has already been attempted.

2. Verify the issue is done:
   ```bash
   gh issue view <ISSUE> --repo "$REPO" --json state
   ```
   The issue must be **closed**, or must have a **merged PR** in the agent runs.

3. Find the branch — check agent runs in the DB, then fall back to git:
   ```bash
   git branch -a --list "*fix/<ISSUE>-*" "*feat/<ISSUE>-*" "*refactor/<ISSUE>-*" "*chore/<ISSUE>-*" "*docs/<ISSUE>-*" "*ci-fix/<ISSUE>-*"
   ```
   Call the result `BRANCH`.

4. Validate safety:
   - `BRANCH` must not be one of the repo's pipeline branches (dev / staging / prod from `config.yaml`) or a universally protected name (`main`, `master`, `develop`)

5. **Resolve checkout target.** The CLI refuses to delete the currently-checked-out branch, so the agent must move HEAD itself before invoking deletion — and on epic children that means the epic integration branch, not the dev branch.

   ```bash
   WORKFLOW_JSON=$(devwatch --repo "$REPO" workflow-get --issue <ISSUE>)
   ```

   - If `WORKFLOW_JSON` is non-null and `.strategy == "epic_integration"`, set `TARGET` to the workflow's `canonical_branch` — the `epic/<root>-<slug>` integration branch the child landed on.
   - Otherwise, resolve the configured dev branch:

     ```bash
     TARGET=$(devwatch --repo "$REPO" branches dev)
     ```

     If the repo has no `dev` configured (e.g. release-only repos), fall back to `branches prod`. Do **not** hardcode a branch name (e.g. `dev`) — every repo resolves its own pipeline branches through `config.yaml`.

6. **Switch HEAD if it is on `BRANCH`.** Compare `git branch --show-current` to `BRANCH`; if they match, run:

   ```bash
   git checkout "$TARGET"
   ```

   If HEAD is already on a different branch, leave it alone — the agent may be mid-task on something unrelated.

7. If all checks pass, proceed with deletion.

## Execution

If a RUN_ID was provided, pass `--run` only (issue is derived from the run):

```bash
uv run devwatch --repo "$REPO" delete-branch --run <RUN_ID>
```

If no RUN_ID, pass `--issue` explicitly:

```bash
uv run devwatch --repo "$REPO" delete-branch --issue <ISSUE>
```

The CLI command handles all validation and deletion. If it fails, report the error.

## Boundary

This command deletes branches only. It does not close issues, merge PRs, or modify code.
