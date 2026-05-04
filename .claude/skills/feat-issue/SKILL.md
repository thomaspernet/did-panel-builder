---
description: "Read GitHub issue #$ARGUMENTS, create a branch, and implement the feature."
---

Implement a new feature. Read the issue, plan, implement, test, commit, push.

GitHub writing rules: `docs:general/github-writing`.

Read this repo's CLAUDE.md for architecture and rules.

## Parse arguments

Extract issue number, optional run ID, and optional base branch from `$ARGUMENTS`:
- `$ARGUMENTS` = `"42"` -> ISSUE=42, RUN_ID=(none), BASE_BRANCH=(none)
- `$ARGUMENTS` = `"42 --run 7"` -> ISSUE=42, RUN_ID=7, BASE_BRANCH=(none)
- `$ARGUMENTS` = `"42 --run 7 --base-branch feat/364-something"` -> ISSUE=42, RUN_ID=7, BASE_BRANCH=feat/364-something

Use ISSUE for git branch names and GitHub references. Use RUN_ID for all `devwatch` tracking calls. Use BASE_BRANCH (if provided) as the base for the new branch instead of the default dev branch.

## Detect repo

Determine the target repository from the current working directory:

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
```

Pass `--repo "$REPO"` to every `devwatch` command to ensure the correct repo is targeted.

## Context loading

- Read this repo's CLAUDE.md for architecture and rules.
- Read the repo's coding principles docs if they exist (check CLAUDE.md for paths).

## Lineage pre-read

If this issue is a `child-of` another issue (typically: a sub-feature of a
larger epic), pick up the ancestor context — parent feature body, merged PR
description, quality-check reports, sibling bugs — before touching code. For
root issues this is a no-op (the command prints nothing).

```bash
devwatch --repo "$REPO" lineage-context <ISSUE> --format markdown
```

Treat the output as authoritative grounding. When a parent feature's merged
PR description fixes an interface you were about to re-derive from the bug
report, follow the PR — that is what is actually in production.

**Missing parent nudge.** If `lineage-context` returned no ancestors but the
issue body indicates this is a sub-feature — phrases like *"sub-feature of #N"*,
*"part of epic #N"*, *"extends #N"*, or *"child-of #N"* — surface this to the
user once before coding: *"This issue mentions #N and has no `child-of` link.
Link it before I start? Run: `devwatch link <ISSUE> --to N --type child-of`"*
Do not auto-create the link. Proceed if the user says no — weaker mentions
like *"related to #N"* or *"see #N"* are not parent relationships.

## Find the reference

Before writing any code:

1. Read the issue: `gh issue view <ISSUE> --repo "$REPO" --json title,body,labels`
2. Classify the work type (new entity, endpoint, tool, component, page, etc.).
3. Find the closest existing implementation of the same type in the codebase.
4. Read it to understand patterns, conventions, and structure.
5. Do not start coding until you have a reference implementation to follow.

## Issue history (context awareness)

Check the action timeline to understand what has already happened on this issue:

```bash
devwatch --repo "$REPO" issue-history <ISSUE>
```

This shows prior implementation attempts, quality check results, PRs, and branches.

Available arguments (run `devwatch issue-history --help` to see all):
- `--run <RUN_ID>` — full details for a single run (timestamps, summary, files, commits)
- `--phase <phase>` — filter by phase: `impl`, `quality`, `pr`, `docs`, `release`, `cleanup`
- `--full` — expand all runs with full details
- `--comments` — show Lingtai-relevant issue comments (quality reports, fix summaries)

- **If prior runs exist**: this is a re-implementation. Drill into the details:
  - `devwatch --repo "$REPO" issue-history <ISSUE> --phase quality` — see quality check results
  - `devwatch --repo "$REPO" issue-history <ISSUE> --comments` — read quality failure reports and fix summaries
  - Use the failed check items as your primary guide for what to fix
  - Do NOT just re-validate against the original AC — address the specific feedback
- **If no prior runs exist**: this is a first implementation. Proceed normally.

## Workflow detection

Before creating a branch, look up the workflow that owns this issue. Since #1123 / epic #1116, the workflow row is the **source of truth** for `base_branch` and `strategy` — the skill does not re-derive either.

```bash
WORKFLOW_JSON=$(devwatch --repo "$REPO" workflow-get --issue <ISSUE>)
```

Decision matrix based on the response:

1. **`null` or empty** — no workflow.
   - If the issue body contains `child-of: #<epic>` (i.e. it is an epic child), **refuse**. Print a clear error and stop before touching git:

     ```
     Issue #<ISSUE> is a child of epic #<epic> but no workflow contains it.
     Epic children cannot be implemented without a workflow that owns the
     base branch and branching strategy. Create a workflow first:

       - Dashboard: open epic #<epic> and click "Create workflow"
       - CLI:       devwatch --repo "$REPO" workflow-create --root <epic> ...

     Then re-run /feat-issue <ISSUE>.
     ```

     Exit non-zero. Do not run `git fetch`, `git checkout`, or any mutation.
   - If the issue is standalone (no `child-of`), proceed to the **Standalone branch** block below.

2. **Workflow returned** — parse these fields:
   - `WORKFLOW_STRATEGY = workflow.strategy` — one of `"standalone"` or `"epic_integration"`. This decides which ref the child branch is cut off, so read it before either of the two refs below.
   - `WORKFLOW_BRANCH = workflow.canonical_branch` — the canonical implementation branch (for `epic_integration`, this is the **epic integration branch** `epic/<root>-<slug>` derived from contract fields; for `standalone` it falls through to `current_branch`). Do **not** read `workflow.current_branch` directly — it is overwritten by every child's `agent-update --branch`, so under `epic_integration` it returns whichever sibling ran last (#1611).
   - `WORKFLOW_BASE = workflow.base_branch` — the workflow's *parent* branch (typically `dev` / `local-dev-next`). For `standalone` workflows this is the ref children cut off; for `epic_integration` workflows it is the parent the epic was cut from (kept for drift-pull / merge-target context) and is **not** the right ref for children.

## Branch

**If a workflow was returned**, choose the implementation branch based on `WORKFLOW_STRATEGY`:

- **`standalone`** — the workflow has no shared integration branch. Each child cuts its own `feat/<ISSUE>-<slug>` off the workflow's parent ref:
  ```bash
  BASE=$WORKFLOW_BASE
  BRANCH=$(devwatch --repo "$REPO" workflow-branch-name --issue <ISSUE> --prefix feat)
  git fetch origin && git checkout -b "$BRANCH" origin/${BASE}
  ```
- **`epic_integration`** — children land on the shared epic integration branch cut at workflow creation. The child branch must be cut off `WORKFLOW_BRANCH` (the epic integration branch, `workflow.canonical_branch`), **not** `WORKFLOW_BASE` — cutting off `WORKFLOW_BASE` would skip every previously-merged sibling commit and produce a child PR with no shared history with the epic (#1457). Each child still ships on its own short-lived `feat/<childN>-<child-slug>` cut off the epic branch (#1096):
  ```bash
  BRANCH=$(devwatch --repo "$REPO" workflow-branch-name --issue <ISSUE> --prefix feat)
  git fetch origin && git checkout -b "$BRANCH" origin/${WORKFLOW_BRANCH}
  ```

**If no workflow was returned AND the issue is standalone** (no `child-of`):

Resolve the base branch in this order:
1. `BASE_BRANCH` argument if provided.
2. The nearest non-epic `child-of` ancestor's active branch (`lineage-branch`).
3. The default dev branch.

```bash
if [ -n "<BASE_BRANCH>" ]; then
  BASE=<BASE_BRANCH>
else
  BASE=$(devwatch --repo "$REPO" resolve-issue-base --issue <ISSUE> --drift-pull)
fi
BRANCH=$(devwatch --repo "$REPO" workflow-branch-name --issue <ISSUE> --prefix feat)
git fetch origin && git checkout -b "$BRANCH" origin/${BASE}
```

After creating or checking out the branch, record it (use `--run-id` if available, fall back to `--issue`):
```bash
devwatch --repo "$REPO" agent-update --run-id <RUN_ID> --branch "<your-branch-name>"
```
If no RUN_ID was provided:
```bash
devwatch --repo "$REPO" agent-update --issue <ISSUE> --branch "<your-branch-name>"
```

When the run belongs to a workflow step, `agent-update --branch` also updates `workflow_steps.branch` and `workflows.current_branch` — no separate call needed.

## Intelligence (what you decide)

1. Read the issue and acceptance criteria. Understand what the feature should do.
2. Assess complexity. If the scope is too large for a single branch, break into sub-issues.
3. Implement the feature following the patterns from your reference implementation.
4. Write tests for every new function/endpoint.
5. Run tests.

## Wrap up

After implementation is complete and tests pass:

1. Commit and push:
```bash
git add <changed-files>
git commit -m "feat(scope): <description> (closes #<ISSUE>)"
git push -u origin <your-branch-name>
```

2. Read `docs:general/github-writing` — banned tokens, no personal data, per-artifact skeletons. Apply to every title, body, and comment below.

3. Record completion (use `--run-id` if available, fall back to `--issue`):
```bash
devwatch --repo "$REPO" agent-update \
  --run-id <RUN_ID> \
  --status ready_for_review \
  --summary "<one-line summary of what you built>" \
  --files "<comma-separated changed files>" \
  --commits "$(git rev-parse HEAD)"
```

4. Post completion comment to GitHub issue:
```bash
devwatch --repo "$REPO" agent-comment \
  --issue <ISSUE> \
  --body "## Feature Complete\n\n**Summary**: <what you built and why>\n**Branch**: <branch-name>\n**Files**: <changed files>\n\nReady for review."
```

## Boundary

This command stops after committing and pushing. Do NOT create a PR. Tell the user to review the branch, then run `/submit-pr`.
