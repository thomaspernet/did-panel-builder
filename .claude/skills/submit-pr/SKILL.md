---
description: "Submit the current branch as a PR."
---

Wrap up a branch: commit, push, create PR, record trace.

GitHub writing rules: `docs:general/github-writing`.

## Parse arguments

Extract issue number and optional run ID from `$ARGUMENTS`:
- `$ARGUMENTS` = `""` → ISSUE=(none), RUN_ID=(none)
- `$ARGUMENTS` = `"42"` → ISSUE=42, RUN_ID=(none)
- `$ARGUMENTS` = `"42 --run 7"` → ISSUE=42, RUN_ID=7
- `$ARGUMENTS` = `"--run 7"` → ISSUE=(none), RUN_ID=7

If RUN_ID is present, forward it as `--run-id` to the `devwatch submit-pr` call.

## Detect repo

Determine the target repository from the current working directory:

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
```

Pass `--repo "$REPO"` to every `devwatch` command to ensure the correct repo is targeted.

## Checkout the correct branch

If ISSUE is present, verify the current branch belongs to that issue:

```bash
git branch --show-current
```

The branch name must start with `fix/<ISSUE>-`, `feat/<ISSUE>-`, `refactor/<ISSUE>-`, `chore/<ISSUE>-`, `docs/<ISSUE>-`, or `ci-fix/<ISSUE>-`.

If the current branch does **not** match, find and checkout the correct branch:

```bash
git branch -a | grep -E "(fix|feat|ci-fix)/<ISSUE>-"
```

Checkout the matching branch. If no matching branch exists, **stop** — tell the user no branch exists for issue #ISSUE.

If ISSUE is not present, stay on the current branch.

## Prerequisites

Run `/check-code-quality` if not already done. Do not submit a PR that hasn't passed the quality gate.

Check the issue history for context (run `devwatch issue-history --help` for all options):
```bash
devwatch --repo "$REPO" issue-history <ISSUE>
```

## Intelligence (what you decide)

1. Review the diff: `git diff --stat` and `git diff`. Check: no secrets, no debug logs.
2. Write a commit message: conventional prefix, explains the WHY.
3. Write a PR summary: what changed and how to verify.
4. Choose labels: area labels (`area:backend`, `area:frontend`, etc.). Labels are best-effort — the CLI retries without them if they don't exist in the target repo.

## Execution

1. Read `docs:general/github-writing` — banned tokens, no personal data, per-artifact skeletons. Apply to every title, body, and comment below.

```bash
devwatch --repo "$REPO" submit-pr \
  --message "<your commit message>" \
  --summary "<your PR summary>" \
  --label "<label1>" --label "<label2>" \
  --run-id <RUN_ID>
```

Omit `--run-id` if no RUN_ID was parsed from arguments.

The CLI handles everything deterministically: branch detection, commit, push, PR creation, sync.

### PR body auto-close contract

The CLI prepends `Closes #<issue>` as the first line of the rendered PR body so GitHub auto-closes the linked issue on merge. Do **not** put `Closes #N` (or `Fixes` / `Resolves` variants) in `--message` — that lands in the PR title, where GitHub ignores it. Keep the magic word in the body, where the CLI puts it.

## Epic-rooted workflows

When the current issue is the root of an epic-rooted workflow (i.e. the workflow's `root_issue_number` equals this issue and `is_epic = 1`), the dashboard's Submit Workflow button and the `submit-workflow-pr` action route to the epic PR backend (`devwatch submit-epic-pr <epic>`), not a per-child PR. The routing is driven by `workflows.root_issue_number` on the backend — there is no "step 1 is the epic" heuristic. For child issues of an epic, `/submit-pr` still performs the local merge into the epic branch as described in the framework's epic pipeline; only the final epic PR goes through `submit-epic-pr`.

### Child behaviour by branch strategy

For a child issue of an epic the CLI routes on the child's workflow step strategy:

- **`SAME`** — the child branch is merged locally into `epic/<N>-<slug>` and the child issue stays **open**. It auto-closes when the epic-level workflow PR merges.
- **`EPIC_INTEGRATION`** — the child branch is merged locally into `epic/<N>-<slug>` and the child issue is **closed immediately** (short comment linking the merge commit). No per-child GitHub PR is opened. The epic-level PR runs CI once, for the whole integration branch.

Both paths skip `gh pr create` — only the epic-level PR opens later via `submit-epic-pr`.

## Boundary

This command does NOT merge the PR. Report the PR URL and stop.
