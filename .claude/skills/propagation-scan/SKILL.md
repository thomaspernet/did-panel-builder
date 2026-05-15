---
description: "Scan the codebase for sites where the current diff's change could apply elsewhere. Files child issues, never edits inline."
---

After a fix or feature lands, find every other site where the same change could apply — and file each site as its own tracked unit of work. File-only: never edits code, never commits, never opens a PR.

## Mandatory reads — do this first

Run:

    devwatch --repo "$REPO" doc-read --skill propagation-scan --display

The output contains every doc you must read; treat it as if you opened each file directly. Do not proceed with the skill body until done. The mandatory-reads include the authoritative `propagation-scan` rule — that doc is loaded once here and referenced (not re-read) throughout the rest of this skill.

## Parse arguments

- `$ARGUMENTS` = `"42"` → ISSUE=42, RUN_ID=(none), BASE_BRANCH=(none), HEAD_SHA=(none), CAP=5
- `$ARGUMENTS` = `"42 --run 7"` → ISSUE=42, RUN_ID=7
- `$ARGUMENTS` = `"42 --run 7 --base-branch feat/364-x"` → ISSUE=42, RUN_ID=7, BASE_BRANCH=feat/364-x
- `$ARGUMENTS` = `"42 --run 7 --base-branch local-dev-next --head 9f3a2b1"` → ISSUE=42, RUN_ID=7, BASE_BRANCH=local-dev-next, HEAD_SHA=9f3a2b1
- `$ARGUMENTS` = `"42 --cap 3"` → CAP=3 (overrides the default top-5 cap)

`--head <sha>` is set by the dispatcher when the action fires inside an `epic_integration` chain (#1916): by the time this skill runs, `merge-to-epic` and `delete-branch` have already destroyed the feature branch, so a working-tree diff would be empty. The flag pins the diff to the implement run's tip — the same diff every time, regardless of where HEAD now sits.

## Detect repo

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
```

Pass `--repo "$REPO"` to every `devwatch` and `gh` command.

## 1. Validate branch

```bash
git branch --show-current
```

If RUN_ID or HEAD_SHA is present, skip branch-name validation — the dispatcher resolved the diff coordinates and the working tree is no longer authoritative. Otherwise require the branch to start with `fix/<ISSUE>-`, `feat/<ISSUE>-`, `refactor/<ISSUE>-`, `chore/<ISSUE>-`, or `docs/<ISSUE>-`. If not, stop and tell the user:

> "Current branch `<branch>` does not belong to issue #<ISSUE>. Check out the correct branch first."

## 2. Apply the rule

The mandatory-reads block already loaded the authoritative `propagation-scan` rule — the source of truth on when propagation applies, when to skip, the file-only boundary, the cap, and the human gate. Evaluate the current diff against it; if the rule says "skip," emit `status: skipped` with the reason and stop.

## 3. Gather the diff

```bash
BASE="${BASE_BRANCH:-$(devwatch --repo "$REPO" branches dev)}"
if [ -n "$HEAD_SHA" ]; then
  git diff "origin/${BASE}...${HEAD_SHA}"
else
  git diff "origin/${BASE}...HEAD"
fi
```

When `--head <sha>` is supplied the diff is pinned to that SHA — deterministic across re-runs and immune to checkout state. The implement run's tip SHA is recorded on the `agent_runs` row by `/fix-issue` / `/feat-issue`; the dispatcher resolves it at trigger time and passes it here.

If the diff is empty, purely cosmetic (formatting or import reorder), or limited to a dependency bump (`uv.lock`, `package-lock.json`, `package.json` deps), emit `status: skipped` with a reason and stop.

## 4. Identify candidate patterns

Read the diff. Identify each change that might propagate and classify it into one of:

- **new_helper** — a function, class, hook, arrow-const, or interface was added that might replace inline duplication elsewhere.
- **new_pattern** — a reusable shape (error handling, logging, validation, pagination) appears in two or more added sites and was not present before.
- **perf_fix** — a loop replaced by a batch / bulk / parallel call, a blocking call made async, or a comprehension replacing an appending loop.
- **bugfix_shape** — a defensive guard (None / empty / bounds), a corrected conditional, an off-by-one boundary fix.

For each candidate, note:

- A **one-line summary** of the change.
- The **originating evidence** — `file:line` in the diff.
- The **class** — which of the four kinds above.
- The **search targets** — glob patterns narrowing where to scan (`**/*.py`, `server/**`, etc.).

If no candidate fits any of the four kinds, emit `status: skipped` with reason `"no propagation-firing patterns in diff"` and stop.

## 5. Scan the codebase

For each candidate:

1. Grep the repository for sibling sites using `rg` (ripgrep), scoped to the candidate's search targets:
   ```bash
   rg -n --glob '<search_target>' '<signal keyword or regex>'
   ```
2. Filter the hits:
   - **Drop** hits inside files the current diff already modified (the originating work).
   - **Drop** hits inside vendor / generated / lockfile paths.
   - **Drop** hits that already use the new helper / pattern (import check or call-site match).
3. **Rank** remaining hits by confidence — higher rank for matches in the same module / area, the same language, and lines whose shape matches the candidate's class.

Diff-scoped rule: **only patterns derived from the current diff.** Do not sweep the codebase for unrelated patterns.

## 6. Cap and present

Truncate to the top `CAP` opportunities (default `5`). If the raw count exceeds the cap, record the overflow for the summary.

Print the list, one entry per opportunity:

```
## Propagation opportunities — 4 of 12 (cap: 5)

1. new_helper — format_currency
   Originating site: server/billing/invoice.py:142
   Candidate sites:
     - server/reports/invoice.py:88
     - dashboard/src/components/price-tag.tsx:28
   Proposed action: extract call sites to use format_currency

2. bugfix_shape — None guard on user lookup
   Originating site: server/routers/admin.py:67
   Candidate sites:
     - server/routers/reports.py:98
     - server/services/user_service.py:211
   Proposed action: apply the same None-guard pattern
```

## 7. Human gate

**Do not** file issues until the human confirms. Present the list and wait.

- **Approves all** → proceed with the full list.
- **Approves a subset** → proceed with the selected subset only.
- **Rejects** → emit `status: skipped` with reason `"user declined proposed opportunities"` and stop. Not a failure.

Agent-to-agent confirmation is not sufficient for this skill. Filing issues has larger blast radius than writing a rule; the human gate is load-bearing.

## 8. Execute — file issues

1. Apply the GitHub-writing rules from the mandatory-reads block (banned tokens, no personal data, per-artifact skeletons) to every title, body, and comment below.

2. For each approved opportunity, create the issue. Pass exactly one `--parent` — the scan target from `$ARGUMENTS`:

```bash
devwatch --repo "$REPO" create-issue \
  --type feature \
  --title "propagation: <one-line summary>" \
  --body "<markdown body>" \
  --area <area> \
  --priority <P2-medium|P3-low> \
  --parent <ISSUE> \
  --run-id <RUN_ID> \
  --no-claim-run
```

`--no-claim-run` is mandatory here. This skill files several issues
against one `--run-id`; without the flag each `create-issue` would
overwrite the run's `github_issue`/`summary` and the run row would end
up pointing at the last child instead of the scan target. The run's
terminal status/summary is owned by the step-9 `agent-update` below —
written once, not re-stamped per child.

**Title format:** every propagation issue title is `propagation: <one-line summary>` — the `propagation:` prefix exactly once, then the candidate's one-line summary. No other prefix, no descriptive-only title.

**Parentage — pass only the scan target; `create-issue` derives the epic edge.** Every propagation issue carries `--parent <ISSUE>` — the scan target — and that is the **only** `--parent` you pass. When the scan target is itself a child of an epic, `create-issue` walks the scan target's `child-of` chain in code and writes the `child-of: <epic>` edge for you, so the issue groups beside the epic's other children instead of sitting one hop below them (#2095). You do **not** resolve the epic, and you do **not** pass it as a second `--parent` — the epic edge is a deterministic property of issue creation, not something the skill assembles per run.

Do **not**:

- pass any `--parent` other than the scan target — not the resolved epic (`create-issue` adds it), not a non-epic intermediate ancestor, not a second epic deeper in the chain,
- pass the workflow root (or any workflow the scan target is a step in) as a `--parent` — the workflow is execution metadata, not issue-tree parentage,
- omit `--parent <ISSUE>` — the scan-target link is always present.

The scan target records *why* the issue exists; its epic records *where the work groups*. Those are the only two `child-of` links a propagation issue may carry — and `create-issue` is the single thing that assembles them.

Issue body template:

```markdown
## Propagation candidate

Surfaced by `/propagation-scan` on #<ISSUE>.

**Originating change:** `<file:line>` — `<commit SHA>`
**Candidate site:** `<file:line range>`
**Class:** `<new_helper | new_pattern | perf_fix | bugfix_shape>`
**Proposed action:** <one-line summary>

## Evidence

<three-line window of the grep match>

## Why this is tracked, not inlined

The originating PR is scoped to #<ISSUE>. This site is a candidate for the same treatment but belongs in its own branch and review. See the propagation-scan rule.
```

**Priority** defaults to `P3-low` (opportunistic, not blocking). Escalate to `P2-medium` when the originating kind is `bugfix_shape` — a bug pattern that recurs is higher-signal than a helper opportunity.

**Area** is inferred from the candidate path: `server/` → `backend`, `dashboard/` → `frontend`, `src/` (CLI roots) → `cli`, infra paths → `infrastructure`.

After each `create-issue`, note the returned issue number — you need them for the summary.

The `child-of` set is **not** something the skill assembles or self-verifies any more. `create-issue` writes the scan-target edge and — when the scan target has an epic — the epic edge, both in code (#2095). There is no per-run conditional for an agent to skip and no blind re-read for it to rubber-stamp: a missing or wrong `child-of` set is now a `create-issue` bug, caught by that command's own tests, not a propagation-scan responsibility.

## 9. Summary comment

Post a single summary comment on the parent `<ISSUE>`:

```bash
devwatch --repo "$REPO" agent-comment \
  --issue <ISSUE> \
  --body "## Propagation scan — <N> issues filed

- #<N1> — <summary>
- #<N2> — <summary>
- #<N3> — <summary>

Cap: <CAP>. Overflow: <K>.
See the propagation-scan rule."
```

## 10. Record completion

Update the agent-run trace (use `--run-id` if available, otherwise `--issue`):

```bash
devwatch --repo "$REPO" agent-update \
  --run-id <RUN_ID> \
  --status completed \
  --summary "Filed <N> propagation issues under #<ISSUE>"
```

For skipped runs (empty diff, cosmetic change, dependency bump, no firing pattern, user declined), still call `agent-update` with `--status completed` and prefix the summary with `Skipped —` so the dashboard can distinguish:

```bash
devwatch --repo "$REPO" agent-update \
  --run-id <RUN_ID> \
  --status completed \
  --summary "Skipped — <reason>"
```

## Boundary

- **File-only.** Never edits code, never commits, never opens a PR.
- **Diff-scoped.** Only patterns derived from the current diff. No full-repo audits.
- **Capped.** Top `CAP` opportunities (default 5). Overflow counted, not filed.
- **Human-gated.** The user approves the opportunity list before any issue is created.
- **No cross-skill writes.** Does not write rules, docs, or code. `/issue-to-rule` handles rules; `/add-documentation` handles docs.
- **Never creates an epic.** The skill files **flat** `child-of` children of the scan target. It never passes `--epic`, never files an umbrella / epic issue, and never re-roots the filed children under a new parent. If the candidate count exceeds the cap, the overflow is **counted in the summary** (step 9) — it is never absorbed by inventing an epic to hold the extra issues.
- **Never touches a pre-existing issue.** The skill only **creates new** flat children of the scan target. It never modifies, re-links, re-titles, re-parents, re-labels, or closes an issue that already exists — including issues filed by an *earlier* propagation run. Each run owns only the issues it creates in step 8.
