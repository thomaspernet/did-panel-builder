# GitHub Writing

Rules for anything an agent writes into GitHub — issue titles, issue bodies, PR titles, PR bodies, and comments. Applies to both human-triggered and autonomous runs.

Anything written here is **public**, **permanent**, and **read without session context**. Treat every message as something a future external contributor will open cold, months from now, with no knowledge of the run that produced it.

---

## Audience

Assume the reader has:

- No access to the agent's prompt, skill, or tool transcript.
- No knowledge of the internal workflow that led to the artifact.
- No relationship with the author — they are a contributor landing on the repo through a search result.

Every issue, PR, and comment must stand on its own. If a sentence only makes sense to someone who was in the session, rewrite it or delete it.

## Never Mention

These belong inside the agent loop, not on GitHub. Strip them before writing:

- **Slash commands** — `/feat-issue`, `/fix-issue`, `/submit-pr`, `/review`, `/release`, any `/<name>`. The reader does not have these commands and cannot run them.
- **Skill names** — `feat-issue`, `propagation-scan`, `check-code-quality`, `add-documentation`. Skills are internal plumbing.
- **Run IDs** — `run 325`, `RUN_ID=7`, `agent run #42`. Tracking identifiers from the orchestration layer.
- **Internal phase names** — `impl`, `quality`, `pr`, `docs`, `release`, `cleanup`, `workflow step`. Names from the action pipeline.
- **Agent self-references** — "the agent", "I", "Claude", "this run", "I'll…", "I've implemented…". Write in third person about the change, not about the process that produced it.
- **Tool names and orchestration plumbing** — `devwatch`, `agent-update`, `agent-comment`, `lineage-context`. Readers do not invoke these.

If the agent's workflow needs to communicate a next step (e.g. "branch is ready, submit a PR next"), that signal belongs in the orchestration system (run status, structured summary, dashboard), **not** in a public comment.

## No Personal Data

GitHub issues and PRs are indexed by search engines and mirrored by archives. Never paste:

- **Real names** — "Thomas asked", "Alice pointed out". Use a role: *the reporter*, *the reviewer*, *the author*.
- **Handles, emails, phone numbers** — `@someone`, `x@example.com`. Reference the role or the existing @-mention in the issue metadata.
- **Customer names, tenant identifiers, account IDs** — even when they appear in logs the agent read. Redact to a placeholder or describe the shape (`<customer-id>`, "a tenant with more than 10k rows").
- **Quotes from Slack, email, DMs, private docs** — paraphrase the substance instead. The original source is not reachable by the reader and citing it leaks context.
- **Internal URLs** — staging dashboards, local paths, VPN-only links. If a reader cannot reach it from a fresh browser, do not link it.

When in doubt, describe the role and the observation, not the person or the source.

## Tone

- **Third person, present tense.** "The endpoint returns 500 when the payload is empty" — not "I saw that when I sent an empty payload I got a 500".
- **Neutral and factual.** No hype ("this is great"), no hedging ("hopefully this works"), no apologies ("sorry for the delay").
- **Direct.** State the change, the cause, or the ask. Skip preamble.
- **Specific.** Name files, symbols, behaviors. Avoid vague phrases like "some improvements" or "various fixes".

## Per-Artifact Skeletons

These are the shapes. Keep them short — a reader decides in seconds whether to read further.

### Issue title

One line, under ~80 characters. States the problem or the ask, not the fix.

- Good: `Empty payload to /ingest returns 500 instead of 400`
- Good: `Add pagination to the members list endpoint`
- Bad: `bug` — not specific.
- Bad: `Fix the bug where the endpoint crashes (found during the quality run)` — mentions internal phase, describes the fix rather than the problem, too long.

### Issue body

```markdown
## Description

<What is broken or missing, and the observable impact. One short paragraph.>

## Acceptance criteria

- [ ] <Concrete, testable outcome>
- [ ] <Concrete, testable outcome>

## Affected areas

- `<path/or/module>` — <why it is affected>

## Notes

<Optional: constraints, prior art, decisions already taken. Omit if empty.>
```

Acceptance criteria are the contract. Each one should be something a reviewer can tick off by reading the diff or running a command. "Works well" is not a criterion; "returns 400 with body `{"error": "..."}` on empty payload" is.

### PR title

Conventional-commit style, one line:

```
<type>(<scope>): <imperative summary> (closes #<issue>)
```

- `<type>` — `feat`, `fix`, `refactor`, `docs`, `chore`, `test`, `perf`.
- `<scope>` — directory or subsystem (`cli`, `bundles`, `rules`). Optional if the change spans the whole repo.
- `<summary>` — imperative mood, lowercase, no trailing period. Under ~60 characters.

Examples:

- `feat(cli): expose per-doc descriptions via describe command (closes #11)`
- `fix(bundles): stop sorting categories case-insensitively (closes #47)`
- `docs: add GitHub writing guardrails (closes #12)`

### PR body

```markdown
## Summary

<1–3 bullets describing what changed and why. Link the issue.>

## Test plan

- [ ] <Command or check the reviewer can run>
- [ ] <Behavior the reviewer can verify>
```

The summary answers "what does this PR do and why should I merge it". The test plan answers "how do I convince myself it works". Keep both scannable. If the change is trivial (one-line doc fix), a single summary line is fine; drop the test plan section.

### Completion comment

Posted by the agent when work on a branch is done and the artifact is ready for review. Describe **the artifact**, not the process:

```markdown
## Feature complete

**Summary**: <one line on what was built and why>
**Branch**: `<branch-name>`
**Files**: `<comma-separated paths>`

Ready for review.
```

Do not include next-step instructions aimed at another agent run (no "run the PR skill next", no "invoke the review command"). The orchestration layer routes the next step; the comment is for the human reviewer.

## Review Before Posting

Before any `gh issue create`, `gh pr create`, `gh issue comment`, or equivalent, re-read the draft and check:

1. No slash commands, skill names, run IDs, or phase names.
2. No real names, handles, emails, customer data, or internal-only URLs.
3. No first-person agent voice.
4. The artifact stands on its own for a reader with zero session context.
5. Title and body match the skeletons above.

If any check fails, rewrite before posting. A banned-token grep is a reasonable last-mile guard, but it does not replace reading the draft.
