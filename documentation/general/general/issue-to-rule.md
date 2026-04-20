# Issue to Rule

When a resolved issue reveals a **class of mistake** — not a single incident — capture the underlying constraint as a persistent rule so the same mistake does not recur. Rule feedback is about text, not code: the fix lives in the code, the **lesson** lives in `.claude/rules/` or a principle doc.

Used by the `/issue-to-rule` skill as its authoritative rule. Read this before running the skill, and whenever an issue feels like "we'll hit this again."

---

## What a rule is

A rule is a one-entry constraint that a future agent or reviewer can apply without reading the original incident. It has:

- A **one-line statement** of the constraint (declarative, 2nd-person for `_shared_docs/general/`; file-referencing imperative for project `.claude/rules/`).
- A **`Why:`** line — the reason the rule exists. Usually a past incident, a stated preference, or a guarantee the framework enforces.
- A **`How to apply:`** line — when and where the rule kicks in, concrete enough that a reader can judge edge cases.

A rule is **not** a bug fix, a todo, a retrospective, or a summary of one PR. Those belong in the code, the issue tracker, or the PR description.

## When rule extraction applies

Extract a rule when **all three** hold:

- The diff fixes a generalizable class — a missed guard, a wrong pattern, a repeated mistake — not a one-off typo or a purely local refactor.
- At least **two concrete instances** of the class exist. They can be different files, different sessions, or the same author repeating the mistake. One occurrence is iteration feedback; two is structural.
- An agent or reviewer in the future would plausibly violate the rule again without the written constraint. If the rule is obvious from the code itself, the code is enough.

When any of the three fails, skip. Iteration feedback (the fix) is sufficient.

## When to skip

- **Single-instance fixes.** One file, one session, no prior history of the same mistake.
- **Incident-specific details.** "Don't merge on Fridays" is a preference, not a rule about the code.
- **Purely cosmetic or stylistic changes.** Handled by a formatter, not a rule.
- **Dependency bumps, generated-file updates, lockfile changes.** Nothing generalizable.
- **Temporary workarounds.** If the fix itself is marked as a workaround, the rule would be "don't do this," which is weaker than the comment next to the workaround already is.

## The four operations

A rule write is one of:

- **`add`** — no existing rule covers this class. Write a new entry in the right file.
- **`update`** — an existing rule is right in spirit but imprecise. Tighten the wording, cite the new instance, keep the entry in place.
- **`supersede`** — an existing rule is wrong or obsoleted by a broader rule. Write the new entry, leave a **breadcrumb** on the old one (`⚠ superseded by <anchor>`) so anyone reading history can trace it. Do not delete the old entry.
- **`flag-stale`** — an existing rule references code or flow that no longer exists. Surface the drift to a human via a comment on the triggering issue. Do **not** auto-delete or auto-rewrite the rule. Stale detection is a signal, not a mandate.

`remove` is not an autonomous operation. Rule pruning is a human decision — the strongest deprecation the skill performs is `supersede`.

## Where a rule lives

| Claim | Target | Voice |
|---|---|---|
| Applies to any project adopting the framework | `_shared_docs/general/` (or equivalent portable docs) | declarative, 2nd-person |
| Project-wide, any area, always loaded | `.claude/rules/critical.md` | file-referencing, imperative |
| Domain-scoped (backend / frontend / CLI / infra) | `.claude/rules/<domain>.md` | file-referencing, imperative |
| Language pattern or architecture idiom | `documentation/general/principles/{python,typescript}/*` | declarative, 2nd-person |

If you cannot pick a tier confidently, the rule is probably not general enough to extract. Skip.

## Verifier gate

Before writing, run the rule draft through the [Rules checklist](rules-checklist) — accuracy, completeness, consistency, structure, operations. Every item is a yes/no question. If any item fails, revise the draft or skip.

The verifier is mechanical. An agent running the skill is allowed to apply it without human review for `add` and `update`; `supersede` and `flag-stale` should surface the intent to a human before writing.

## Cross-skill coupling

A failed `/check-code-quality` report that does not match any existing rule is a prime candidate for `/issue-to-rule`. The failing report's reasons are the payload — they carry the instance list and the class of miss. This is the one permitted coupling between review-lineup skills: no direct calls, no shared state, just report text the next skill can read.

## Cap

There is no hard cap on rule writes per run, but writing more than one rule per issue usually means the issue covered unrelated concerns. Split the work if the draft wants to write three rules at once.
