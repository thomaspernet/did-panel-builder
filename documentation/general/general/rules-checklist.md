# Rules Checklist

Run through this checklist before adding or updating a rule. Every item is a yes/no question. Used by the `/issue-to-rule` skill as a mechanical verifier gate — if any item fails, the rule does not ship.

For rule-authoring principles, read [Writing rules](vibe-coding/rules). For the documentation equivalent, see [Documentation checklist](documentation-checklist).

---

## Accuracy

- [ ] The generalization claim is defensible: the rule describes a **class** of mistake or missed pattern, not a single instance. One occurrence is not yet a rule.
- [ ] Every file, path, or identifier cited in the rule body exists. Grep for each one.
- [ ] The rule holds up under dissent cases: list 2-3 hypothetical scenarios where it might over-fire, and confirm the rule still gives the right call.
- [ ] The evidence that prompted the rule (diff, past incident, failed review) is summarized in the rule body or linked from it. A rule with no citable trigger is a suggestion, not a rule.

## Completeness

- [ ] The target file matches the tier policy:
  - `general/` (or `_shared_docs/general/`) — declarative, portable, applies to any project
  - `.claude/rules/critical.md` (project-level) — always-loaded, project-specific guardrails
  - `.claude/rules/<domain>.md` — scope-loaded, applies when touching `<domain>` files
- [ ] Every `docs:` reference in the rule resolves to an existing file, or to a path the same change commits to create.
- [ ] If the rule adds a new artifact (doc, CLI command, endpoint), the corresponding doc/checklist is listed in the mutation set.

## Consistency

- [ ] No conflict with an existing rule. Grep the rule files (`.claude/rules/**`, `_shared_docs/general/**`) for neighbouring keywords before writing; if you find an overlap, either **update** the existing rule or **supersede** it rather than adding a parallel one.
- [ ] No subsumption. If a higher-tier rule already covers this case (e.g. `clean-code.md` already says "no magic strings"), do not duplicate it at a lower tier with narrower wording.
- [ ] Voice matches the target tier:
  - `general/` → declarative, 2nd-person ("Do X. Never Y.")
  - Project `.claude/rules/` → file-referencing, imperative ("Read `documentation/project/X.mdx` before touching `Y`.")
- [ ] Terminology matches the project's concept doc. No ad-hoc synonyms.

## Structure

- [ ] One rule per entry. If the rule has two independent claims, split them.
- [ ] `**Why:**` line present — the reason the rule exists (usually a past incident, a stated preference, or a framework property the rule enforces).
- [ ] `**How to apply:**` line present — when and where the rule kicks in, concretely enough that a reader can judge edge cases instead of blindly applying it.
- [ ] Frontmatter format valid for the target file format (YAML for `.md`/`.mdx`, heading-based for plain rule files). Existing neighbours parse — yours should too.
- [ ] No filler, no hedging. A rule that starts with "try to" or "where possible" is not a rule.

## Operations

- [ ] Operation is declared: `add`, `update`, `supersede`, or `flag-stale`.
  - `add` — a new rule; the target file did not cover this case.
  - `update` — an existing rule's wording, evidence, or scope changes. Reference the prior entry.
  - `supersede` — the existing rule is wrong or obsoleted; cite what it is replaced with.
  - `flag-stale` — the existing rule references an artifact that no longer exists. Flag for follow-up; do not delete autonomously.
- [ ] If `supersede` or `flag-stale`, the prior rule is left with a visible breadcrumb (pointer to the superseding rule, or a `⚠ stale:` tag with the reason).
