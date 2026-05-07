---
mandatory_for:
  universal: true
---

# Documentation

How to write and maintain documentation. These principles apply to any project.

---

## What Documentation Is For

Documentation serves two audiences: humans onboarding to the project, and AI agents that need to understand the codebase before modifying it. Both need the same thing — accurate, findable, current information about how the system works and why it was built that way.

Documentation is not a formality. It is the mechanism through which knowledge survives across sessions, across developers, and across time. An undocumented decision will be re-made differently by the next person who encounters the same problem.

## What Belongs in Documentation

Document the things that cannot be derived from reading the code:

- **Why** something is designed a certain way (the code shows what, not why)
- **How components connect** (the code shows individual files, not the system)
- **Domain concepts** (the code uses terms like "organize group" — what does that mean?)
- **Contracts between layers** (what the API promises, what the service expects)
- **Decisions and constraints** (why this database, why this architecture, why not the obvious alternative)

Do not document:

- How a specific function works (that should be clear from the code itself)
- Step-by-step tutorials for basic operations (the framework and conventions docs cover patterns)
- Temporary state or in-progress work (that belongs in issues or commits)

## Structure

Every doc should have a clear scope — one topic, one file. If a doc covers two unrelated things, split it. If two docs cover the same thing, merge them.

A doc should be readable without context. A developer opening the file for the first time should understand what it covers from the first paragraph. No preamble, no history of how it evolved — just what the reader needs to know now.

### Cross-Referencing

Docs reference each other, never duplicate each other. If concept X is explained in doc A, doc B links to doc A — it does not re-explain X.

Reference docs by their bare path under the documentation root (e.g. `general/clean-code.md`). The audience contract — which docs a given skill must read — is declared in each doc's `mandatory_for:` frontmatter and resolved by `devwatch doc-read --skill <name> --display`. See `general/mandatory-for-schema.md` for the schema; `devwatch doc-validate` enforces it.

### Hierarchy

Docs live under `documentation/` in three top-level areas. Each area has a distinct audience and purpose — keep them separate.

- **`general/`** — cross-project principles, conventions, and language guides (clean code, testing, Python patterns, TypeScript patterns, checklists). Reusable across any codebase.
- **`project/`** — developer-facing docs for *this* codebase: architecture, layer boundaries, key directories, design decisions, trade-offs.
- **`product/`** — user-facing docs: what the product does, who it's for, and how to use it. Written for non-technical readers.

Reading order for someone new to the project:

1. `general/` — how we write code in general (only the parts relevant to the area you will touch).
2. `project/` — how *this* codebase is structured and why.
3. `product/` — what the product is and who uses it, if the work needs that context.

A developer fixing a bug typically needs `general/` + `project/`. Someone writing release notes or user-facing copy needs `product/`. An AI agent onboarding to the repo should read in the same order, scoped to the area it is about to modify.

## Per-Area Writing Guidance

Each area has a different audience. What belongs in one does not belong in the others.

### `general/`

- **Audience**: developers and AI agents, across any project using these templates.
- **Scope**: principles, patterns, conventions, checklists that hold regardless of the specific codebase.
- **Tone**: direct, prescriptive, example-driven. State the rule, show a short example, explain *why*.
- **Avoid**: references to this project's directories, entities, or business logic. If a doc only makes sense in one codebase, it belongs in `project/`, not here.

### `project/`

- **Audience**: developers and AI agents working in this specific codebase.
- **Scope**: architecture, module boundaries, data flow, the real names of entities and services, decisions specific to this project and the trade-offs behind them.
- **Tone**: concrete and specific. Name real files, real modules, real tables. Explain *why* this codebase is shaped this way.
- **Avoid**: restating general principles already in `general/` — link to them. Avoid user-facing framing ("what the product does") — that belongs in `product/`.

### `product/`

- **Audience**: non-technical readers — end users, stakeholders, new hires before they touch the code.
- **Scope**: what the product is, who it's for, the problem it solves, how to use it, common workflows, examples, troubleshooting.
- **Tone**: plain language. Short sentences. Concrete outcomes ("upload a file and get a summary"), not implementation ("the ingest pipeline writes to the blob store").
- **Avoid**: internal jargon, module names, class names, file paths, API signatures, database terms, code snippets beyond what a user would paste into a UI. If a reader needs to know the codebase to understand a sentence, rewrite the sentence.

## Tone

Direct and concise. No filler, no hedging ("perhaps," "it might be useful to"), no marketing language. State what is true and what to do. If something is uncertain, say so explicitly.

Use present tense. "Services receive dependencies via constructor injection" — not "services should receive" or "services will receive."

Use second person for instructions. "Read the entity-map before adding a new entity" — not "the developer should read."

## Maintenance

Documentation rots when the code it describes changes but the doc does not. The solution is not periodic review — it is updating docs as part of every code change.

The pipeline is:

```
Code change → /add-documentation → verify doc is still accurate → update if not
```

A doc that is wrong is worse than no doc. It teaches the next developer (or agent) the wrong pattern. When you find a stale doc, fix it immediately — do not leave it for later.

### Signals That a Doc Needs Updating

- You added a new entity, tool, endpoint, or config section
- You renamed or moved something that a doc references
- You changed a pattern that a doc describes
- You read a doc and it did not match what you found in the code

## One Source of Truth

Every piece of knowledge lives in one place. Docs reference each other — they do not copy content. If the same information appears in two docs, one of them will eventually be wrong.

This applies to:
- Checklists (centralized, not scattered across guides)
- Architecture descriptions (one doc per topic, not repeated in every guide)
- Rules (in the rules files, referenced by docs, not duplicated)
- Configuration (documented once in the config doc, not re-explained in every service doc)
