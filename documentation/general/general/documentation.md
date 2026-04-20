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

Use the `docs:` prefix convention for references: `docs:general/principles/clean-code` resolves to the actual file path. This makes references grep-able and verifiable.

### Hierarchy

Organize docs by who needs them and when:

- **General principles** — reusable across any project. Read once, apply everywhere.
- **Project-specific** — how this particular codebase works. Read when working in this project.
- **Guides** — step-by-step instructions for specific tasks. Read when doing that task.

A developer new to the project reads: general principles → project architecture → the specific area they are working in. The docs should support this reading order.

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
