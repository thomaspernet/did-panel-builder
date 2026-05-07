---
mandatory_for:
  skills: [add-documentation]
---

# Documentation Checklist

Run through this checklist after updating documentation. Used by the `/add-documentation` skill as a quality gate.

For documentation principles, read `general/documentation.md`.
For project-specific doc mapping, read `project/core-app/documentation.md`.

Doc references in this checklist use bare paths relative to the documentation root. The audience contract — which docs a given skill must read — lives in each doc's `mandatory_for:` frontmatter and is resolved by `devwatch doc-read --skill <name> --display`. See `general/mandatory-for-schema.md`.

---

## Accuracy

- [ ] Every statement in the doc matches the current code. No stale references to renamed files, removed features, or changed patterns.
- [ ] File paths referenced in the doc exist. Grep for them if unsure.
- [ ] Code examples compile / are syntactically valid. No pseudo-code presented as real code.

## Completeness

- [ ] New entities are in the entity-map with paths across all layers. → `project/core-app/entity-map.md`
- [ ] New node types and relationships are in the graph entity model. → `project/core-app/architecture/graph-entity-model.md`
- [ ] New API endpoints are in the API design doc. → `project/core-app/backend/api-design.md`
- [ ] New agent tools are in the tools inventory. → `project/core-app/agents/tools.md`
- [ ] New config sections are in the configuration system doc. → `project/core-app/architecture/configuration-system.md`
- [ ] New frontend stores, query keys, or routes are in the relevant frontend doc.
- [ ] New workflow steps are in the workflows doc. → `project/core-app/backend/workflows.md`

## Consistency

- [ ] No duplicated content. If the same concept is explained in two docs, one references the other — it does not re-explain.
- [ ] Cross-references use bare paths and point to existing files. Run `devwatch doc-validate` to catch dangling frontmatter.
- [ ] Terminology matches `project/core-app/product/concepts.md` — no old names, no ad-hoc synonyms.

## Structure

- [ ] One topic per doc. If the doc covers two unrelated things, split it.
- [ ] First paragraph explains what the doc covers. Readable without context.
- [ ] No filler, no hedging, no marketing language. Direct and concise.
- [ ] Tables for structured data (properties, mappings, inventories). Prose for explanations.

## Product Tier

User-facing docs in `product/` have a different audience than technical docs — non-technical readers — so they need their own checks. See `general/documentation.md` for per-area writing guidance.

- [ ] `product/overview.md` still reflects current product scope. Adding, removing, or repositioning a user-facing capability means updating *what the product does* and *who it's for*. → `product/overview.md`
- [ ] New user-facing features are documented in `product/usage.md` — getting started, workflows, examples, or troubleshooting as appropriate. If a user can do something new, they should be able to find out how. → `product/usage.md`
- [ ] Renamed or removed user-facing features are updated or removed from `product/` docs. No references to capabilities that no longer exist or names users no longer see.
- [ ] No internal jargon, module names, file paths, class names, or code-level detail in `product/` docs. If a sentence requires reading the code to understand, rewrite it in plain language.
- [ ] Terminology in `product/` matches what users actually see in the UI, CLI, or output — not internal entity names. If the UI says "project" but the code calls it `Workspace`, the product doc says "project."

## Rules Integration

- [ ] If a new doc was created, does its `mandatory_for:` frontmatter list every skill that must read it? Run `devwatch doc-read --skill <name> --display` to confirm the doc shows up where expected.
- [ ] If a doc was moved or renamed, are all references across all repos updated? Bare paths are grep-able — search for the old path.
