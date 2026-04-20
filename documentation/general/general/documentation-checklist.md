# Documentation Checklist

Run through this checklist after updating documentation. Used by the `/add-documentation` skill as a quality gate.

For documentation principles, read [Documentation](documentation).
For project-specific doc mapping, read `docs:project/core-app/documentation`.

---

## Accuracy

- [ ] Every statement in the doc matches the current code. No stale references to renamed files, removed features, or changed patterns.
- [ ] File paths referenced in the doc exist. Grep for them if unsure.
- [ ] Code examples compile / are syntactically valid. No pseudo-code presented as real code.

## Completeness

- [ ] New entities are in the entity-map with paths across all layers. → `docs:project/core-app/entity-map`
- [ ] New node types and relationships are in the graph entity model. → `docs:project/core-app/architecture/graph-entity-model`
- [ ] New API endpoints are in the API design doc. → `docs:project/core-app/backend/api-design`
- [ ] New agent tools are in the tools inventory. → `docs:project/core-app/agents/tools`
- [ ] New config sections are in the configuration system doc. → `docs:project/core-app/architecture/configuration-system`
- [ ] New frontend stores, query keys, or routes are in the relevant frontend doc.
- [ ] New workflow steps are in the workflows doc. → `docs:project/core-app/backend/workflows`

## Consistency

- [ ] No duplicated content. If the same concept is explained in two docs, one references the other — it does not re-explain.
- [ ] Cross-references use `docs:` prefix and point to existing files.
- [ ] Terminology matches `docs:project/core-app/product/concepts` — no old names, no ad-hoc synonyms.

## Structure

- [ ] One topic per doc. If the doc covers two unrelated things, split it.
- [ ] First paragraph explains what the doc covers. Readable without context.
- [ ] No filler, no hedging, no marketing language. Direct and concise.
- [ ] Tables for structured data (properties, mappings, inventories). Prose for explanations.

## Rules Integration

- [ ] If a new doc was created, is it referenced in the relevant `.claude/rules/` file so the agent knows to read it?
- [ ] If a doc was moved or renamed, are all `docs:` references across all repos updated?
