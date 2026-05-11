---
description: "Open a brainstorming session — a folder under documentation/project/brainstorming/<DD-MM-YY>/<slug>/ with a frontmatter README. Optionally pre-link it to an issue."
---

Create a brainstorming session folder. Stop.

A brainstorming session is the pre-issue thinking space — the "why" that produced a feature or epic. It lives on disk under `documentation/project/brainstorming/<DD-MM-YY>/<slug>/` with a mandatory `README.md` carrying frontmatter (title, status, linked_issues). The session is a folder so it can hold many files and subfolders as the thinking grows.

This command only creates the scaffold. It does NOT open an issue. Use `/new-feature --from-brainstorm <session>` (or `/new-bug --from-brainstorm <session>`) when the brainstorming converges into something actionable.

## Mandatory reads — do this first

Run:

    devwatch --repo "$REPO" doc-read --skill new-brainstorm --display

The output contains every doc you must read; treat it as if you opened each file directly. Do not proceed with the skill body until done.

## Parse arguments

Extract slug and optional flags from `$ARGUMENTS`:
- `$ARGUMENTS` = `"dark-mode-rollout"` -> SLUG="dark-mode-rollout", EPIC=(none), ISSUE=(none), DESCRIPTION=(none)
- `$ARGUMENTS` = `"dark-mode-rollout --description 'figure out theming primitives'"` -> SLUG="dark-mode-rollout", DESCRIPTION="figure out theming primitives"
- `$ARGUMENTS` = `"dark-mode-rollout --epic 1234"` -> EPIC=1234 (pre-link to epic #1234)
- `$ARGUMENTS` = `"dark-mode-rollout --issue 1234"` -> ISSUE=1234 (pre-link to feature #1234)
- `$ARGUMENTS` = `"dark-mode-rollout --body-file /tmp/devwatch-brainstorm-body-XXX.json"` -> BODY_FILE="/tmp/..."

`--epic` and `--issue` are mutually exclusive — a session pre-links to at most one issue at creation time. Use `/link-brainstorm <session> <issue>` later to add more.

SLUG must be kebab-case (`[a-z0-9-]+`). The CLI rejects anything else.

**If `--body-file <PATH>` is present:** read the file with your Read tool — it is a JSON object with `{title, body}` keys. Use those as the starter README's title and body. Do NOT paste the file contents into Bash; read it with the Read tool.

## Detect repo

Determine the target repository from the current working directory:

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
```

Pass `--repo "$REPO"` to every `devwatch` command to ensure the correct repo is targeted.

## Intelligence (what you decide)

1. Pick a good slug. Kebab-case, short, descriptive (`dark-mode-rollout`, not `dm` or `dark_mode`). Reflects the topic, not the conclusion.
2. Author a starter README body. Three sections, each one sentence to start — the human will expand each one when they open the file:
   - `## Why` — the question or pain that triggered the session.
   - `## Open Questions` — the unknowns the session needs to resolve.
   - `## What's next` — the rough shape of the output (single feature? epic? scrapped idea?).
3. Pick a title for the frontmatter. Title-case, short — what a reader scanning a list of sessions would scan for.
4. If `--description` was passed inline, weave it into the `## Why` section.

Pass the starter README content via a `--body-file` JSON file rather than `--body` on the command line so newlines round-trip cleanly:

```bash
BODY_FILE=$(mktemp -t devwatch-brainstorm-body-XXXX.json)
cat > "$BODY_FILE" <<'JSON'
{
  "title": "<title>",
  "body": "## Why\n\n<one sentence>\n\n## Open Questions\n\n- <bullet>\n\n## What's next\n\n<one sentence>\n"
}
JSON
```

## Execution

```bash
devwatch --repo "$REPO" new-brainstorm "$SLUG" \
  --body-file "$BODY_FILE" \
  [--epic $EPIC | --issue $ISSUE]
```

Add `--epic <N>` OR `--issue <N>` only if the user passed one (never both — they are mutually exclusive). The CLI:

- Creates `documentation/project/brainstorming/<DD-MM-YY>/<SLUG>/README.md` with frontmatter (title, status=draft, linked_issues=[N] when pre-linked, else `[]`).
- When `--epic` or `--issue` is set, appends `brainstorm: documentation/project/brainstorming/<DD-MM-YY>/<SLUG>` under the issue body's `Links:` block.
- Fails cleanly if the folder already exists — overwriting a session is never an accident.

Delete the body-file after the CLI succeeds:

```bash
rm -f "$BODY_FILE"
```

## Boundary

This command creates the session scaffold. It does NOT open an issue and does NOT write code. Report the absolute session path and ask: *"Want to expand it? Open `<path>/README.md`. When the thinking converges, run `/new-feature --from-brainstorm <session>` (or `/new-bug --from-brainstorm <session>`)."*

When you launched from a `--body-file`, delete the file after `devwatch new-brainstorm` succeeds:

```bash
rm -f "$BODY_FILE"
```
