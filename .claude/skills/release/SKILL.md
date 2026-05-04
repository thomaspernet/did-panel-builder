---
description: "Create a release from the dev branch to production."
---

Ship a release: staging PR, production PR, GitHub release.

## Parse arguments

Extract optional run ID from `$ARGUMENTS`:
- `$ARGUMENTS` = `""` → RUN_ID=(none)
- `$ARGUMENTS` = `"--run 7"` → RUN_ID=7

If RUN_ID is present, forward it as `--run-id` to all `devwatch --repo "$REPO" release` calls.

## Detect repo

Determine the target repository from the current working directory:

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
```

Pass `--repo "$REPO"` to every `devwatch` command to ensure the correct repo is targeted.

## Intelligence (what you decide)

1. Read commits since the last tag. Group by type (feat, fix, refactor, other).
2. Determine the version tag (patch vs minor vs major).

## Execution

The CLI auto-detects which step to run. Pass `--tag vX.Y.Z` on every call — the
staging step uses it to bump `pyproject.toml`, `dashboard/package.json`, and
`src-electron/package.json` on the dev branch before opening the staging PR,
so the bump rides the cascade through staging → main:

```bash
devwatch --repo "$REPO" release --step auto --tag vX.Y.Z --notes "<release notes>" --run-id <RUN_ID>
```

Omit `--run-id` if no RUN_ID was parsed from arguments.

Possible outcomes:
- **staging** -- bumps version files, creates staging PR (dev -> staging). Wait for CI + merge.
- **staging-tag** -- creates a staging tag (when `tag_staging: true` in repo config). Wait for build if `build: true`.
- **production** -- creates production PR (staging -> main). Wait for CI + merge.
- **wait** -- a release PR is already open or a build is pending. Report its status.
- **done** -- all branches in sync. Nothing to release.

After production PR is merged, run the same command again to tag:

```bash
devwatch --repo "$REPO" release --step auto --tag vX.Y.Z --run-id <RUN_ID>
```

Each step is idempotent. Running `/release` again after a partial release picks up where it left off — the bump step skips silently when files already match the target version.

## Boundary

Does NOT build or deploy. Deployment is handled by CI/CD.
