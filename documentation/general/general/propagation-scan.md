# Propagation Scan

After you land a change, ask: *is this fix or helper useful anywhere else?* Propagation scanning is the reflex that answers that question systematically — without violating scope hygiene.

Used by the `/propagation-scan` skill as its authoritative rule. Read this before running the skill, and whenever you suspect a fix generalizes.

---

## What it is

A one-pass scan over the diff you just produced. For each change that might generalize, find every other site in the codebase where the same change could apply — and **file each site as its own tracked unit of work**. You do not edit those sites now.

## Why it's separate from iteration-time refactoring

Iteration-time refactoring says "while I'm here, fix this too." Propagation scanning says "this is worth doing elsewhere — track it, do it in its own branch."

The two are in tension. Doing a fix inline at ten sites in the same PR is faster in the short term but breaks scope hygiene: the PR stops being about the issue it claims to fix, review gets harder, and regressions surface without a clean revert point. Doing nothing means the pattern exists in the codebase but only one site benefits from the insight.

Propagation scanning resolves the tension: **discover now, fix later, in tracked units**. The scan runs at the boundary between implementation and review; it writes issues, not code.

## When it fires

Run the scan when the diff introduces any of:

- **A new helper.** A function, class, component, hook, or arrow-const that might replace inline duplication elsewhere.
- **A new pattern.** A reusable shape — error handling, logging, validation, pagination — that appears in two or more added sites and was not present before.
- **A performance fix.** An N+1 loop replaced by a batch call, a blocking call made async, a comprehension replacing an appending loop.
- **A bug-fix shape.** A defensive guard (None / empty / bounds check), a conditional correction, an off-by-one boundary fix.

These are the four `kind` values the shared diff-analysis module extracts. A change that doesn't fit any of them usually does not propagate.

## When to skip

Do not run the scan when the diff is:

- **Purely local.** A typo, a comment fix, a one-off string change with no wider pattern.
- **Cosmetic.** Formatting, import reordering, renaming a single symbol without changing behaviour.
- **A dependency bump.** The code didn't change, the lock file did.
- **A revert** or **rollback.** You are restoring prior state, not advancing it.

`/propagation-scan` reports `skipped` in these cases with a reason. That is not a failure — it means the channel correctly detected no signal.

## The file-only rule

Scanning files issues. Scanning does not edit inline.

Each candidate site becomes its own issue with:

- A one-line summary
- The file and line range of the candidate site
- The originating change's evidence (commit, PR, or branch)
- A `child-of` link to the originating issue

You do not edit any of the candidate sites in the current branch. You do not open a single "fix everything" PR. Each issue is picked up through the normal `/fix-issue` or `/feat-issue` pipeline by whoever takes it, with its own review and its own test plan.

Why:

- **Scope hygiene.** The originating PR stays about its own issue.
- **Traceability.** Each propagation has its own trace — author, review, test, merge.
- **Explicit resolution.** Drive-by refactoring muddies "why was this changed?"; a tracked issue makes the *why* explicit before work starts.

## Cap

Top 5 candidate sites per run, ranked by confidence. If the diff would generate more, the skill surfaces the top 5 and lists the overflow as a summary count. This keeps the issue tracker from getting flooded by mechanical matches that a human would collapse anyway.

## Human gate

`/propagation-scan` requires human confirmation before it files issues. Filing issues has larger blast radius than writing a rule into a local file — once an issue is filed, it is visible, it is tracked, and unfiling it costs as much as filing it. Agent-to-agent verification is not sufficient for this boundary.
