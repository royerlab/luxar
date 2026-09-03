<!--
Thanks for contributing. Delete any section that does not apply — a short PR
with a clear "why" is better than a long one filled in out of obligation.
-->

## What this changes

<!-- The behaviour before and after. -->

## Why

<!--
The part reviewers cannot reconstruct from the diff. If it fixes an issue,
"Closes #123" here.
-->

## How it was verified

<!--
The commands you actually ran and what they said, rather than the ones that
should pass. If you added a guard or a test, say how you confirmed it fails
without the fix — a test that cannot go red is not evidence.
-->

## Checklist

- [ ] `make test-fast` (or the scoped subset covering this change) passes
- [ ] `make check-all` is clean — note it **reformats** the tree, so prefer the
      read-only targets (`make lint-python`, `make type-check-python`,
      `make check-typescript`) when others are working in the same checkout
- [ ] A changelog fragment exists at `changelog.d/<PR-number>.md`, if this is
      user-visible. Do **not** edit `CHANGELOG.md` directly.
- [ ] READMEs and docs updated if behaviour or an interface changed
